import torch.nn as nn
import MinkowskiEngine as ME
import torch

from mask3d_models.model import Model
from mask3d_models.modules.common import ConvType, NormType, conv, get_norm, sum_pool
from mask3d_models.modules.resnet_block import BasicBlock, Bottleneck
from mask3d_models.position_embedding import PositionEmbeddingCoordsSine

class ResNetBase(Model):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    DILATIONS = (1, 1, 1, 1)
    OUT_PIXEL_DIST = 1
    # HAS_LAST_BLOCK = False
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, in_channels, out_channels, config, D=3, out_fpn=False, **kwargs):
        assert self.BLOCK is not None
        assert self.OUT_PIXEL_DIST > 0

        super().__init__(in_channels, out_channels, config, D, **kwargs)
        self.out_fpn = out_fpn

        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, config, D):
        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if config is not None:
            bn_momentum = config.bn_momentum
            conv1_kernel_size = config.conv1_kernel_size
        else:
            bn_momentum = 0.02
            conv1_kernel_size = 5

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        dilations = self.DILATIONS
        self.inplanes = self.INIT_DIM
        self.conv1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn1 = get_norm(
            NormType.BATCH_NORM,
            self.inplanes,
            D=self.D,
            bn_momentum=bn_momentum,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = sum_pool(kernel_size=space_n_time_m(2, 1), stride=space_n_time_m(2, 1), D=D)

        self.layer1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[0], 1))
        self.atten1 = nn.MultiheadAttention(self.inplanes, 8, batch_first=True)
        self.pos1 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.inplanes))

        self.layer2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[1], 1))
        self.atten2 = nn.MultiheadAttention(self.inplanes, 8, batch_first=True)
        self.pos2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.inplanes))

        self.layer3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[2], 1))
        self.atten3 = nn.MultiheadAttention(self.inplanes, 8, batch_first=True)
        self.pos3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.inplanes))

        self.layer4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            stride=space_n_time_m(2, 1),
            dilation=space_n_time_m(dilations[3], 1))
        self.atten4 = nn.MultiheadAttention(self.inplanes, 8, batch_first=True)
        self.pos4 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.inplanes))

        self.final = conv(
            self.PLANES[3] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            D=D,
        )
        self.pos_func = PositionEmbeddingCoordsSine(pos_type="sine", d_pos=64, gauss_scale=1.0, normalize=True).cuda()

    def mink_atten(self, in_field, raw_grid_coords, atten_layer, pos_layer):
        ###
        # coords_field = ME.SparseTensor(features=raw_grid_coords, coordinate_manager=in_field.coordinate_manager,
        #                                coordinate_map_key=in_field.coordinate_map_key, device=in_field.device)
        # voxel_coords = coords_field.decomposed_features
        voxel_feats = in_field.decomposed_features
        max_voxel_num = 0
        for i in range(len(voxel_feats)):
            if voxel_feats[i].shape[0] > max_voxel_num:
                max_voxel_num = voxel_feats[i].shape[0]
        mask = []
        for i in range(len(voxel_feats)):
            if voxel_feats[i].shape[0] < max_voxel_num:
                mask.append(torch.cat((torch.ones(voxel_feats[i].shape[0]), torch.zeros(max_voxel_num - voxel_feats[i].shape[0]))).unsqueeze(0))
                voxel_feats[i] = torch.cat((voxel_feats[i], torch.zeros((max_voxel_num - voxel_feats[i].shape[0], voxel_feats[i].shape[1])).cuda())).unsqueeze(0)
                # voxel_coords[i] = torch.cat((voxel_coords[i], torch.zeros((max_voxel_num - voxel_coords[i].shape[0], voxel_coords[i].shape[1])).cuda())).unsqueeze(0)
            else:
                mask.append((torch.ones(voxel_feats[i].shape[0])).unsqueeze(0))
                voxel_feats[i] = voxel_feats[i].unsqueeze(0)
                # voxel_coords[i] = voxel_coords[i].unsqueeze(0)
        voxel_feats, mask = torch.cat(voxel_feats), ~torch.cat(mask).bool()
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  ##[bs, N, N]
        attn_mask = attn_mask.cuda()
        ##
        # voxel_coords = torch.cat(voxel_coords)
        # with autocast(enabled=False):
        #     mins, maxs = voxel_coords.min(dim=1)[0], voxel_coords.max(dim=1)[0]
        #     cossin_pos = self.pos_func(voxel_coords.float(), input_range=[mins, maxs]).permute((0, 2, 1))  # Batch, Dim, queries
        # pos_encoding = pos_layer(cossin_pos)

        # atten_feat = atten_layer(query=voxel_feats + pos_encoding, key=voxel_feats + pos_encoding, value=voxel_feats, attn_mask=attn_mask.repeat_interleave(8, dim=0))[0]
        atten_feat = atten_layer(query=voxel_feats, key=voxel_feats, value=voxel_feats, attn_mask=attn_mask.repeat_interleave(8, dim=0))[0]
        atten_out = []
        for i in range(len(voxel_feats)):
            atten_out.append(atten_feat[i][torch.where(~mask[i])[0]])
        atten_out = torch.cat(atten_out) + in_field.F
        out = ME.SparseTensor(features=atten_out, coordinate_manager=in_field.coordinate_manager, coordinate_map_key=in_field.coordinate_map_key, device=in_field.device)
        return out

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type=NormType.BATCH_NORM,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D,
                ),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x0):
        feature_maps = []
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        feature_maps.append(x)

        x = self.layer1(x)
        # x = self.mink_atten(x, x0.F, self.atten1, self.pos1)
        feature_maps.append(x)
        x = self.layer2(x)
        x = self.mink_atten(x, x0.F, self.atten2, self.pos2)
        feature_maps.append(x)
        x = self.layer3(x)
        x = self.mink_atten(x, x0.F, self.atten3, self.pos3)
        feature_maps.append(x)
        x = self.layer4(x)
        x = self.mink_atten(x, x0.F, self.atten4, self.pos4)
        feature_maps.append(x)
        # x = self.final(x)
        # return x
        if not self.out_fpn:
            return self.final(x)
        else:
            return self.final(x), feature_maps

class MyResNet14(ResNetBase):
    BLOCK = BasicBlock
    PLANES = (32, 64, 128, 256)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


class STResNetBase(ResNetBase):

    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
        super().__init__(in_channels, out_channels, config, D, **kwargs)


class STResNet14(STResNetBase, ResNet14):
    pass


class STResNet18(STResNetBase, ResNet18):
    pass


class STResNet34(STResNetBase, ResNet34):
    pass


class STResNet50(STResNetBase, ResNet50):
    pass


class STResNet101(STResNetBase, ResNet101):
    pass


class STResTesseractNetBase(STResNetBase):
    CONV_TYPE = ConvType.HYPERCUBE


class STResTesseractNet14(STResTesseractNetBase, STResNet14):
    pass


class STResTesseractNet18(STResTesseractNetBase, STResNet18):
    pass


class STResTesseractNet34(STResTesseractNetBase, STResNet34):
    pass


class STResTesseractNet50(STResTesseractNetBase, STResNet50):
    pass


class STResTesseractNet101(STResTesseractNetBase, STResNet101):
    pass
