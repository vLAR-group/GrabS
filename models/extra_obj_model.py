import time
import torch
import torch.nn as nn
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.VN_lib.vec_sim3.vec_dgcnn_atten import VecDGCNN_att
from models.VN_lib.vec_sim3.vec_dgcnn_atten_patch import VecDGCNN_att_patch
from models.VN_lib.vec_sim3 import vec_dgcnn_atten_vae, vec_dgcnn_atten_vae_so3, vec_dgcnn_atten_vae_efem_so3, vec_dgcnn_atten_vae_efem_so3_unet, vec_dgcnn_atten_vae_efem_so3_unet_innermasked, vec_dgcnn_fc_vae_efem_so3
from models.VN_lib.vec_sim3.vec_dgcnn import VecDGCNN, VNN_ResnetPointnet
from models.VN_lib.vec_sim3.vec_point_convocc import VNN_ResnetPointnet_local
from models.VN_lib.vec_sim3.vec_dgcnn_atten_unet2 import VecDGCNN_att_unet
from models.VN_lib.implicit_func.onet_decoder import DecoderCat, DecoderCatBN
from pointnet2.pointnet2 import three_nn, three_interpolate, knn
from models.VN_lib.implicit_func.decoder_inner import DecoderInnerUNet, DecoderInner
from models.VN_lib.implicit_func.layers_equi import VNLinear
from models.VN_lib.implicit_func.pointnet import ResnetPointnet
from models.VN_lib.vec_sim3.vec_dgcnn_partseg import VecDGCNN_seg
from torch_scatter import scatter_mean
import torch.nn.functional as F
from mask3d_models.resnet_atten import MyResNet14
from models.layers import ResnetBlockFC, ResnetBlockFCBN, ResnetBlockConv1d
from models.backbone_module import Pointnet2Backbone, Pointnet2Backbone_tiny, Pointnet2Backbone_tiny_noatten
import math
# ----------------------------------------------------------------------------------------------------

class Diffusion_war(nn.Module):
    def __init__(self, diffusion_net, cond_net, VAE):
        super(Diffusion_war, self).__init__()
        self.diffusion_net = diffusion_net
        self.cond_net = cond_net
        self.VAE = VAE


class NDF(nn.Module):
    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)
        self.last_actvn = nn.Tanh()

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self,x):# [bs, 256, 256]
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
        ## p: [bs, num_off_surface_points, 3]
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        feature_0 = F.grid_sample(f_0, p, padding_mode='border', align_corners=True)
        feature_1 = F.grid_sample(f_1, p, padding_mode='border', align_corners=True)
        feature_2 = F.grid_sample(f_2, p, padding_mode='border', align_corners=True)
        feature_3 = F.grid_sample(f_3, p, padding_mode='border', align_corners=True)
        feature_4 = F.grid_sample(f_4, p, padding_mode='border', align_corners=True)
        feature_5 = F.grid_sample(f_5, p, padding_mode='border', align_corners=True)
        feature_6 = F.grid_sample(f_6, p, padding_mode='border', align_corners=True)

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return  out

    def forward(self, p, x):
        out = self.decoder(p, *self.encoder(x))
        return out


class SDF(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SDF, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)
        self.last_actvn = nn.Tanh()

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self,x):# [bs, 256, 256]
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
        ## p: [bs, num_off_surface_points, 3]
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        feature_0 = F.grid_sample(f_0, p, padding_mode='border', align_corners=True)
        feature_1 = F.grid_sample(f_1, p, padding_mode='border', align_corners=True)
        feature_2 = F.grid_sample(f_2, p, padding_mode='border', align_corners=True)
        feature_3 = F.grid_sample(f_3, p, padding_mode='border', align_corners=True)
        feature_4 = F.grid_sample(f_4, p, padding_mode='border', align_corners=True)
        feature_5 = F.grid_sample(f_5, p, padding_mode='border', align_corners=True)
        feature_6 = F.grid_sample(f_6, p, padding_mode='border', align_corners=True)

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        # net = self.actvn(self.fc_out(net))
        net = self.last_actvn(self.fc_out(net))
        out = net.squeeze(1)

        return  out

    def forward(self, p, x):
        out = self.decoder(p, *self.encoder(x))
        return out

class SDF_simple_raw(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SDF_simple_raw, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128 in_channel can also be 1
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16
        # self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        # self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8
        # self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        # self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128) * 1 + 3
        # self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        # self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        # self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        # self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        # self.conv3_1_bn = nn.BatchNorm3d(128)
        # self.conv4_1_bn = nn.BatchNorm3d(128)

    def encoder(self, x):# [bs, 256, 256]
        ### x: [bs, N, 3]
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        # net = self.maxpool(net)
        #
        # net = self.actvn(self.conv_3(net))
        # net = self.actvn(self.conv_3_1(net))
        # net = self.conv3_1_bn(net)
        # f_5 = net
        # net = self.maxpool(net)
        #
        # net = self.actvn(self.conv_4(net))
        # net = self.actvn(self.conv_4_1(net))
        # net = self.conv4_1_bn(net)
        # f_6 = net

        return f_0, f_1, f_2, f_3, f_4#, f_5, f_6

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4):
    # def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
        ## p: [bs, num_off_surface_points, 3], have to be [-1, 1], [zyx]
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        #
        # # feature extraction
        feature_0 = F.grid_sample(f_0, p, padding_mode='border', align_corners=True)
        feature_1 = F.grid_sample(f_1, p, padding_mode='border', align_corners=True)
        feature_2 = F.grid_sample(f_2, p, padding_mode='border', align_corners=True)
        feature_3 = F.grid_sample(f_3, p, padding_mode='border', align_corners=True)
        feature_4 = F.grid_sample(f_4, p, padding_mode='border', align_corners=True)
        # feature_5 = F.grid_sample(f_5, p, padding_mode='border', align_corners=True)
        # feature_6 = F.grid_sample(f_6, p, padding_mode='border', align_corners=True)

        # here every channel corresponds to one feature.

        # features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=1)  # (B, features, 1,7,sample_num)
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4), dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1).permute(0, 2, 1)  # (B, featue_size, samples_num)
        # features = torch.cat((features, f_global.unsqueeze(1).repeat(1, features.shape[1], 1)), dim=-1)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = (self.fc_out(net))
        # out = net.squeeze(1)
        out = net.squeeze(-1)
        return out

    def forward(self, p, x):
        out = self.decoder(p, *self.encoder(x))
        return out


class SDF_simple(nn.Module):
    def __init__(self, hidden_dim=128, res=128):
        super(SDF_simple, self).__init__()
        self.res = res
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128 in_channel can also be 1
        self.conv_0 = nn.Conv3d(16*1, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32*1, 64, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64*1, 128, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128*1, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128*1, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8

        # self.conv_4 = nn.Conv3d(128, 128, 4)  # out: 128 ->4
        # self.conv_4_1 = nn.Conv3d(128, 128, 4)

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 +128 + 0) * 1 + 3
        # feature_size = 128 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)
        self.meanpool = nn.AvgPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)

    def get_grid_feats(self, xyz, feat, res=64):# xyz: [B, N, c], feat:[B, N, C]
        c_dim = feat.shape[-1]
        xyz = self.normalize_3d_coordinate(xyz)## [-0.5, 0.5] --> [0, 1]
        xyz = (xyz * res).long()
        index = xyz[:, :, 0] + res * (xyz[:, :, 1] + res * xyz[:, :, 2])
        index = index[:, None, :]
        # scatter grid features from points
        fea_grid = feat.new_zeros(xyz.size(0), c_dim, res**3) ## [N, C, res*3]
        feat = feat.permute(0, 2, 1)
        fea_grid = scatter_mean(feat, index, out=fea_grid)
        # point_feat = fea_grid.gather(dim=2, index=index)
        fea_grid = fea_grid.reshape(xyz.size(0), c_dim, res, res, res)
        return fea_grid #[B, C, res*3]

    def sample_grid_feature(self, query_point, grid_feats, neighbors=None):
        # query_point : [bs, N, 3]
        ### first make it be zyx
        # query_point = query_point[:, :, [2, 1, 0]]
        p_nor = self.normalize_3d_coordinate(query_point.clone()) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float() ## [bs, N, 1, 1, 3] -0.5 -- 0.5
        # p_nor = p_nor.unsqueeze(1).unsqueeze(1).float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # vgrid = torch.cat([vgrid + d for d in self.displacments], dim=3)## [bs, N, 1, 1, 3], so not sure dim should be 2 or 3?
        # vgrid = p_nor
        # acutally trilinear interpolation if mode = 'bilinear'
        query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(-1).squeeze(-1)
        # query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(2).squeeze(2)
        return query_feats.transpose(1, 2)
        # return query_feats.transpose(1, 2).reshape(query_point.shape[0], query_point.shape[1], -1)

    def normalize_3d_coordinate(self, p):
        p_nor = p
        p_nor = p_nor + 0.5  # range (0, 2)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1.0 - 1e-4
        # if p_nor.max() >= 2:
        #     p_nor[p_nor >= 2] = 2 - 1e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor#/2

    def encoder(self, x):# [bs, 256, 256]
        ### x: [bs, N, 3]
        # xyz_grid = self.get_grid_feats(x, x.permute(0, 2, 1), res=256)
        # bbox_min, bbox_max = x.min(1, keepdim=True).values, x.max(1, keepdim=True).values ##[bs, 1, 3]
        # center, scale = (bbox_min + bbox_max)/2, (bbox_max - bbox_min).max(-1).values.unsqueeze(-1)+1e-6 ## [bs, 1]
        # x = (x-center)/scale
        # x = x/scale
        xyz_grid = self.get_grid_feats(x, torch.ones_like(x)[:, :, 0].unsqueeze(-1), res=self.res)
        # xyz_grid = self.get_grid_feats(x, x, res=256)
        x = (xyz_grid>0).float()
        # x = xyz_grid
        #
        # x = x.float().unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128
        # net = torch.cat((net, net.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True).repeat(1, 1, net.shape[2], net.shape[3], net.shape[4])), dim=1)
        # net = self.meanpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64
        # net = self.meanpool(net)  # out 128
        # net = torch.cat((net, net.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True).repeat(1, 1, net.shape[2], net.shape[3], net.shape[4])), dim=1)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)
        # net = self.meanpool(net)  # out 128
        # net = torch.cat((net, net.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True).repeat(1, 1, net.shape[2], net.shape[3], net.shape[4])), dim=1)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)
        # net = self.meanpool(net)  # out 128
        # net = torch.cat((net, net.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True).repeat(1, 1, net.shape[2], net.shape[3], net.shape[4])), dim=1)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)
        # net = self.meanpool(net)  # out 128
        # net = torch.cat((net, net.mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True).repeat(1, 1, net.shape[2], net.shape[3], net.shape[4])), dim=1)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        # f_6 = net
        f_global = net.mean(-1).mean(-1).mean(-1)#.squeeze(-1).squeeze(-1).squeeze(-1) ## [bs, C]
        # return f_0, f_1, f_2, f_3, f_4, f_5, f_6
        return f_0, f_1, f_2, f_3, f_4, f_5, f_global

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_global):#, center, scale):
    # def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):#, center, scale):
        ## p: [bs, num_off_surface_points, 3], have to be [-1, 1], [zyx]
        p_features = p
        #
        # # p = p[:, :, [2, 1, 0]] ### make it be xyz, [-1, 1]
        # # p = (p-center)/scale ### to be [-0.5, 0.5]
        feature_0 = self.sample_grid_feature(p, f_0)
        feature_1 = self.sample_grid_feature(p, f_1)
        feature_2 = self.sample_grid_feature(p, f_2)
        feature_3 = self.sample_grid_feature(p, f_3)
        feature_4 = self.sample_grid_feature(p, f_4)
        feature_5 = self.sample_grid_feature(p, f_5)
        # feature_6 = self.sample_grid_feature(p, f_6)

        feature_6 = f_global.unsqueeze(1).repeat(1, p.shape[1], 1)
        #
        # # here every channel corresponds to one feature.
        #
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=-1)  # (B, N, C)
        features = torch.cat((features, p_features), dim=-1)  # (B, N, C)
        # global_features = feature_6.mean(1, keepdim=True)
        # features = torch.cat((p, f.unsqueeze(1).repeat(1, p.shape[1], 1)), dim=-1)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        sign = self.fc_sign(net).sigmoid()
        net = (self.fc_out(net))#*scale
        # out = net.squeeze(1)
        out = net.squeeze(-1)
        return out, sign.squeeze(-1)

    def forward(self, p, x):
        out, sign = self.decoder(p, *self.encoder(x))
        # out = self.decoder(p, self.encoder(x))
        return out, sign



class SDF_simple2(nn.Module):
    def __init__(self, hidden_dim=128, res=64):
        super(SDF_simple2, self).__init__()
        self.res = res

        self.fc_in = nn.Linear(4, 32)
        self.bn_in = nn.BatchNorm1d(32)

        self.conv_0 = nn.Conv3d(32*2, 32, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 64 ->m.p. 32

        self.conv_1 = nn.Conv3d(32*2, 64, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16

        self.conv_2 = nn.Conv3d(64*2, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8

        self.conv_3 = nn.Conv3d(128*2, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8 -> mp 4

        self.conv_4 = nn.Conv3d(128*2, 128, 3, padding=1, padding_mode='zeros')  # out: 4
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 4
        # self.conv_4 = nn.Conv3d(128, 128, 4)  # out: 4

        feature_size = 132#(32 + 32 + 64 + 128 + 128 + 128) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.ReLU() #nn.ELU()

        self.maxpool = nn.MaxPool3d(2)
        self.meanpool = nn.AvgPool3d(2)

        self.conv0_1_bn = nn.BatchNorm3d(32)#nn.GroupNorm(num_groups=8, num_channels=32)#
        self.conv1_1_bn = nn.BatchNorm3d(64)#nn.GroupNorm(num_groups=8, num_channels=64)#
        self.conv2_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)

    def get_grid_feats(self, xyz, feat, res=64):# xyz: [B, N, c], feat:[B, N, C]
        c_dim = feat.shape[-1]
        xyz = self.normalize_3d_coordinate(xyz)## [-0.5, 0.5] --> [0, 1]
        xyz = (xyz * res).long()
        index = xyz[:, :, 0] + res * (xyz[:, :, 1] + res * xyz[:, :, 2])
        index = index[:, None, :]
        # scatter grid features from points
        fea_grid = feat.new_zeros(xyz.size(0), c_dim, res**3) ## [N, C, res*3]
        feat = feat.permute(0, 2, 1)
        fea_grid = scatter_mean(feat, index, out=fea_grid)
        fea_grid = fea_grid.reshape(xyz.size(0), c_dim, res, res, res)
        return fea_grid #[B, C, res*3]

    def sample_grid_feature(self, query_point, grid_feats, neighbors=None):
        # query_point : [bs, N, 3]
        p_nor = self.normalize_3d_coordinate(query_point.clone()) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float() ## [bs, N, 1, 1, 3] -0.5 -- 0.5
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(-1).squeeze(-1)
        return query_feats.transpose(1, 2)

    def normalize_3d_coordinate(self, p):
        p_nor = p
        p_nor = p_nor + 0.5  # range (0, 2)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1.0 - 1e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor#/2

    def encoder(self, x):# [bs, 256, 256]
        ### x: [bs, N, 3]
        # xyz_grid = self.get_grid_feats(x, torch.ones_like(x)[:, :, 0].unsqueeze(-1), res=self.res)
        # x = (xyz_grid>0).float()
        # f_0 = x
        x_length = torch.norm(x, p=2, dim=-1, keepdim=True)
        net = self.fc_in(torch.cat((x, x_length), dim=-1))

        # net = self.fc_in(x)
        net = self.bn_in(net.transpose(1, 2)).transpose(1, 2)
        net = self.actvn(net)
        f_0 = self.get_grid_feats(x, net, res=self.res)
        pooled = torch.max(f_0, 1, keepdim=True)[0].expand(f_0.size())
        net = torch.cat((f_0, pooled), dim=1)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_global = net.mean(-1).mean(-1).mean(-1)

        # return f_0, f_2, f_3, f_4, f_5, f_global
        return f_global
    #
    # def decoder(self, p, f_0, f_2, f_3, f_4, f_5, f_global):
    #     p_features = p
    #     #
    #     feature_0 = self.sample_grid_feature(p, f_0)
    #     feature_2 = self.sample_grid_feature(p, f_2)
    #     feature_3 = self.sample_grid_feature(p, f_3)
    #     feature_4 = self.sample_grid_feature(p, f_4)
    #     feature_5 = self.sample_grid_feature(p, f_5)
    #
    #     features = torch.cat((feature_0, feature_2, feature_3, feature_4, feature_5), dim=-1)  # (B, N, C)
    #     features = torch.cat((features, f_global.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
    #
    #     net = self.actvn(self.fc_0(features))
    #     net = self.actvn(self.fc_1(net))
    #     net = self.actvn(self.fc_2(net))
    #     sign = self.fc_sign(net).sigmoid()
    #     net = F.tanh(self.fc_out(net))#*scale
    #     out = net.squeeze(-1)
    #     return out, sign.squeeze(-1)
    #
    # def forward(self, p, x):
    #     out, sign = self.decoder(p, *self.encoder(x))
    #     return out, sign
    def decoder(self, p, f_global):
        # p_features = p
        p_length = torch.norm(p, p=2, dim=-1, keepdim=True)
        p_features = torch.cat((p, p_length), dim=-1)

        features = torch.cat((f_global.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        sign = self.fc_sign(net).sigmoid()
        net = (self.fc_out(net))#*scale
        out = net.squeeze(-1)
        return out, sign.squeeze(-1)

    def forward(self, p, x):
        out, sign = self.decoder(p, self.encoder(x))
        return out, sign



class DISN(nn.Module):
    def __init__(self, hidden_dim=128, res=64):
        super(DISN, self).__init__()
        self.res = res

        self.fc_in = nn.Linear(3, 32)
        self.bn_in = nn.BatchNorm1d(32)

        self.conv_0 = nn.Conv3d(32*1, 32, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 64 ->m.p. 32

        self.conv_1 = nn.Conv3d(32*1, 64, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16

        self.conv_2 = nn.Conv3d(64*1, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8

        self.conv_3 = nn.Conv3d(128*1, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8 -> mp 4

        self.conv_4 = nn.Conv3d(128*1, 128, 3, padding=1, padding_mode='zeros')  # out: 4
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 4
        # self.conv_4 = nn.Conv3d(128, 128, 4)  # out: 4
        ### conv pooling?

        feature_size = (64 + 128 + 128) * 1 + 64
        self.fc_1 = nn.Linear(feature_size, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        self.fc_sign = nn.Linear(64, 1)
        self.actvn = nn.ReLU() #nn.ELU()

        self.query_point_encoding = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
        self.global_decoder = nn.Sequential(nn.Linear(128+64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        self.maxpool = nn.MaxPool3d(2)
        self.meanpool = nn.AvgPool3d(2)

        self.conv0_1_bn = nn.BatchNorm3d(32)#nn.GroupNorm(num_groups=8, num_channels=32)#
        self.conv1_1_bn = nn.BatchNorm3d(64)#nn.GroupNorm(num_groups=8, num_channels=64)#
        self.conv2_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)#nn.GroupNorm(num_groups=8, num_channels=128)#nn.BatchNorm3d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.displacments = torch.Tensor(displacments).cuda()

    def get_grid_feats(self, xyz, feat, res=64):# xyz: [B, N, c], feat:[B, N, C]
        c_dim = feat.shape[-1]
        xyz = self.normalize_3d_coordinate(xyz)## [-0.5, 0.5] --> [0, 1]
        xyz = (xyz * res).long()
        index = xyz[:, :, 0] + res * (xyz[:, :, 1] + res * xyz[:, :, 2])
        index = index[:, None, :]
        # scatter grid features from points
        fea_grid = feat.new_zeros(xyz.size(0), c_dim, res**3) ## [N, C, res*3]
        feat = feat.permute(0, 2, 1)
        fea_grid = scatter_mean(feat, index, out=fea_grid)
        fea_grid = fea_grid.reshape(xyz.size(0), c_dim, res, res, res)
        return fea_grid #[B, C, res*3]

    def sample_grid_feature(self, query_point, grid_feats, neighbors=None):
        # query_point : [bs, N, 3]
        ### first make it be zyx
        # query_point = query_point[:, :, [2, 1, 0]]
        p_nor = self.normalize_3d_coordinate(query_point.clone()) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float() ## [bs, N, 1, 1, 3] -0.5 -- 0.5
        # p_nor = p_nor.unsqueeze(1).unsqueeze(1).float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # vgrid = torch.cat([vgrid + d for d in self.displacments], dim=3)## [bs, N, 1, 1, 3], so not sure dim should be 2 or 3?
        # vgrid = p_nor
        # acutally trilinear interpolation if mode = 'bilinear'
        query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(-1).squeeze(-1)
        # query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(2).squeeze(2)
        return query_feats.transpose(1, 2)
        # return query_feats.transpose(1, 2).reshape(query_point.shape[0], query_point.shape[1], -1)

    def normalize_3d_coordinate(self, p):
        p_nor = p
        p_nor = p_nor + 0.5  # range (0, 2)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1.0 - 1e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor#/2

    def encoder(self, x):# [bs, 256, 256]
        ### x: [bs, N, 3]

        net = self.fc_in(x)
        net = self.bn_in(net.transpose(1, 2)).transpose(1, 2)
        net = self.actvn(net)
        f_1 = self.get_grid_feats(x, net, res=self.res)
        # pooled = torch.max(f_1, 1, keepdim=True)[0].expand(f_1.size())
        # net = torch.cat((f_0, pooled), dim=1)
        net = f_1

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64
        # pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        # net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)
        # pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        # net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        # pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        # net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)
        # pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())
        # net = torch.cat((net, pooled), dim=1)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_global = net.mean(-1).mean(-1).mean(-1)

        return f_1, f_2, f_3, f_4, f_5, f_global
    #
    def decoder(self, p, f_1, f_2, f_3, f_4, f_5, f_global):
        p_features = self.query_point_encoding(p)
        #
        # feature_1 = self.sample_grid_feature(p, f_1)
        # feature_2 = self.sample_grid_feature(p, f_2)
        feature_3 = self.sample_grid_feature(p, f_3)
        feature_4 = self.sample_grid_feature(p, f_4)
        feature_5 = self.sample_grid_feature(p, f_5)

        features = torch.cat((feature_3, feature_4, feature_5, p_features), dim=-1)  # (B, N, C)

        net = self.actvn(self.fc_1(features))
        net = self.actvn(self.fc_2(net))
        local_sdf = self.fc_out(net).squeeze(-1)

        sign = self.fc_sign(net).sigmoid()

        global_sdf = self.global_decoder(torch.cat((f_global.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)).squeeze(-1)
        out = local_sdf + global_sdf
        return out, sign.squeeze(-1)

    def forward(self, p, x):
        out, sign = self.decoder(p, *self.encoder(x))
        return out, sign


class PointNet_block(nn.Module):
    def __init__(self, in_channel, out_channel, hidden):
        super().__init__()
        self.fc_0 = nn.Linear(in_channel, hidden)
        self.fc_1 = nn.Linear(hidden, out_channel)
        self.actvn = torch.nn.LeakyReLU()
        self.shortcut = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        net = self.actvn(self.fc_0(x))
        dx = self.actvn(self.fc_1(net))
        return dx + self.shortcut(x)


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.mlp_1 = nn.Linear(3, 64)
        self.mlp_2 = nn.Linear(64, 256)
        self.mlp_3 = nn.Linear(256, 512)

        self.mlp_4 = nn.Linear(512*2, 256)
        self.mlp_5 = nn.Linear(256, 64)
        self.mlp_6 = nn.Linear(64, 9)
        # self.relu = nn.ReLU()

        self.mlp_bn1 = nn.BatchNorm1d(64)
        self.mlp_bn2 = nn.BatchNorm1d(256)
        self.mlp_bn3 = nn.BatchNorm1d(512)
        self.mlp_bn4 = nn.BatchNorm1d(256)
        self.mlp_bn5 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.mlp_bn1(self.mlp_1(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.mlp_bn2(self.mlp_2(x).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.mlp_bn3(self.mlp_3(x).transpose(1, 2)).transpose(1, 2))
        x = torch.cat((torch.max(x, 1)[0], torch.mean(x, 1)), dim=1)

        x = F.relu(self.mlp_bn4(self.mlp_4(x)))
        x = F.relu(self.mlp_bn5(self.mlp_5(x)))
        x = self.mlp_6(x)

        # iden = torch.eye(3).flatten().unsqueeze(0).repeat(batchsize, 1).cuda()
        # x = x + iden
        x = x.view(-1, 3, 3)
        return x

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss



class PointNet(nn.Module):
    def __init__(self, hidden_dim=128, res=64):
        super(PointNet, self).__init__()
        self.res = res
        self.so3 = STN3d()

        self.fc_in = nn.Linear(3, 32)
        self.block_1 = PointNet_block(32*2, 64, 64)#nn.Linear(32*2, 64)
        self.block_2 = PointNet_block(64*2, 128, 128)#nn.Linear(64*2, 128)
        self.block_3 = PointNet_block(128*2, 256, 256)#nn.Linear(128*2, 256)
        self.block_4 = PointNet_block(256*2, 512, 512)#nn.Linear(256*2, 512)

        feature_size = (512) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU() #nn.ELU()

        self.conv0_1_bn = nn.BatchNorm1d(32)
        self.conv1_1_bn = nn.BatchNorm1d(64)
        self.conv2_1_bn = nn.BatchNorm1d(128)
        self.conv3_1_bn = nn.BatchNorm1d(256)
        self.conv4_1_bn = nn.BatchNorm1d(512)

    def encoder(self, x):# [bs, 256, 256]
        ### x: [bs, N, 3]
        # x_length = torch.norm(x, p=2, dim=-1, keepdim=True)
        # net = self.fc_in(torch.cat((x, x_length), dim=-1))
        trans = self.so3(x)
        x = torch.bmm(x, trans)

        net = self.fc_in(x)
        net = self.conv0_1_bn(net.transpose(1, 2)).transpose(1, 2)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())

        net = self.block_1(torch.cat((net, pooled), dim=-1))
        net = self.conv1_1_bn(net.transpose(1, 2)).transpose(1, 2)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())

        net = self.block_2(torch.cat((net, pooled), dim=-1))
        net = self.conv2_1_bn(net.transpose(1, 2)).transpose(1, 2)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())

        net = self.block_3(torch.cat((net, pooled), dim=-1))
        net = self.conv3_1_bn(net.transpose(1, 2)).transpose(1, 2)
        pooled = torch.max(net, 1, keepdim=True)[0].expand(net.size())

        net = self.block_4(torch.cat((net, pooled), dim=-1))
        net = self.conv4_1_bn(net.transpose(1, 2)).transpose(1, 2)

        return torch.max(net, 1)[0], trans, feature_transform_regularizer(trans)

    def decoder(self, p, features, trans, trans_loss):
        p = torch.bmm(p, trans)
        p_features = p
        # p_length = torch.norm(p, p=2, dim=-1, keepdim=True)
        # p_features = torch.cat((p, p_length), dim=-1)
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        # sign = self.fc_sign(net).sigmoid()
        net = (self.fc_out(net))#*scale
        out = net.squeeze(-1)
        return out, trans, trans_loss#sign.squeeze(-1)

    def forward(self, p, x):
        out, trans, trans_loss = self.decoder(p, *self.encoder(x))
        # out = self.decoder(p, self.encoder(x))
        return out, trans, trans_loss



class PointNet2(nn.Module):
    def __init__(self, hidden_dim=256, res=64):
        super(PointNet2, self).__init__()
        self.res = res
        # self.so3 = Pointnet2Backbone(input_feature_dim=16)

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        # feature_size = (512) * 1 + 3
        # self.fc_0 = nn.Linear(feature_size, hidden_dim)
        # # self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        # # self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        # self.fc_out = nn.Linear(hidden_dim, 1)
        # self.fc_sign = nn.Linear(hidden_dim, 1)
        # self.actvn = nn.LeakyReLU()

        # self.conv0_1_bn = nn.BatchNorm1d(32)
        # self.conv1_1_bn = nn.BatchNorm1d(64)
        # self.conv2_1_bn = nn.BatchNorm1d(128)
        # self.conv3_1_bn = nn.BatchNorm1d(256)
        # self.conv4_1_bn = nn.BatchNorm1d(512)

        # self.fc_trans = nn.Linear(512 , 9)
        # self.fc_trans = nn.Linear(512 , 3)
        self.temp_feature = nn.Embedding(1, 512//2)
        # self.decode_mlp = nn.Sequential(ResnetBlockFC(hidden_dim), ResnetBlockFC(hidden_dim),
        #                                 ResnetBlockFC(hidden_dim), ResnetBlockFC(hidden_dim))
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        # trans = self.so3(x)

        # trans_feature = self.so3(x, x).mean(1) ##[B, N, C]
        # trans_feature = self.fc_trans(trans_feature)
        # # #
        # iden = torch.eye(3).flatten().unsqueeze(0).repeat(batchsize, 1).cuda()
        # trans = (trans_feature + iden).view(-1, 3, 3)
        # # trans = (trans_feature).view(-1, 3, 3)
        # # trans = trans.transpose(2, 1)
        # #
        # x = torch.bmm(x, trans).detach()
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z#, mu, log_var#, trans

    def decode(self, p, features):#, mu, log_var):#, trans):
        # p = torch.bmm(p, trans)
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        out = self.decode_mlp(features)
        return out#, mu, log_var#, trans

    def forward(self, p, x):
        # out, mu, log_var, trans = self.decoder(p, *self.encoder(x))
        out, mu, log_var = self.decode(p, *self.encode(x))
        return out, mu, log_var#, trans
    # def forward(self, p, x):
    #     batchsize = x.shape[0]
    #
    #     trans_feature = self.so3(x, x).mean(1)  ##[B, N, C]
    #     trans_feature = self.fc_trans(trans_feature)
    #     trans = trans_feature
    #     # #
    #     # iden = torch.eye(3).flatten().unsqueeze(0).repeat(batchsize, 1).cuda()
    #     # trans = (trans_feature + iden).view(-1, 3, 3)
    #     return trans


class PointNet2_cls(nn.Module):
    def __init__(self, cls_num=8):
        super(PointNet2_cls, self).__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.cls_head = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, cls_num, bias=False))

    def forward(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        logits = self.cls_head(global_feature)
        return logits



class PointNet2_VAEpatch(nn.Module):
    def __init__(self, hidden_dim=128, res=64):
        super(PointNet2_VAEpatch, self).__init__()
        self.res = res
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512+512, 512), nn.LeakyReLU(), nn.Linear(512, 128))
        self.log_var = nn.Sequential(nn.Linear(512+512, 512), nn.LeakyReLU(), nn.Linear(512, 128))

        feature_size = (128) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        # self.fc_trans = nn.Linear(512 , 9)
        self.fc_trans = nn.Linear(512 , 3)
        self.temp_feature = nn.Embedding(1, 512)
        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

        self.mlp1 = nn.Linear(4, 64)
        self.att_pooling = Att_pooling2(64+128, 128, d_f=128)

    def index_points(self, points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encoder(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        global_feature, patch_feature, patch_coord = self.backbone(x, x) ## [bs, 16, 512] [bs, 128, 128], [bs, 128, 3]
        patch_feature = torch.cat((patch_feature, global_feature.mean(1, keepdim=True).repeat(1, patch_feature.shape[1], 1)), dim=-1) ## [bs, 16, 128+512]
        mu, log_var = self.mu(patch_feature), self.log_var(patch_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            z = z_length*z/(z.norm(p=2, dim=-1, keepdim=True)+1e-6)
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z, patch_coord, mu.reshape(-1, mu.shape[-1]), log_var.reshape(-1, mu.shape[-1])
        # return z, patch_coord, mu.reshape(mu.shape[0], -1), log_var.reshape(mu.shape[0], -1)

    def decoder(self, p, features, coord, mu, log_var):
        p_features = p

        # dist, idx = knn(1, p, coord)
        # query_feat = self.index_points(features, idx.long()).squeeze(2)
        # # query_feat = torch.sum(self.index_points(features, idx.long()) * weight.unsqueeze(-1), dim=2)

        # relative_xyz = coord.unsqueeze(1).repeat(1, p.shape[1], 1, 1) - p.unsqueeze(2) ## [bs, 5k, 16, 3]
        # dist = relative_xyz.norm(p=2, dim=-1)
        # dist_recip = 1.0 / (dist + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight = dist_recip / norm
        # query_feat = torch.bmm(weight, features)

        neighor_feats = features.unsqueeze(1).repeat(1, p.shape[1], 1, 1)
        neighor_xyz = coord.unsqueeze(1).repeat(1, p.shape[1], 1, 1)
        relative_xyz = neighor_xyz - p.unsqueeze(2) ## [bs, 5k, 16, 3]
        dist = relative_xyz.norm(p=2, dim=-1)
        # xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz, p.unsqueeze(2).repeat(1, 1, relative_xyz.shape[2], 1), neighor_xyz], dim=-1)  # [bs, 5k, 16, 10]
        xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz], dim=-1)  # [bs, 5k, 16, 10]
        f_xyz = self.mlp1(xyz_set) ## [bs, 5k, 16, 64]
        f_concat = torch.cat((neighor_feats, f_xyz), dim=-1)## [bs, N, 16, 32+128]
        query_feat = self.att_pooling(f_concat, features)  # [bs, N, 128]

        features = torch.cat((query_feat, p_features), dim=-1)  # (B, N, C)
        #
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net.transpose(2, 1)).transpose(2, 1)
        out = self.fc_out(net).squeeze(-1)
        # out = F.tanh(out)
        return out, mu, log_var

    def forward(self, p, x):
        out, mu, log_var = self.decoder(p, *self.encoder(x))
        return out, mu, log_var


class PointNet2_AE(nn.Module):
    def __init__(self, hidden_dim=128, res=64):
        super(PointNet2_AE, self).__init__()
        self.res = res
        self.so3 = Pointnet2Backbone(input_feature_dim=16)

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))
        self.log_var = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))

        feature_size = (512) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        self.conv0_1_bn = nn.BatchNorm1d(32)
        self.conv1_1_bn = nn.BatchNorm1d(64)
        self.conv2_1_bn = nn.BatchNorm1d(128)
        self.conv3_1_bn = nn.BatchNorm1d(256)
        self.conv4_1_bn = nn.BatchNorm1d(512)

        # self.fc_trans = nn.Linear(512 , 9)
        self.fc_trans = nn.Linear(512 , 3)
        self.temp_feature = nn.Embedding(1, 512)
        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encoder(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        # trans = self.so3(x)
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        z = self.mu(global_feature)
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z

    def decoder(self, p, features):
        # p = torch.bmm(p, trans)
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net.transpose(2, 1)).transpose(2, 1)
        out = self.fc_out(net).squeeze(-1)
        # out = F.tanh(out)
        return out

    def forward(self, p, x):
        out = self.decoder(p, self.encoder(x))
        return out


class PointNet3(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PointNet3, self).__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))
        self.log_var = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))

        feature_size = (512) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        self.fc_trans = nn.Linear(512 , 3)
        self.fc_svd_r0 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3*128))
        self.fc_svd_r1 = nn.Linear(128, 3, bias=False)
        self.temp_feature = nn.Embedding(1, 512)
        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def SVD_R(self, global_feature):
        global_feature = self.fc_svd_r0(global_feature).reshape(global_feature.shape[0], 3, -1) ## [bs, 3, 128]
        z_so3 = self.fc_svd_r1(global_feature)#.reshape(-1,3,3)  # B,3,3
        R_pred = z_so3.transpose(2, 1)  # B,3,3
        _U, _, _Vh = torch.linalg.svd(R_pred.double())
        z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,3,3
        return z_so3.transpose(2, 1) # [bs, 3, 3]

    def encoder(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        trans = self.fc_trans(global_feature) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(trans)
        # z = self.mu(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            z = z_length*z/(z.norm(p=2, dim=-1, keepdim=True)+1e-6)
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z, mu, log_var, trans, R

    def decoder(self, p, features, mu, log_var, trans, R):
        # p = torch.bmm(p, R.transpose(2, 1))
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net.transpose(2, 1)).transpose(2, 1)
        out = self.fc_out(net).squeeze(-1)
        return out, mu, log_var, trans, R

    def forward(self, p, x):
        out, mu, log_var, trans, R = self.decoder(p, *self.encoder(x))
        return out, mu, log_var, trans, R




class PointNet4(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PointNet4, self).__init__()
        self.so3 = Pointnet2Backbone(input_feature_dim=16)
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))
        self.log_var = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 512))

        feature_size = (512) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        self.fc_trans = nn.Linear(512 , 3)
        self.fc_svd_r0 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3*128))
        self.fc_svd_r1 = nn.Linear(128, 3, bias=False)
        self.temp_feature = nn.Embedding(1, 512)
        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def SVD_R(self, global_feature):
        global_feature = self.fc_svd_r0(global_feature).reshape(global_feature.shape[0], 3, -1) ## [bs, 3, 128]
        z_so3 = self.fc_svd_r1(global_feature)#.reshape(-1,3,3)  # B,3,3
        R_pred = z_so3.transpose(2, 1)  # B,3,3
        _U, _, _Vh = torch.linalg.svd(R_pred.double())
        z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,3,3
        return z_so3.transpose(2, 1) # [bs, 3, 3]

    def encode(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        ###
        so3_feature = self.so3(x, x)[0].mean(1) ##[B, N, C]
        trans = self.fc_trans(so3_feature) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(trans)
        x = torch.bmm(x, R.transpose(2, 1))
        ###
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        # z = self.mu(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            z = z_length*z/(z.norm(p=2, dim=-1, keepdim=True)+1e-6)
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z, mu, log_var, trans, R

    def decode(self, p, features, mu, log_var, trans, R):
        p = torch.bmm(p, R.transpose(2, 1))
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net.transpose(2, 1)).transpose(2, 1)
        out = self.fc_out(net).squeeze(-1)
        return out, mu, log_var, trans, R

    def forward(self, p, x):
        out, mu, log_var, trans, R = self.decode(p, *self.encode(x))
        return out, mu, log_var, trans, R



class PointNet4_VAEpatch(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PointNet4_VAEpatch, self).__init__()
        self.so3 = Pointnet2Backbone(input_feature_dim=16)
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512+512, 512), nn.LeakyReLU(), nn.Linear(512, 128))
        self.log_var = nn.Sequential(nn.Linear(512+512, 512), nn.LeakyReLU(), nn.Linear(512, 128))

        feature_size = (128) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        # self.fc_trans = nn.Linear(512 , 9)
        self.fc_trans = nn.Linear(512 , 3)
        self.temp_feature = nn.Embedding(1, 512)
        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

        self.mlp1 = nn.Linear(4, 64)
        self.att_pooling = Att_pooling2(64+128, 128, d_f=128)

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        so3_feature = self.so3(x, x)[0].mean(1) ##[B, N, C]
        trans = self.fc_trans(so3_feature) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(trans)
        x = torch.bmm(x, R.transpose(2, 1))
        ####
        global_feature, patch_feature, patch_coord = self.backbone(x, x) ## [bs, 16, 512] [bs, 128, 128], [bs, 128, 3]
        patch_feature = torch.cat((patch_feature, global_feature.mean(1, keepdim=True).repeat(1, patch_feature.shape[1], 1)), dim=-1) ## [bs, 16, 128+512]
        mu, log_var = self.mu(patch_feature), self.log_var(patch_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            z = z_length*z/(z.norm(p=2, dim=-1, keepdim=True)+1e-6)
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z, patch_coord, mu.reshape(-1, mu.shape[-1]), log_var.reshape(-1, mu.shape[-1]), trans, R
        # return z, patch_coord, mu.reshape(mu.shape[0], -1), log_var.reshape(mu.shape[0], -1)

    def decode(self, p, features, coord, mu, log_var, trans, R):
        p = torch.bmm(p, R.transpose(2, 1))
        p_features = p
        neighor_feats = features.unsqueeze(1).repeat(1, p.shape[1], 1, 1)
        neighor_xyz = coord.unsqueeze(1).repeat(1, p.shape[1], 1, 1)
        relative_xyz = neighor_xyz - p.unsqueeze(2) ## [bs, 5k, 16, 3]
        dist = relative_xyz.norm(p=2, dim=-1)
        # xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz, p.unsqueeze(2).repeat(1, 1, relative_xyz.shape[2], 1), neighor_xyz], dim=-1)  # [bs, 5k, 16, 10]
        xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz], dim=-1)  # [bs, 5k, 16, 10]
        f_xyz = self.mlp1(xyz_set) ## [bs, 5k, 16, 64]
        f_concat = torch.cat((neighor_feats, f_xyz), dim=-1)## [bs, N, 16, 32+128]
        query_feat = self.att_pooling(f_concat, features)  # [bs, N, 128]

        features = torch.cat((query_feat, p_features), dim=-1)  # (B, N, C)
        #
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net.transpose(2, 1)).transpose(2, 1)
        out = self.fc_out(net).squeeze(-1)
        # out = F.tanh(out)
        return out, mu, log_var, trans, R

    def forward(self, p, x):
        out, mu, log_var, trans, R = self.decode(p, *self.encode(x))
        return out, mu, log_var, trans, R




class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        # self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.backbone = Pointnet2Backbone_tiny(input_feature_dim=16)

        self.fc_trans = nn.Linear(512//2 , 3)
        self.fc_svd_r0 = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 3*128))
        self.fc_svd_r1 = nn.Linear(128, 3, bias=False)

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def SVD_R(self, global_feature):
        global_feature = self.fc_svd_r0(global_feature).reshape(global_feature.shape[0], 3, -1) ## [bs, 3, 128]
        z_so3 = self.fc_svd_r1(global_feature)#.reshape(-1,3,3)  # B,3,3
        R_pred = z_so3.transpose(2, 1)  # B,3,3
        _U, _, _Vh = torch.linalg.svd(R_pred.double())
        z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,3,3
        return z_so3.transpose(2, 1) # [bs, 3, 3]

    def forward(self, x):
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        angles = self.fc_trans(global_feature) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(angles)
        # R = self.SVD_R(global_feature)
        return angles, R




class PoseNet_cls(nn.Module):
    def __init__(self):
        super(PoseNet_cls, self).__init__()
        # self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.backbone = Pointnet2Backbone_tiny(input_feature_dim=16)

        self.fc_trans = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 3))
        self.fc_svd_r0 = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 3*128))
        self.fc_svd_r1 = nn.Linear(128, 3, bias=False)

        self.cls_head = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 6))

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def SVD_R(self, global_feature):
        global_feature = self.fc_svd_r0(global_feature).reshape(global_feature.shape[0], 3, -1) ## [bs, 3, 128]
        z_so3 = self.fc_svd_r1(global_feature)#.reshape(-1,3,3)  # B,3,3
        R_pred = z_so3.transpose(2, 1)  # B,3,3
        _U, _, _Vh = torch.linalg.svd(R_pred.double())
        z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,3,3
        return z_so3.transpose(2, 1) # [bs, 3, 3]

    def forward(self, x):
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        angles = self.fc_trans(global_feature) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(angles)
        logits = self.cls_head(global_feature) ## 3 eluer angles
        return angles, R, logits




class PoseNet_cls2(nn.Module):
    def __init__(self):
        super(PoseNet_cls2, self).__init__()
        # self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.backbone = Pointnet2Backbone_tiny(input_feature_dim=16)

        self.fc_trans = nn.Sequential(nn.Linear(512//2+6, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 3))
        self.fc_svd_r0 = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 3*128))
        self.fc_svd_r1 = nn.Linear(128, 3, bias=False)

        self.cls_head = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 6))

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R.cuda()

    def SVD_R(self, global_feature):
        global_feature = self.fc_svd_r0(global_feature).reshape(global_feature.shape[0], 3, -1) ## [bs, 3, 128]
        z_so3 = self.fc_svd_r1(global_feature)#.reshape(-1,3,3)  # B,3,3
        R_pred = z_so3.transpose(2, 1)  # B,3,3
        _U, _, _Vh = torch.linalg.svd(R_pred.double())
        z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,3,3
        return z_so3.transpose(2, 1) # [bs, 3, 3]

    def forward(self, x):
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        logits = self.cls_head(global_feature) ## [bs, 6]
        angles = self.fc_trans(torch.cat((global_feature, logits), dim=-1)) ## 3 eluer angles
        R = self.euler_angles_to_rotation_matrix_batch(angles)
        return angles, R, logits







class PointNet2_inv(nn.Module):
    def __init__(self, hidden_dim=256):
        super(PointNet2_inv, self).__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        self.temp_feature = nn.Embedding(1, 512//2)
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        # return z, mu, log_var
        return z, mu

    # def decode(self, p, features, mu, log_var):
    def decode(self, p, features, mu):
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        out = self.decode_mlp(features)
        return out#, mu, log_var

    def forward(self, p, x):
        out, mu, log_var = self.decode(p, *self.encode(x))
        return out, mu, log_var


class PointNet2_wpos2(nn.Module):
    def __init__(self, hidden_dim=256, res=64):
        super(PointNet2_wpos2, self).__init__()
        self.res = res
        # self.so3 = PoseNet_cls()
        self.so3 = PoseNet()

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        self.temp_feature = nn.Embedding(1, 512//2)
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_pred_R_inv(self, x):
        self.so3.eval()
        with torch.no_grad():
            _, pred_R = self.so3(x)
            pred_R_inv = pred_R.transpose(2, 1).detach()
        return pred_R_inv

    def encode(self, x, vae_training=False):# [bs, N, 3]
        batchsize = x.shape[0]
        self.so3.eval()
        with torch.no_grad():
            _, pred_R = self.so3(x)
            pred_R_inv = pred_R.transpose(2, 1).detach()
        x = torch.bmm(x, pred_R_inv).detach()

        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        # z = self.mu(global_feature)
        z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        if vae_training:
            return z, mu, log_var, pred_R_inv
        else:
            return z, pred_R_inv

    # def decode(self, p, features, mu, log_var, pred_R_inv=torch.eye(3)[None, ...].cuda()):
    def decode(self, p, features, pred_R_inv=None):
        # if pred_R_inv is None:
        #     pred_R_inv = torch.eye(3)[None, ...].cuda().repeat(features.shape[0], 1, 1)
        p = torch.bmm(p, pred_R_inv)#.detach()

        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        out = self.decode_mlp(features)
        return out

    def forward(self, p, x):
        z, mu, log_var, pred_R_inv = self.encode(x, vae_training=True)
        out = self.decode(p, z, pred_R_inv)
        return out, mu, log_var




class PointNet2_wpos(nn.Module):
    def __init__(self, hidden_dim=256, res=64):
        super(PointNet2_wpos, self).__init__()
        self.res = res
        # self.so3 = PoseNet_cls()
        self.so3 = PoseNet()

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        self.temp_feature = nn.Embedding(1, 512//2)
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_pred_R_inv(self, x):
        self.so3.eval()
        with torch.no_grad():
            _, pred_R = self.so3(x)
            pred_R_inv = pred_R.transpose(2, 1).detach()
        return pred_R_inv

    def encode(self, x, vae_training=False):# [bs, N, 3]
        batchsize = x.shape[0]
        self.so3.eval()
        with torch.no_grad():
            _, pred_R = self.so3(x)
            pred_R_inv = pred_R.transpose(2, 1).detach()
        x = torch.bmm(x, pred_R_inv).detach()

        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        # z = self.mu(global_feature)
        z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        if vae_training:
            return z, mu, log_var, pred_R_inv
        else:
            return z, pred_R_inv

    # def decode(self, p, features, mu, log_var, pred_R_inv=torch.eye(3)[None, ...].cuda()):
    def decode(self, p, features, pred_R_inv=None):
        # if pred_R_inv is None:
        #     pred_R_inv = torch.eye(3)[None, ...].cuda().repeat(features.shape[0], 1, 1)
        p = torch.bmm(p, pred_R_inv)#.detach()

        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        out = self.decode_mlp(features)
        return out

    def forward(self, p, x):
        z, mu, log_var, pred_R_inv = self.encode(x, vae_training=True)
        out = self.decode(p, z, pred_R_inv)
        return out, mu, log_var


class Diffusion_cond(nn.Module):
    def __init__(self, hidden_dim=256, res=64):
        super(Diffusion_cond, self).__init__()
        self.so3 = PoseNet()

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        self.temp_feature = nn.Embedding(1, 512//2)
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)

    def forward(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        # self.so3.eval()
        # with torch.no_grad():
        #     _, pred_R = self.so3(x)
        #     pred_R_inv = pred_R.transpose(2, 1).detach()
        # x = torch.bmm(x, pred_R_inv).detach()

        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        z = self.mu(global_feature) + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z#, pred_R_inv


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

### if noisy_feature and condintion are both 1-D vector, no need attention
class Diffusion_net(nn.Module):
    def __init__(self, time_embedding_dim=256, max_period=1000):
        super(Diffusion_net, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.max_period = max_period
        self.time_embedding = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                                nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.fusion = nn.Sequential(nn.Linear(256*3, 256*2, bias=False), nn.LeakyReLU(), nn.Linear(256*2, 256, bias=False),
                                    nn.LeakyReLU(), nn.Linear(256, 256, bias=False))

    def forward(self, noisy_feat, cond_feat, t):
        batchsize = noisy_feat.shape[0]
        time_embedding = timestep_embedding(t, self.time_embedding_dim, self.max_period).cuda()
        return self.fusion(torch.cat((noisy_feat, cond_feat, self.time_embedding(time_embedding)), dim=-1))

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, group):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.group = group

        self.embedding = nn.Embedding(self.group*self.n_e, self.e_dim)

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def sample_gumbel(self, shape, eps=1e-10):
        U = torch.rand(shape, device="cuda")
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=1):
        g = self.sample_gumbel(logits.size())
        y = logits + 0.01*g
        return F.softmax(y / temperature, dim=-1)

    def forward(self, z):
        # """
        # Inputs the output of the encoder network z and maps it to a discrete
        # one-hot vector that is the index of the closest embedding vector e_j
        #
        # z (continuous) -> z_q (discrete)
        #
        # z.shape = (batch, channel, height, width)
        #
        # quantization pipeline:
        #
        #     1. get encoder input (B,C,H,W)
        #     2. flatten input to (B*H*W,C)
        #
        # """
        # # reshape z -> (batch, height, width, channel) and flatten
        #
        # z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened = z.view(-1, self.e_dim)
        # # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        #
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        #
        # # find closest encodings
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).cuda()
        # min_encodings.scatter_(1, min_encoding_indices, 1)
        #
        # # get quantized latent vectors
        # z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #
        # # compute loss for embedding
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        #
        # # preserve gradients
        # z_q = z + (z_q - z).detach()
        #
        # # perplexity
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        #
        # # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        ###
        ## z should be [bs, group, c_dim], self.embedding.weight [group, code_num, code_dim]
        d = (z.unsqueeze(2) - self.embedding.weight.reshape(self.group, self.n_e, self.e_dim).unsqueeze(0)).norm(p=2, dim=-1) ## [bs, group, code_num]
        if self.training:
            logits = self.gumbel_softmax_sample(-d)
        else:
            logits = (-d).softmax(2)
        # find closest encodings
        min_encoding_indices = torch.argmax(logits, dim=-1).unsqueeze(2) ### [bs, group, 1]

        # logits = (-d*30).softmax(2)
        # if self.training:
        #     dist = Categorical(logits)
        #     min_encoding_indices = dist.sample().unsqueeze(2)  ### [bs, group, 1]
        # else:
        #     min_encoding_indices = torch.argmax(logits, dim=-1).unsqueeze(2)  ### [bs, group, 1]

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.group, self.n_e).cuda() ## [bs, group, code num]
        min_encodings.scatter_(2, min_encoding_indices, 1) ##?
        min_encodings = min_encodings.unsqueeze(2) ## [bs, group, 1, code_num]
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight.reshape(self.group, self.n_e, self.e_dim).unsqueeze(0).repeat(min_encodings.shape[0],1,1,1)).squeeze(2) ## [bs, group, c_dim]
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # perplexity
        # e_mean = torch.mean(min_encodings.reshape(min_encodings.shape[0], -1), dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))/self.group
        useage = []
        for g in range(self.group):
            valid_index = min_encoding_indices[:, g]
            useage.append(len(torch.unique(valid_index))/self.n_e)
        perplexity = sum(useage)/len(useage)
        return loss, z_q, perplexity, min_encodings, min_encoding_indices




class PointNet2_VQVAE(nn.Module):
    def __init__(self, hidden_dim=128, beta=0.25):
        super(PointNet2_VQVAE, self).__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim=16)

        feature_size = (512) * 1 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        # self.fc_sign = nn.Linear(hidden_dim, 1)
        self.actvn = nn.LeakyReLU()

        self.conv0_1_bn = nn.BatchNorm1d(32)
        self.conv1_1_bn = nn.BatchNorm1d(64)
        self.conv2_1_bn = nn.BatchNorm1d(128)
        self.conv3_1_bn = nn.BatchNorm1d(256)
        self.conv4_1_bn = nn.BatchNorm1d(512)

        self.fc_trans = nn.Linear(512 , 9)

        self.temp_feature = nn.Embedding(1, 512)
        self.fc_global_feature = nn.Sequential(nn.Linear(512 , 512), nn.LeakyReLU(), nn.Linear(512, 512))

        self.decode_mlp = nn.Sequential(ResnetBlockFC(hidden_dim), ResnetBlockFC(hidden_dim),
                                        ResnetBlockFC(hidden_dim), ResnetBlockFC(hidden_dim))

        self.vector_quantization = VectorQuantizer(32, 512//16, beta, group=16)

    def encoder(self, x):# [bs, N, 3]
        batchsize = x.shape[0]
        global_feature = self.backbone(x, x).mean(1) ##[B, N, C]
        global_feature = self.fc_global_feature(global_feature) ## [bs, 512]
        embedding_loss, new_global_feature, perplexity, _, _ = self.vector_quantization(global_feature.reshape(global_feature.shape[0], 16, -1)) ## input need be [B, C, H, W]
        new_global_feature = new_global_feature.reshape(new_global_feature.shape[0], -1) + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return new_global_feature, embedding_loss, perplexity

    def decoder(self, p, features):  # , trans, trans_loss):
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net)
        out = self.fc_out(net).squeeze(-1)
        # out = F.tanh(out)
        return out

    def forward(self, p, x):
        global_feature, vq_loss, perplexity = self.encoder(x)
        out = self.decoder(p, global_feature)
        return out, vq_loss, perplexity





class SPSDF_inter(nn.Module):
    def __init__(self, hidden_dim=64, res=128):
        super(SPSDF_inter, self).__init__()
        self.res = res
        self.SPUNet = MyRes16UNet14(in_channels=3, out_channels=hidden_dim, out_fpn=True, config=None).cuda()
        # self.SPUNet = MyRes16UNet18(in_channels=3, out_channels=hidden_dim, out_fpn=True, config=None).cuda()

        feature_size = hidden_dim+256 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.actvn = nn.ReLU()
        self.fc_sign = nn.Linear(hidden_dim, 1)

    def index_points(self, points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def encoder(self, coords, in_field, inverse_map):# [bs, 1024, 3]
        out, feat_map = self.SPUNet(in_field)
        out = self.SPUNet.final(out)
        ##
        point_feat = []
        for bs in range(coords.shape[0]):
            point_feat.append(out.decomposed_features[bs][inverse_map[bs]].unsqueeze(0))
        global_feat = []
        for bs in range(coords.shape[0]):
            global_feat.append(feat_map[0].decomposed_features[bs].mean(0, keepdim=True))
        return torch.cat(point_feat), coords, torch.cat(global_feat)

    def decoder(self, p, point_feat, point_xyz, global_feat):
        p_features = p ## query
        #
        dist, idx = knn(8, p, point_xyz)
        # dist, idx = three_nn(can_q, can_x)

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        query_feat = torch.sum(self.index_points(point_feat, idx.long()) * weight.unsqueeze(-1), dim=2)
        features = torch.cat((query_feat, p_features, global_feat.unsqueeze(1).repeat(1, p.shape[1], 1)), dim=-1)  # (B, N, C)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        sign = self.fc_sign(net).sigmoid()
        net = self.fc_out(net)
        out = net.squeeze(-1)
        return out, sign.squeeze(-1)

    def forward(self, p, in_field, coords, inverse_map):
        out, sign = self.decoder(p, *self.encoder(coords, in_field , inverse_map))
        return out, sign


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_in, bias=False)
        self.mlp = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, feature_set):
        #feature_set: ## [bs, N, 8, 128]
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=2)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=2)## [bs, N, 128]
        f_agg = self.mlp(f_agg)
        f_agg = self.bn(f_agg.transpose(1, 2)).transpose(1, 2)
        f_agg = F.relu(f_agg)
        return f_agg


class Att_pooling2(nn.Module):
    def __init__(self, d_in, d_out, d_f):
        super().__init__()
        self.fc = nn.Linear(d_in, 1, bias=False)
        self.mlp = nn.Linear(d_f, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, feature_set, patch_feats):
        #feature_set: ## [bs, N, 8, 128]
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=2)
        f_agg = torch.bmm(att_scores.squeeze(3), patch_feats)
        f_agg = self.mlp(f_agg)
        f_agg = self.bn(f_agg.transpose(1, 2)).transpose(1, 2)
        f_agg = F.relu(f_agg)
        return f_agg


class SPSDF_nninter(nn.Module):
    def __init__(self, hidden_dim=64, res=128):
        super(SPSDF_nninter, self).__init__()
        self.res = res
        self.SPUNet = MyRes16UNet14(in_channels=3, out_channels=hidden_dim, out_fpn=True, config=None).cuda()
        # self.SPUNet = MyRes16UNet18(in_channels=3, out_channels=hidden_dim, out_fpn=True, config=None).cuda()
        # self.query_point_encoding = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
        self.cls_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

        feature_size = hidden_dim*2+256 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim * 2)
        self.fc_1 = nn.Linear(hidden_dim *2, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc_out = nn.Linear(64, 1)
        self.actvn = nn.ReLU()
        self.fc_sign = nn.Linear(64, 1)

        self.mlp1 = nn.Linear(10, 32)
        self.att_pooling = Att_pooling(32+hidden_dim, 128)


    def index_points(self, points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def encoder(self, coords, in_field, inverse_map):# [bs, 1024, 3]
        out, feat_map = self.SPUNet(in_field)
        out = self.SPUNet.final(out)
        ##
        point_feat = []
        for bs in range(coords.shape[0]):
            point_feat.append(out.decomposed_features[bs][inverse_map[bs]].unsqueeze(0))
        global_feat = []
        for bs in range(coords.shape[0]):
            global_feat.append(feat_map[0].decomposed_features[bs].mean(0, keepdim=True))
        return torch.cat(point_feat), coords, torch.cat(global_feat)

    def decoder(self, p, point_feat, point_xyz, global_feat):
        p_features = p ## query
        #
        dist, idx = knn(8, p, point_xyz)

        neighor_feats = self.index_points(point_feat, idx.long()) ##[bs, N, 8, c]
        neighor_xyz = self.index_points(point_xyz, idx.long()) ##[bs, N, 8, 3]
        relative_xyz = neighor_xyz - p.unsqueeze(2)
        xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz, p.unsqueeze(2).repeat(1, 1, 8, 1), neighor_xyz], dim=-1)  # [bs, N, 8, 10]
        # f_xyz = self.mlp1(xyz_set.reshape(xyz_set.shape[0]*xyz_set.shape[1]*xyz_set.shape[2], -1)).reshape(xyz_set.shape[0], xyz_set.shape[1], xyz_set.shape[2], -1) ## [bs, N, 8, 64]
        f_xyz = self.mlp1(xyz_set) ## [bs, N, 8, 64]
        f_concat = torch.cat((neighor_feats, f_xyz), dim=-1)## [bs, N, 8, 128]
        # f_concat = torch.cat((f_concat, global_feat.unsqueeze(1).unsqueeze(1).repeat(1, p.shape[1], 8, 1)), dim=-1)## [bs, N, 8, 128]
        query_feat = self.att_pooling(f_concat)  # [bs, N, 128]

        features = torch.cat((query_feat, p_features, global_feat.unsqueeze(1).repeat(1, p.shape[1], 1)), dim=-1)  # (B, N, C)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        logits = self.fc_sign(net).sigmoid()
        net = self.fc_out(net)
        out = net.squeeze(-1)
        # return out, logits#self.cls_head(point_feat)
        return out, self.cls_head(point_feat)

    def forward(self, p, in_field, coords, inverse_map):
        out, cls = self.decoder(p, *self.encoder(coords, in_field, inverse_map))
        return out, cls



class SPSDF_glob(nn.Module):
    def __init__(self, hidden_dim=128, res=50):
        super(SPSDF_glob, self).__init__()
        self.res = res
        self.SPUNet = MyResNet14(in_channels=3, out_channels=256, out_fpn=False, config=None).cuda()
        self.mu = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.log_var = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256))

        feature_size = 256 + 3
        self.fc_0 = nn.Linear(feature_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.actvn = nn.ReLU()
        self.temp_feature = nn.Embedding(1, 256)

        self.decode_mlp = nn.Sequential(ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim), ResnetBlockConv1d(hidden_dim),
                                        ResnetBlockConv1d(hidden_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encoder(self, coords, in_field):# [bs, 1024, 3]
        batchsize = coords.shape[0]
        out = self.SPUNet(in_field)
        #
        global_feat = []
        for bs in range(coords.shape[0]):
            global_feat.append(out.decomposed_features[bs].mean(0, keepdim=True))
        global_feature = torch.cat(global_feat)
        mu, log_var = self.mu(global_feature), self.log_var(global_feature)
        # if self.training:
        #     z = self.reparameterize(mu, log_var)
        # else:
        #     z = mu
        z = mu
        # z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z, mu, log_var

    def decoder(self, p, features):
        p_features = p
        #
        features = torch.cat((features.unsqueeze(1).repeat(1, p.shape[1], 1), p_features), dim=-1)  # (B, N, C)
        net = self.actvn(self.fc_0(features))
        net = self.decode_mlp(net)
        out = self.fc_out(net).squeeze(-1)
        # out = F.tanh(out)
        return out

    def forward(self, p, in_field, coords, inverse_map):
        feature, mu, log_var = self.encoder(coords, in_field)
        out = self.decoder(p, feature)
        return out, mu, log_var


class SDF_simple_global(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SDF_simple_global, self).__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='zeros')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='zeros')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='zeros')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='zeros')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='zeros')  # out: 8

        feature_size = 128 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)
        # self.last_actvn = nn.Tanh()

    def encoder(self,x):# [bs, 256, 256]
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_6.mean(-1).mean(-1).mean(-1)

    def decoder(self, p, f_6):
        ## p: [bs, num_off_surface_points, 3], have to be [-1, 1], [zyx]
        p_features = p.transpose(1, -1)

        features = torch.cat((f_6.unsqueeze(-1).expand(-1, -1, p_features.shape[-1]), p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        # net = self.actvn(self.fc_out(net))
        net = (self.fc_out(net))
        out = net.squeeze(1)

        return out#, dense_feats.squeeze(0).t(), features.squeeze(0).t()
        # return out, feature_6.squeeze().t()#, features.squeeze(0).t()

    def forward(self, p, x):
        out = self.decoder(p, self.encoder(x))
        return out


# class ConvOcc(nn.Module):
#     def __init__(self):
#         super(ConvOcc, self).__init__()
#         unet3d_kwargs = {"num_levels": 3, "f_maps": 32, "in_channels": 32, "out_channels": 32}
#         self.encoder = LocalVoxelEncoder(dim=3, c_dim=32, unet3d=True, unet3d_kwargs=unet3d_kwargs, grid_resolution=32, plane_type='grid', padding=0.1)
#         self.decoder = LocalDecoder(dim=3, c_dim=32, hidden_size=32, padding=0.1)
#         self.last_actvn = nn.Tanh()
#
#     def forward(self, p, x):
#         tmp_p = p.clone()
#         p[:, :, 0], p[:, :, 2] = tmp_p[:, :, 2], tmp_p[:, :, 0]
#         out = self.decoder(p/2, self.encoder(x))
#         return self.last_actvn(out)


class EFEMSDF(nn.Module):
    def __init__(self):
        super(EFEMSDF, self).__init__()
        feature_size = 256#32+64+128+64+1
        self.encoder = VecDGCNN_att(c_dim=feature_size, num_layers=7, feat_dim=[32, 32, 64, 64, 128, 256, 512],#[16, 32, 32, 64, 64, 128],
                                    down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        # feature_size = 128
        # self.encoder = VecDGCNN_att(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256],#[16, 32, 32, 64, 64, 128],
        #                             down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        kwargs = {"legacy": False}
        self.decoder = DecoderCat(input_dim=feature_size*2+1, hidden_size=256, **kwargs)
        # self.decoder = DecoderCat(input_dim=feature_size*2+1, hidden_size=128, **kwargs)

    def encode(self, x, centroid=None):
        if centroid is None:
            centroid = x.mean(1, keepdim=True)
        x = x.transpose(1, -1)
        B, _, N = x.shape

        # encoding
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat = self.encoder(x)
        # centroid = center_pred.squeeze(1) + centroid
        centroid = center_pred + centroid.unsqueeze(1)

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()

        embedding = {"z_so3": pred_so3_feat, "z_inv": pred_inv_feat, "s": pred_scale, "t": centroid}#.unsqueeze(1)}
        return embedding, loss_scale, loss_center

    def decode(self, query, embedding):
        B, M, _ = query.shape

        z_so3, z_inv = embedding["z_so3"], embedding["z_inv"]
        scale, center = embedding["s"], embedding["t"]
        # z_so3 = torch.zeros_like(z_so3)

        q = (query - center) / scale[:, None, None]
        # q = (query) / scale[:, None, None]

        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

        input = torch.cat([inv_query, z_inv[:, None, :].expand(-1, M, -1)], -1)
        # input = torch.cat([q, z_inv.transpose(2, 1)], -1)
        out = self.decoder(input)
        return out

    def forward(self, query, x, centroid=None):
        embedding, loss_scale, loss_center = self.encode(x, centroid)
        out = self.decode(query, embedding)
        return out, loss_scale, loss_center




class EFEMSDF_VAE(nn.Module):
    def __init__(self):
        super(EFEMSDF_VAE, self).__init__()
        # feature_size = 256#32+64+128+64+1
        # self.encoder = VecDGCNN_att(c_dim=feature_size, num_layers=7, feat_dim=[32, 32, 64, 64, 128, 256, 512],#[16, 32, 32, 64, 64, 128],
        #                             down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        feature_size = 256*1
        self.encoder = VecDGCNN_att(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256*1],#[16, 32, 32, 64, 64, 128],
                                    down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True, z_so3_as_Omtx=True)#64000=40**3
        kwargs = {"legacy": False}
        # self.decoder = DecoderCatBN(input_dim=feature_size+3+1, hidden_size=128, **kwargs)
        self.decoder = DecoderCat(input_dim=feature_size+3+1, hidden_size=128, **kwargs)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.temp_feature = nn.Embedding(1, 512//2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x, centroid=None):
        if centroid is None:
            centroid = x.mean(1, keepdim=True)
        x = x.transpose(1, -1)
        B, _, N = x.shape

        # encoding
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat = self.encoder(x)
        # centroid = center_pred.squeeze(1) + centroid
        centroid = center_pred + centroid

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()

        mu, log_var = self.mu(pred_inv_feat), self.log_var(pred_inv_feat)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            # z = torch.clamp(z, min=-3, max=3)
            # z_length = z.norm(p=2, dim=-1, keepdim=True)+1e-6
            # z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            # z = (z/z_length)*torch.clamp(z_length, min=0, max=3)
        # z = mu
        z = z + self.temp_feature.weight.repeat(pred_inv_feat.shape[0], 1)

        embedding = {"z_so3": pred_so3_feat, "z_inv": z, "s": pred_scale, "t": centroid}#.unsqueeze(1)}
        return embedding, loss_scale, loss_center, mu, log_var

    def decode(self, query, embedding):#, loss_scale, loss_center, mu, log_var):
    # def decode(self, query, embedding, loss_scale, loss_center, mu, log_var):
        B, M, _ = query.shape

        z_so3, z_inv = embedding["z_so3"], embedding["z_inv"]
        scale, center = embedding["s"], embedding["t"]

        q = (query - center) / scale[:, None, None]

        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

        input = torch.cat([inv_query, z_inv[:, None, :].expand(-1, M, -1)], -1)
        # input = input.transpose(2, 1)
        out = self.decoder(input)
        return out
        # return out, loss_scale, loss_center, mu, log_var

    def decode_list(self, query, embedding, require_grad=False):#, query is a list [[1, N1, 3], [1, N2, 3]]
        nmd = time.time()
        input_list, query_list = [], []

        z_so3_list, z_inv_list, scale_list, center_list = [], [], [], []
        for bs in range(len(query)):
            z_so3, z_inv = embedding["z_so3"][bs].unsqueeze(0).repeat(query[bs].shape[1], 1, 1), embedding["z_inv"][bs].unsqueeze(0).repeat(query[bs].shape[1], 1)
            scale, center = embedding["s"][bs].unsqueeze(0).repeat(query[bs].shape[1]), embedding["t"][bs].unsqueeze(0).repeat(query[bs].shape[1], 1, 1)
            z_so3_list.append(z_so3), z_inv_list.append(z_inv), scale_list.append(scale), center_list.append(center)
        z_so3, z_inv, scale, center = torch.cat(z_so3_list), torch.cat(z_inv_list), torch.cat(scale_list), torch.cat(center_list)

        query = torch.cat(query, dim=1).transpose(0, 1).cuda() ## [N*bs, 1, 3]
        M, bs, _ = query.shape

        if require_grad:
            query.requires_grad = True

        q = (query - center) / scale[:, None, None]
        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

        input = torch.cat([inv_query, z_inv[:, None, :]], -1) ## [bs*N, 1, C]
        input = input.transpose(0, 1)#.transpose(2, 1)
        out = self.decoder(input)
        return out.squeeze(0), query#, loss_scale, loss_center, mu, log_var

    def forward(self, query, x, centroid=None):
        out, loss_scale, loss_center, mu, log_var = self.decode(query, *self.encode(x, centroid))
        return out, loss_scale, loss_center, mu, log_var
    # def forward(self, query, x, centroid=None):
    #     return self.decode(query, self.encode(x, centroid)[0])


class EFEMSDF_VAEpatch(nn.Module):
    def __init__(self):
        super(EFEMSDF_VAEpatch, self).__init__()
        # feature_size = 256#32+64+128+64+1
        # self.encoder = VecDGCNN_att(c_dim=feature_size, num_layers=7, feat_dim=[32, 32, 64, 64, 128, 256, 512],#[16, 32, 32, 64, 64, 128],
        #                             down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        feature_size = 256*1
        self.encoder = VecDGCNN_att_patch(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256*1],#[16, 32, 32, 64, 64, 128],
                                    down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True, z_so3_as_Omtx=True)#64000=40**3
        kwargs = {"legacy": False}
        # self.decoder = DecoderCatBN(input_dim=128+3+1, hidden_size=128, **kwargs)
        self.decoder = DecoderCat(input_dim=256+3+1, hidden_size=128, **kwargs)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        # self.temp_feature = nn.Embedding(1, 512//4)

        self.mlp1 = nn.Linear(4, 64)
        self.att_pooling = Att_pooling2(64+512//2, 512//2, d_f=512//2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x, centroid=None):
        if centroid is None:
            centroid = x.mean(1, keepdim=True)
        x = x.transpose(1, -1)
        B, _, N = x.shape

        # encoding
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat, patch_coord = self.encoder(x)
        # centroid = center_pred.squeeze(1) + centroid
        centroid = center_pred + centroid

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()

        pred_inv_feat = pred_inv_feat.transpose(2, 1) ## [bs, K, C]
        # pred_inv_feat = torch.cat((pred_inv_feat, pred_inv_feat.mean(1, keepdim=True).repeat(1, pred_inv_feat.shape[1], 1)), dim=-1)  ## [bs, 128, 128+512]
        mu, log_var = self.mu(pred_inv_feat), self.log_var(pred_inv_feat) ##[bs, 16, 256]
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
            # z_length = torch.clamp((z.norm(p=2, dim=-1, keepdim=True)+1e-6), min=0, max=3)
            # z = z_length*z/(z.norm(p=2, dim=-1, keepdim=True)+1e-6)
        # z = mu

        embedding = {"z_so3": pred_so3_feat, "z_inv": z, "s": pred_scale, "t": centroid, "sub_center": patch_coord}#.unsqueeze(1)}
        return embedding, loss_scale, loss_center, mu, log_var

    def decode(self, query, embedding):#, loss_scale, loss_center, mu, log_var):
    # def decode(self, query, embedding, loss_scale, loss_center, mu, log_var):
        B, M, _ = query.shape

        z_so3, z_inv = embedding["z_so3"], embedding["z_inv"]
        scale, center = embedding["s"], embedding["t"]
        patch_coord = embedding["sub_center"]

        ####
        neighor_feats = z_inv.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        neighor_xyz = patch_coord.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        relative_xyz = neighor_xyz - query.unsqueeze(2)  ## [bs, 5k, 16, 3]
        dist = relative_xyz.norm(p=2, dim=-1)
        xyz_set = torch.cat([dist.unsqueeze(-1), relative_xyz], dim=-1)  # [bs, 5k, 16, 10]
        f_xyz = self.mlp1(xyz_set)  ## [bs, 5k, 16, 32]
        f_concat = torch.cat((neighor_feats, f_xyz), dim=-1)## [bs, 5000, 16, 256+64]
        query_feat = self.att_pooling(f_concat, z_inv)  # [bs, 5000, 256]
        z_inv = query_feat
        ####

        q = (query - center) / scale[:, None, None]

        inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
        length = q.norm(dim=-1).unsqueeze(1)
        inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

        input = torch.cat([inv_query, z_inv], -1) ## [bs, N, C]
        # input = input.transpose(2, 1)
        out = self.decoder(input)
        return out
        # return out, loss_scale, loss_center, mu, log_var

    def forward(self, query, x, centroid=None):
        # embedding, loss_scale, loss_center, mu, log_var, patch_coord = self.encode(x, centroid)
        out, loss_scale, loss_center, mu, log_var = self.decode(query, *self.encode(x, centroid))
        return out, loss_scale, loss_center, mu, log_var



class EFEMSDF_UNet(nn.Module):
    def __init__(self):
        super(EFEMSDF_UNet, self).__init__()
        feature_size = 128#32+64+128+64+1
        # self.encoder = VecDGCNN_att_unet(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256],#[16, 32, 32, 64, 64, 128],
        #                             down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        self.encoder = VecDGCNN_att_unet(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256],#[16, 32, 32, 64, 64, 128],
                                    down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        kwargs = {"legacy": False}
        self.decoder = DecoderCat(input_dim=64+1, hidden_size=64, **kwargs)
        # self.decoder = DecoderCat(input_dim=16*3+1, hidden_size=64, **kwargs)

    def get_grid_feats(self, xyz, feat, res=32):# xyz: [B, N, c], feat:[B, C, N]
        xyz = self.normalize_3d_coordinate(xyz)## [-0.5, 0.5] --> [0, 1]
        xyz = (xyz * res).long()
        index = xyz[:, :, 0] + res * (xyz[:, :, 1] + res * xyz[:, :, 2])
        index = index[:, None, :]
        # scatter grid features from points
        fea_grid = feat.new_zeros(xyz.size(0), feat.size(1), res**3)
        # feat = feat.reshape(feat.size(0), feat.size(1)*feat.size(2), feat.size(3))
        fea_grid = scatter_mean(feat, index, out=fea_grid)
        fea_grid = fea_grid.reshape(xyz.size(0), feat.size(1), res, res, res)
        return fea_grid #[B, C, res*3]

    def sample_grid_feature(self, query_point, grid_feats):
        p_nor = self.normalize_3d_coordinate(query_point.clone()) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        # p_nor = p_nor.unsqueeze(1).unsqueeze(1).float()#[32, N, 1, 1, 3]
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        query_feats = F.grid_sample(grid_feats, vgrid, padding_mode='border', align_corners=True).squeeze(-1).squeeze(-1)
        return query_feats.transpose(1, 2)

    def normalize_3d_coordinate(self, p):
        p_nor = p
        p_nor = p_nor + 1#0.5  # range (0, 2)
        # f there are outliers out of the range
        # if p_nor.max() >= 1:
        #     p_nor[p_nor >= 1] = 1 - 1e-4
        if p_nor.max() >= 2:
            p_nor[p_nor >= 2] = 2 - 1e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor/2

    def encode(self, x, centroid=None):
        if centroid is None:
            centroid = x.mean(1, keepdim=True)
        x = x.transpose(1, -1)
        B, _, N = x.shape

        # encoding
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat = self.encoder(x)
        # centroid = center_pred.squeeze(1) + centroid
        centroid = center_pred + centroid

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()

        can_x = (x.transpose(1, -1) - centroid) / pred_scale[:, None, None] ###[-0.5, 0.5]
        fea_grid = self.get_grid_feats(can_x, pred_inv_feat.transpose(1, -1))

        # res = 64
        # grid_index = torch.arange(res * res * res)
        # zz = grid_index // (res * res)
        # yy = (grid_index - zz * res * res) // res
        # xx = grid_index - zz * res * res - yy * res
        # xyz_grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1)
        #
        # pca = PCA(n_components=3)
        # rgb = pca.fit_transform(fea_grid[0].reshape(64, -1).detach().t().cpu().numpy())
        # rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
        # write_ply('grid_feat.ply', [xyz_grid.cpu().float().numpy(), rgb.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # write_ply('raw_xyz.ply', [x[0].t().cpu().float().numpy()], ['x', 'y', 'z'])

        embedding = {"z_so3": pred_so3_feat, "z_inv": fea_grid, "s": pred_scale, "t": centroid}
        return embedding, loss_scale, loss_center

    def decode(self, query, embedding):
        B, M, _ = query.shape

        z_so3, fea_grid = embedding["z_so3"], embedding["z_inv"]
        scale, center = embedding["s"], embedding["t"]

        can_q = (query - center) / scale[:, None, None]
        query_fea = self.sample_grid_feature(can_q, fea_grid)
        length = can_q.norm(dim=-1).unsqueeze(-1)
        input = torch.cat([query_fea, length], -1)

        out = self.decoder(input)
        return out

    def forward(self, query, x, centroid=None):
        embedding, loss_scale, loss_center = self.encode(x, centroid)
        out = self.decode(query, embedding)
        return out, loss_scale, loss_center



class EFEMSDF_UNetinter(nn.Module):
    def __init__(self):
        super(EFEMSDF_UNetinter, self).__init__()
        feature_size = 128#32+64+128+64+1
        self.encoder = VecDGCNN_att_unet(c_dim=feature_size, num_layers=7, feat_dim=[16, 16, 32, 32, 64, 128, 256],#[16, 32, 32, 64, 64, 128],
                                    down_sample_layers=[2, 4, 5], scale_factor=64000.0, center_pred=True)#64000=40**3
        kwargs = {"legacy": False}
        self.decoder = DecoderCat(input_dim=64+1, hidden_size=64, **kwargs)
        # self.decoder = DecoderCat(input_dim=16*3+1, hidden_size=64, **kwargs)


    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def index_points(self, points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def encode(self, x, centroid=None):
        if centroid is None:
            centroid = x.mean(1, keepdim=True)
        x = x.transpose(1, -1)
        B, _, N = x.shape

        # encoding
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat = self.encoder(x)
        # centroid = center_pred.squeeze(1) + centroid
        centroid = center_pred + centroid

        loss_scale = abs(pred_scale - 1.0).mean()
        loss_center = centroid.norm(1, dim=-1).mean()

        can_x = (x.transpose(1, -1) - centroid) / pred_scale[:, None, None] ###[-0.5, 0.5]

        embedding = {"z_so3": pred_so3_feat, "z_inv": pred_inv_feat, "s": pred_scale, "t": centroid, "can_x": can_x}
        return embedding, loss_scale, loss_center

    def decode(self, query, embedding):
        B, M, _ = query.shape

        z_so3, pred_inv_feat = embedding["z_so3"], embedding["z_inv"]
        scale, center = embedding["s"], embedding["t"]
        can_x = embedding["can_x"] ### [bs, N, 3]

        can_q = (query - center) / scale[:, None, None]

        # dist, idx = knn(8, can_q, can_x)
        dist, idx = three_nn(can_q, can_x)
        # dist = self.square_distance(can_q, can_x)
        # dist, idx = dist.sort(dim=-1)
        # dist, idx = dist[:, :, :8], idx[:, :, :8]  # [B, N, 3]

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        # query_fea = three_interpolate(pred_inv_feat.permute(0, 2, 1).contiguous(), idx.contiguous(), weight.contiguous()).permute(0, 2, 1)
        query_fea = torch.sum(self.index_points(pred_inv_feat, idx.long()) * weight.unsqueeze(-1), dim=2)
        # query_fea = self.index_points(pred_inv_feat, idx.long()).squeeze(2)

        length = can_q.norm(dim=-1).unsqueeze(-1)
        input = torch.cat([query_fea, length], -1)

        out = self.decoder(input)
        return out

    def forward(self, query, x, centroid=None):
        embedding, loss_scale, loss_center = self.encode(x, centroid)
        out = self.decode(query, embedding)
        return out, loss_scale, loss_center
