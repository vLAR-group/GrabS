# From VN DGCNN add scale equiv, no se(3) is considered in plain dgcnn

import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vec_layers import *
from vec_layers import VecLinearNormalizeActivate as VecLNA
from layers_equi import *
from torch_scatter import scatter_mean


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNN_ResnetPointnet_local(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, hidden_dim=128, k=16, scale_factor=640.0, use_dg=False, z_so3_as_Omtx=False, center_pred=True,
        center_pred_scale=True):
        super().__init__()
        self.c_dim = c_dim
        self.k = k

        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False) # ! this is critical, use inplace=False!
        # self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        # self.fc_pos = VNLinear(128, 2 * hidden_dim)
        # self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.fc_c = VNLinear(hidden_dim, c_dim)
        #
        # self.block_3_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_3_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        # self.block_3_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.conv_pos = VNLinearLeakyReLU(3, 32, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(32, 2 * 32)
        self.block_0 = VNResnetBlockFC(2 * 32, 64)
        self.block_1 = VNResnetBlockFC(2 * 64, 64)
        self.block_2 = VNResnetBlockFC(2 * 64, 64)
        self.block_3 = VNResnetBlockFC(2 * 64, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, 2*hidden_dim)
        self.fc_c = VNLinear(2*hidden_dim, c_dim)

        self.block_3_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)

        self.actvn_c = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        # self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        self.z_so3_as_Omtx_flag = z_so3_as_Omtx
        if z_so3_as_Omtx:
            self.fc_O = VecLinear(c_dim, 3, mode="so3")
        self.mean = nn.Sequential(VecLNA(c_dim, c_dim, mode="so3", act_func=act_func),#, shared_nonlinearity=True),
                                    VecLNA(c_dim, c_dim, mode="so3", act_func=act_func))#, shared_nonlinearity=True))

        self.logvar = nn.Sequential(VecLNA(c_dim, c_dim, mode="so3", act_func=act_func),#, shared_nonlinearity=True),
                                    VecLNA(c_dim, c_dim, mode="so3", act_func=act_func))#, shared_nonlinearity=True))

        self.fc_inv_mean = VecLinear(c_dim, c_dim, mode="so3")
        self.fc_inv_logvar = VecLinear(c_dim, c_dim, mode="so3")

        self.fc_scale_mean = VecLinear(c_dim, c_dim, mode="so3")
        self.fc_scale_logvar = VecLinear(c_dim, c_dim, mode="so3")

        self.z_so3_as_Omtx_flag = z_so3_as_Omtx
        if z_so3_as_Omtx:
            self.fc_O = VecLinear(c_dim, 3, mode="so3")

        self.center_pred = center_pred
        self.center_pred_scale = center_pred_scale
        if self.center_pred:
            self.fc_center = VecResBlock(c_dim, 1, c_dim // 2, act_func=act_func, mode="so3")
        self.scale_factor = scale_factor

    def forward(self, p, label=None):
        if label is None:
            label= torch.ones_like(p)[:, 0, :]
        batch_size = p.size(0)
        p = p.unsqueeze(1)#.transpose(2, 3)
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        # net = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())

        net = self.fc_pos(net)

        net = self.block_0(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_1(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_2(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_3(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_3_1(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_3_2(net)
        # pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, net], dim=1)

        net = self.block_4(net)  ## [bs, N, 3, C]?
        x = self.fc_c(self.actvn_c(net))#[bs, C, 3, N]

        # y = (x * label[:, None, None, :]).sum(dim=-1, keepdim=False) / (label[:, None, None, :].sum(dim=-1, keepdim=False)+1e-12)
        y = x.mean(dim=-1)
        so3 = channel_equi_vec_normalize(y)
        ###############
        if self.z_so3_as_Omtx_flag:
            z_so3 = self.fc_O(so3)  # B,Basis,3
            R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
            _U, _, _Vh = torch.linalg.svd(R_pred.double())
            so3 = (_U @ _Vh).transpose(2, 1)#.type(R_pred.dtype)  # B,num_basis,3

            det = torch.det(so3)
            # Correct reflection matrix to rotation matrix
            diag = torch.ones_like(so3[..., 0], requires_grad=False)
            diag[:, 2] = det
            so3 = _Vh.bmm(torch.diag_embed(diag).bmm(_U.transpose(1, 2))).type(R_pred.dtype)
        ##################
        scale = (y.norm(dim=-1)+1e-12).mean(1) * self.scale_factor
        center = self.fc_center(y[..., None]).squeeze(-1)
        if self.center_pred_scale:
            center = center * self.scale_factor

        x = channel_equi_vec_normalize(x)
        # y = channel_equi_vec_normalize(y)
        mean = self.mean(x)#.mean(dim=-1, keepdim=False)
        logvar = self.logvar(x)#.mean(dim=-1, keepdim=False)

        so3_mean = self.fc_inv_mean(mean)#[bs, C, 3]
        inv_mean = (mean * so3_mean).sum(2)#[bs, C, N]

        so3_logvar = self.fc_inv_logvar(logvar)#[bs, C, 3]
        inv_logvar = (logvar * so3_logvar).sum(2)

        return inv_mean, inv_logvar, so3, scale, center


    def get_grid_feats(self, xyz, feat, res=64, label=None):# xyz: [B, N, c], feat:[B, C, N]
        xyz = self.normalize_3d_coordinate(xyz)## [-0.5, 0.5] --> [0, 1]
        xyz = (xyz * res).long()
        index = xyz[:, 0, :] + res * (xyz[:, 1, :] + res * xyz[:, 2, :])
        index = index[:, None, :]
        # scatter grid features from points
        fea_grid = feat.new_zeros(xyz.size(0), feat.size(1), res**3)
        # feat = feat.reshape(feat.size(0), feat.size(1)*feat.size(2), feat.size(3))
        # feat = feat*label[:, None, :]
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

    def normalize_3d_coordinate(self, p, padding=0.1):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            p (tensor): point
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        '''

        # p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5), no need if already normalized
        p_nor = p
        p_nor = p_nor + 0.5  # range (0, 1)
        # f there are outliers out of the range
        if p_nor.max() >= 1:
            p_nor[p_nor >= 1] = 1 - 10e-4
        if p_nor.min() < 0:
            p_nor[p_nor < 0] = 0.0
        return p_nor

