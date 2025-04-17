# From VN DGCNN add scale equiv, no se(3) is considered in plain dgcnn

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vec_layers import *
from vec_layers import VecLinearNormalizeActivate as VecLNA
from layers_equi import *


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecDGCNN(nn.Module):
    # ! use mean pooling
    def __init__(
        self,
        hidden_dim=128,
        c_dim=128,
        first_layer_knn=16,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=False,
        z_so3_as_Omtx=True,
        center_pred=True,
        center_pred_scale=True):
        super().__init__()

        self.use_dg = use_dg
        if self.use_dg:
            logging.info("DGCNN use Dynamic Graph (different from the input topology)")
        self.scale_factor = scale_factor

        # * prepare layers
        self.h_dim = hidden_dim
        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False)

        self.c_dim = c_dim
        self.k = first_layer_knn

        # self.conv1 = VecLNA(2, hidden_dim, mode="so3", act_func=act_func)
        # self.conv2 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
        # self.conv3 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
        # self.conv4 = VecLNA(hidden_dim * 2, hidden_dim, mode="so3", act_func=act_func)
        self.conv1 = VecLNA(2, 32, mode="so3", act_func=act_func)
        self.conv2 = VecLNA(64, 64, mode="so3", act_func=act_func)
        self.conv3 = VecLNA(128, 128, mode="so3", act_func=act_func)
        self.conv4 = VecLNA(128, hidden_dim, mode="so3", act_func=act_func)


        self.pool1 = meanpool
        self.pool2 = meanpool
        self.pool3 = meanpool
        self.pool4 = meanpool

        self.conv_c = VecLNA(hidden_dim*2+32+64, c_dim, mode="so3", act_func=act_func, shared_nonlinearity=True)

        self.fc_inv = VecLinear(c_dim, c_dim, mode="so3")
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

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None, label=None):
        # x: B,C,3,N return B,C*2,3,N,K

        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            _, knn_idx, neighbors = knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x = x[..., None].expand_as(neighbors)
        y = torch.cat([neighbors - x, x], 1)
        label_map = torch.gather(label.unsqueeze(-1).expand(-1, -1, N), dim=1, index=knn_idx)#[bs, 1024, 16]
        return y, knn_idx, label_map  # B,C*2,3,N,K

    def forward(self, x, label):
        if label is None:
            label= torch.ones_like(x)[:, 0, :]
        B, _, N = x.shape
        x = x.unsqueeze(1)
        x, knn_idx, label_map = self.get_graph_feature(x, k=self.k, knn_idx=None, label=label)
        x = self.conv1(x)
        x1 = self.pool1(x)
        # x1 = (x * label_map[:, None, None, :, :]).sum(dim=-1, keepdim=False) / (label_map[:, None, None, :, :].sum(dim=-1, keepdim=False) + 1e-12)
        if self.use_dg:
            knn_idx = None

        x, _, label_map = self.get_graph_feature(x1, k=self.k, knn_idx=knn_idx, label=label)
        x = self.conv2(x)
        x2 = self.pool2(x)
        # x2 = (x * label_map[:, None, None, :, :]).sum(dim=-1, keepdim=False) / (label_map[:, None, None, :, :].sum(dim=-1, keepdim=False) + 1e-12)

        x, _, label_map = self.get_graph_feature(x2, k=self.k, knn_idx=knn_idx, label=label)
        x = self.conv3(x)
        x3 = self.pool3(x)
        # x3 = (x * label_map[:, None, None, :, :]).sum(dim=-1, keepdim=False) / (label_map[:, None, None, :, :].sum(dim=-1, keepdim=False) + 1e-12)

        x, _, label_map = self.get_graph_feature(x3, k=self.k, knn_idx=knn_idx, label=label)
        x = self.conv4(x)
        x4 = self.pool4(x)
        # x4 = (x * label_map[:, None, None, :, :]).sum(dim=-1, keepdim=False) / (label_map[:, None, None, :, :].sum(dim=-1, keepdim=False) + 1e-12)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_c(x)
        # y = (x * label[:, None, None, :]).sum(dim=-1, keepdim=False) / (label[:, None, None, :].sum(dim=-1, keepdim=False)+1e-12)
        so3 = channel_equi_vec_normalize(x)
        ###############
        if self.z_so3_as_Omtx_flag:
            z_so3 = self.fc_O(so3)  # B,Basis,3
            R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
            _U, _, _Vh = torch.linalg.svd(R_pred.double())
            so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,num_basis,3
        ##################
        scale = (y.norm(dim=-1)+1e-12).mean(1) * self.scale_factor
        center = self.fc_center(y[..., None]).squeeze(-1)
        if self.center_pred_scale:
            center = center * self.scale_factor

        # x = channel_equi_vec_normalize(x)
        y = channel_equi_vec_normalize(y)
        mean = self.mean(y)#.mean(dim=-1, keepdim=False)
        logvar = self.logvar(y)#.mean(dim=-1, keepdim=False)

        so3_mean = self.fc_inv_mean(mean[..., None]).squeeze(-1)#[bs, C, 3]
        inv_mean = (mean * so3_mean).sum(-1)

        so3_logvar = self.fc_inv_logvar(logvar[..., None]).squeeze(-1)#[bs, C, 3]
        inv_logvar = (logvar * so3_logvar).sum(-1)
        return inv_mean, inv_logvar, so3, scale, center



class VNN_ResnetPointnet(nn.Module):
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
        # mean = get_graph_mean(p, k=self.k)
        # mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        # pooled = (net * label[:, None, None, :]).sum(dim=-1, keepdim=True) / (label[:, None, None, :].sum(dim=-1, keepdim=True)+1e-12).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)  ## [bs, N, 3, C]?
        x = self.fc_c(self.actvn_c(net))#[bs, C, 3, N]

        # y = (x * label[:, None, None, :]).sum(dim=-1, keepdim=False) / (label[:, None, None, :].sum(dim=-1, keepdim=False)+1e-12)
        y = x.mean(dim=-1, keepdim=False)
        so3 = channel_equi_vec_normalize(y)
        ###############
        if self.z_so3_as_Omtx_flag:
            z_so3 = self.fc_O(so3)  # B,Basis,3
            R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
            _U, _, _Vh = torch.linalg.svd(R_pred.double())
            so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,num_basis,3
        # if self.z_so3_as_Omtx_flag:
        #     z_so3 = self.fc_O(so3)  # B,Basis,3
        #     R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
        #     _U, _, _Vh = torch.linalg.svd(R_pred.double())
        #     so3 = (_U @ _Vh).transpose(2, 1)#.type(R_pred.dtype)  # B,num_basis,3
        #
        #     det = torch.linalg.det(so3)
        #     # Correct reflection matrix to rotation matrix
        #     diag = torch.ones_like(so3[..., 0], requires_grad=False)
        #     diag[:, 2] = det
        #     so3 = _Vh.bmm(torch.diag_embed(diag).bmm(_U.transpose(1, 2))).type(R_pred.dtype)
        ##################
        scale = y.norm(dim=-1).mean(1) * self.scale_factor
        center = self.fc_center(y[..., None]).squeeze(-1)
        if self.center_pred_scale:
            center = center * self.scale_factor

        # x = channel_equi_vec_normalize(x)
        y = channel_equi_vec_normalize(y)
        mean = self.mean(y)#.mean(dim=-1, keepdim=False)
        logvar = self.logvar(y)#.mean(dim=-1, keepdim=False)

        so3_mean = self.fc_inv_mean(mean[..., None]).squeeze(-1)#[bs, C, 3]
        inv_mean = (mean * so3_mean).sum(-1)
        #
        so3_logvar = self.fc_inv_logvar(logvar[..., None]).squeeze(-1)#[bs, C, 3]
        inv_logvar = (logvar * so3_logvar).sum(-1)
        # inv_mean, inv_logvar = mean.reshape(mean.shape[0], -1), logvar.reshape(mean.shape[0], -1)
        # tmp = channel_equi_vec_normalize(x)
        # tmp_mean = self.mean(tmp)
        return inv_mean, inv_logvar, so3, scale, center#, (tmp_mean*self.fc_inv_mean(tmp_mean)).sum(-2)

