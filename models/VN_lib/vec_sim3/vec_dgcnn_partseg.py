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


class VecDGCNN_seg(nn.Module):
    # ! use mean pooling
    def __init__(
        self,
        hidden_dim=128,
        c_dim=128,
        first_layer_knn=16,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=False):
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
        self.conv1 = VecLNA(2, 16, mode="so3", act_func=act_func)
        self.conv2 = VecLNA(32, 32, mode="so3", act_func=act_func)
        self.conv3 = VecLNA(64, 64, mode="so3", act_func=act_func)
        self.conv4 = VecLNA(128, hidden_dim, mode="so3", act_func=act_func)


        self.pool1 = meanpool
        self.pool2 = meanpool
        self.pool3 = meanpool
        self.pool4 = meanpool

        self.conv_c = VecLNA(hidden_dim+16+32+64, c_dim, mode="so3", act_func=act_func, shared_nonlinearity=True)

        self.fc_inv = nn.Sequential(VecLNA(c_dim*2, c_dim, mode="so3", act_func=act_func),#, shared_nonlinearity=True),
                                    VecLNA(c_dim, c_dim*2, mode="so3", act_func=act_func))#, shared_nonlinearity=True))


    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None):
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
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):
        B, _, N = x.shape
        x = x.unsqueeze(1)
        x, knn_idx = self.get_graph_feature(x, k=self.k, knn_idx=None)
        x = self.conv1(x)
        x1 = self.pool1(x)
        if self.use_dg:
            knn_idx = None

        x, _ = self.get_graph_feature(x1, k=self.k, knn_idx=knn_idx)
        x = self.conv2(x)
        x2 = self.pool2(x)

        x, _ = self.get_graph_feature(x2, k=self.k, knn_idx=knn_idx)
        x = self.conv3(x)
        x3 = self.pool3(x)

        x, _ = self.get_graph_feature(x3, k=self.k, knn_idx=knn_idx)
        x = self.conv4(x)
        x4 = self.pool4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_c(x)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)

        x0 = self.fc_inv(x)

        inv = (x * x0).sum(2)
        return inv

