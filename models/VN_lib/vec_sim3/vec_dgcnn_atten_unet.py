import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_farthest_points
from torch_scatter import scatter_mean, scatter_max

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vec_layers import *
from vec_layers import VecLinearNormalizeActivate as VecLNA
from pointnet2.pointnet2 import three_nn, three_interpolate


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecDGCNN_att_unet(nn.Module):
    def __init__(
        self,
        c_dim=256,
        num_layers=8,
        feat_dim=[32, 32, 64, 64, 128, 256, 512, 512],
        down_sample_layers=[2, 4, 6],
        down_sample_factor=[4, 4, 4],
        atten_start_layer=2,
        atten_multi_head_c=16,  # each N channels will compute one atten weight
        use_res_global_conv=True,
        res_global_start_layer=2,
        num_knn=16,
        # for first several layers, can have different K
        num_knn_early=-1,
        knn_early_layers=-1,
        scale_factor=640.0,
        leak_neg_slope=0.2,
        use_dg=True,  # if true, every layer use a new knn topology
        center_pred=False,
        center_pred_scale=False,
        # for z_so3 to be a rotation matrix in O(3)
        z_so3_as_Omtx=False,
        res = 256
    ):
        super().__init__()

        self.use_dg = use_dg
        if self.use_dg:
            logging.info("DGCNN use Dynamic Graph (different from the input topology)")
        self.scale_factor = scale_factor

        self.use_res_global_conv = use_res_global_conv  # use a global point net as well
        self.res_global_start_layer = res_global_start_layer

        self.num_layers = num_layers
        self.down_sample_layers = down_sample_layers
        self.down_sample_factor = down_sample_factor
        assert len(down_sample_factor) == len(down_sample_layers)
        self.atten_start_layer = atten_start_layer
        assert self.atten_start_layer >= 1, "first layers should use naive DGCNN"
        self.feat_dim = feat_dim
        assert len(self.feat_dim) == num_layers
        self.atten_multi_head_c = atten_multi_head_c
        self.res = res

        act_func = nn.LeakyReLU(negative_slope=leak_neg_slope, inplace=False) # ! this is critical, use inplace=False!

        self.c_dim = c_dim
        self.k = num_knn
        if num_knn_early < 0:
            num_knn_early = num_knn
        self.k_early = num_knn_early
        self.k_early_layers = knn_early_layers
        self.pool = meanpool

        self.global_conv_list, self.V_list = nn.ModuleList(), nn.ModuleList()
        self.Q_list, self.K_list = nn.ModuleList(), nn.ModuleList()

        for i in range(self.num_layers):
            # * Make Regular DGCNN or V layers
            if i == 0:
                self.V_list.append(VecLNA(3, feat_dim[0], mode="so3", act_func=act_func))
            else:
                self.V_list.append(
                    VecLNA(feat_dim[i - 1] * 2, feat_dim[i], mode="so3", act_func=act_func)
                )
            # * Make Global Pooling Layers
            if use_res_global_conv and i >= self.res_global_start_layer:
                self.global_conv_list.append(
                    VecLNA(feat_dim[i] * 2, feat_dim[i], mode="so3", act_func=act_func)
                )

            # * Make Atten QK
            if i >= self.atten_start_layer:
                assert feat_dim[i] % self.atten_multi_head_c == 0, f"hidden % {atten_multi_head_c}"
                self.Q_list.append(
                    VecLNA(feat_dim[i - 1], feat_dim[i], mode="so3", act_func=act_func)
                )
                self.K_list.append(
                    VecLNA(feat_dim[i - 1] * 2, feat_dim[i], mode="so3", act_func=act_func)
                )
            else:
                self.Q_list.append(None), self.K_list.append(None)

        self.conv_c = VecLNA(
            feat_dim[-1],
            c_dim,
            mode="so3",
            act_func=act_func,
            shared_nonlinearity=True,
        )

        self.fc_inv = VecLinear(c_dim, c_dim, mode="so3")

        self.z_so3_as_Omtx_flag = z_so3_as_Omtx
        if z_so3_as_Omtx:
            self.fc_O = VecLinear(c_dim, 3, mode="so3")

        self.center_pred = center_pred
        self.center_pred_scale = center_pred_scale
        if self.center_pred:
            self.fc_center = VecResBlock(c_dim, 1, c_dim // 2, act_func=act_func, mode="so3")

        self.up_layers = nn.ModuleList()
        for layer_id in range(len(self.down_sample_layers)):
            if layer_id< len(self.down_sample_layers)-1:
                curr_layer, next_layer = self.down_sample_layers[layer_id], self.down_sample_layers[layer_id+1]
                self.up_layers.append(VecLNA(feat_dim[next_layer-1] + feat_dim[curr_layer-1], feat_dim[curr_layer-1], mode="so3", act_func=act_func, shared_nonlinearity=True))
            else:
                curr_layer= self.down_sample_layers[layer_id]
                self.up_layers.append(VecLNA(feat_dim[-1] + feat_dim[curr_layer-1], feat_dim[curr_layer-1], mode="so3", act_func=act_func, shared_nonlinearity=True))
        self.up_layers.append(VecLNA(feat_dim[-1], feat_dim[-1], mode="so3", act_func=act_func, shared_nonlinearity=True))
        self.perpoint_feature = VecLinear(feat_dim[down_sample_layers[0]-1], feat_dim[down_sample_layers[0]-1], mode="so3")
        self.fuse_MLP_fc1 = nn.Linear(in_features=c_dim*feat_dim[down_sample_layers[0]-1], out_features=c_dim)
        self.fuse_MLP_fc2 = nn.Linear(in_features=c_dim, out_features=64)
        self.fuse_MLP_bn = nn.BatchNorm1d(c_dim)

    def get_graph_feature(
        self,
        src_f: torch.Tensor,
        dst_f: torch.Tensor,
        k: int,
        src_xyz: torch.Tensor,
        dst_xyz: torch.Tensor,
        cross=False,
    ):
        # * dst is the query points
        # x: B,C,3,N return B,C*2,3,N,K
        B, C, _, N_src = src_f.shape
        N_dst = dst_f.shape[-1]
        if self.use_dg:
            _dst, _src = dst_f.reshape(B, -1, N_dst), src_f.reshape(B, -1, N_src)
            _, knn_idx, dst_nn_in_src = knn_points(
                _dst.transpose(2, 1), _src.transpose(2, 1), K=k, return_nn=True
            )  # B,N_dst,K; B,N_dst,K; B,N_dst,K,D
            dst_nn_in_src = dst_nn_in_src.reshape(B, N_dst, k, C, 3).permute(0, -2, -1, 1, 2)
        else:
            _dst, _src = dst_xyz.reshape(B, -1, N_dst), src_xyz.reshape(B, -1, N_src)
            _, knn_idx, _ = knn_points(
                _dst.transpose(2, 1), _src.transpose(2, 1), K=k, return_nn=False
            )  # B,N_dst,K; B,N_dst,K; B,N_dst,K,D
            dst_nn_in_src = torch.gather(
                src_f[..., None, :].expand(-1, -1, -1, N_dst, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        dst_f_padded = dst_f[..., None].expand_as(dst_nn_in_src)
        if cross:
            x_dir = F.normalize(src_f, dim=2)
            x_dir_padded = x_dir[..., None].expand_as(dst_nn_in_src)
            cross = torch.cross(x_dir_padded, dst_nn_in_src)
            y = torch.cat([cross, dst_nn_in_src - dst_f_padded, dst_f_padded], 1)
        else:
            y = torch.cat([dst_nn_in_src - dst_f_padded, dst_f_padded], 1)
        return y  # B,C*2,3,N,K

    def down_sample(self, x, f, factor):
        # x: B,1,3,N; f: B,C,3,N
        N_ori = x.shape[-1]
        N_new = N_ori // factor
        # down sample based on xyz
        with torch.no_grad():
            x_new, idx = sample_farthest_points(x.squeeze(1).transpose(2, 1), K=N_new)
        x_new = x_new.transpose(2, 1).unsqueeze(1).type(x.dtype)
        # idx: B,N
        C = f.shape[1]
        f_new = torch.gather(f, dim=-1, index=idx[:, None, None, :].expand(-1, C, 3, -1))
        f_new = f_new.type(x.dtype)
        return x_new, f_new

    def forward(self, x):
        B, _, N = x.shape
        feat_list, xyz_list = [], []

        # src means the larger point cloud, dst means the point to query
        src_xyz, src_f = x.unsqueeze(1), x.unsqueeze(1)
        # feat_list.append(src_f), xyz_list.append(src_xyz)

        for i in range(self.num_layers):
            # * down sampling
            if i in self.down_sample_layers:
                feat_list.append(src_f), xyz_list.append(src_xyz.squeeze(1).transpose(1, -1))
                dst_xyz, dst_f = self.down_sample(src_xyz, src_f, self.down_sample_factor[self.down_sample_layers.index(i)])

            else:
                dst_xyz, dst_f = src_xyz, src_f
            # * First get KNN feat
            num_knn = self.k if i > self.k_early_layers else self.k_early
            src_nn_f = self.get_graph_feature(src_f=src_f, dst_f=dst_f, k=num_knn, src_xyz=src_xyz, dst_xyz=dst_xyz, cross=i == 0)  # only has N_dst of K nn

            # * Apply MSG Passing
            if i < self.atten_start_layer:
                # message passing is through pooling
                dst_f = self.pool(self.V_list[i](src_nn_f))
            else:  # message passing is through QKV
                k = self.K_list[i](src_nn_f)  # B,C,3,N,K
                q = self.Q_list[i](dst_f)
                v = self.V_list[i](src_nn_f)
                k = channel_equi_vec_normalize(k)
                q = channel_equi_vec_normalize(q)
                qk = (k * q[..., None]).sum(2)  # B,C,N,K
                B, C, N, K = qk.shape
                N_head = C // self.atten_multi_head_c
                qk_c = qk.reshape(B, N_head, self.atten_multi_head_c, N, K)
                atten = qk_c.sum(2, keepdim=True) / np.sqrt(3 * self.atten_multi_head_c)
                atten = torch.softmax(atten, dim=-1)
                atten = atten.expand(-1, -1, self.atten_multi_head_c, -1, -1)
                atten = atten.reshape(qk.shape).unsqueeze(2)  # B,C,1,N,K
                dst_f = (atten * v).sum(-1)

            # * If use global point net, combine the global info
            if self.use_res_global_conv and i >= self.res_global_start_layer:
                dst_f_global = self.pool(dst_f)
                dst_f = torch.cat([dst_f, dst_f_global[..., None].expand_as(dst_f)], 1)
                dst_f = self.global_conv_list[i - self.res_global_start_layer](dst_f)

            src_xyz, src_f = dst_xyz, dst_f
            # if i in self.down_sample_layers:
            #     feat_list.append(src_f), xyz_list.append(src_xyz.squeeze(1).transpose(1, -1))

        x = self.conv_c(dst_f)
        x = x.mean(dim=-1, keepdim=False)

        feat_list.append(src_f), xyz_list.append(src_xyz.squeeze(1).transpose(1, -1))
        ############################ Up Sampling ####################
        f_encoder = feat_list.pop()
        features_on = self.up_layers[-1](f_encoder)
        for i in range(-1, -(len(self.down_sample_layers)+1), -1):
            dist, idx = three_nn(xyz_list[i-1].contiguous(), xyz_list[i].contiguous())
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            features_on = features_on.reshape(features_on.size(0), features_on.size(1) * features_on.size(2), features_on.size(3))
            f_interp_on = three_interpolate(features_on.contiguous(), idx.contiguous(), weight.contiguous())
            f_interp_on = f_interp_on.reshape(f_interp_on.size(0), -1, 3, f_interp_on.size(-1))

            f_encoder = feat_list.pop()
            f_on_skip = torch.cat([f_encoder, f_interp_on], dim=1)  # concat on channel dimension
            features_on = self.up_layers[i-1](f_on_skip)##[bs, c, 3, N]
        perpoint_feats = self.perpoint_feature(features_on) ## [bs, 32, 3, 256]
        ############################################################
        z_so3 = channel_equi_vec_normalize(x)  # without scale
        scale = x.norm(dim=-1).mean(1) * self.scale_factor

        perpoint_inv_feats = torch.einsum('bcmn,bmdo->bcdo', channel_equi_vec_normalize(perpoint_feats), z_so3.permute(0, 2, 1).unsqueeze(-1).expand(z_so3.shape[0], z_so3.shape[2], z_so3.shape[1], perpoint_feats.shape[-1]))#(channel_equi_vec_normalize(perpoint_feats),  z_so3.permute(0, 2, 1).unsqueeze(-1))
        # perpoint_inv_feats = self.fuse_MLP_fc1(perpoint_inv_feats.permute(0, 3, 1, 2).reshape(perpoint_feats.shape[0], perpoint_feats.shape[-1], -1)) ## [bs, N, 64]
        perpoint_inv_feats = self.fuse_MLP_fc1(perpoint_inv_feats.permute(0, 3, 1, 2).reshape(perpoint_feats.shape[0], perpoint_feats.shape[-1], -1)) ## [bs, N, 64]
        perpoint_inv_feats = self.fuse_MLP_bn(perpoint_inv_feats.permute(0, 2, 1))
        perpoint_inv_feats = self.fuse_MLP_fc2(perpoint_inv_feats.permute(0, 2, 1))
        # perpoint_inv_feats = channel_equi_vec_normalize(perpoint_feats).reshape(perpoint_feats.shape[0], -1, perpoint_feats.shape[-1])

        if self.z_so3_as_Omtx_flag:
            z_so3 = self.fc_O(z_so3)  # B,Basis,3
            R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
            _U, _, _Vh = torch.linalg.svd(R_pred.double())
            z_so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,num_basis,3

        if self.center_pred:
            center = self.fc_center(x[..., None]).squeeze(-1)
            if self.center_pred_scale:
                center = center * self.scale_factor
            return center, scale, z_so3, perpoint_inv_feats
        else:
            return scale, z_so3, perpoint_inv_feats



        # return features_final, x.mean(dim=-1, keepdim=True)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda")
