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


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecDGCNN_att(nn.Module):
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
        res = 256,
        sem_pred = False):
        super().__init__()

        self.sem_pred = sem_pred
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

        self.conv_c = VecLNA(feat_dim[-1], c_dim, mode="so3", act_func=act_func)#, shared_nonlinearity=True)

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
        if self.sem_pred:
            self.sem_classifier = nn.Sequential(nn.Linear(c_dim, c_dim), nn.ReLU(), nn.BatchNorm1d(c_dim), nn.Linear(c_dim, c_dim), nn.ReLU(), nn.Linear(c_dim, 1))
            self.sem = nn.Sequential(VecLNA(c_dim, c_dim, mode="so3", act_func=act_func),#, shared_nonlinearity=True),
                                    VecLNA(c_dim, c_dim, mode="so3", act_func=act_func))#, shared_nonlinearity=True))
            self.fc_inv_sem = VecLinear(c_dim, c_dim, mode="so3")

    def get_graph_feature(
        self,
        src_f: torch.Tensor,
        dst_f: torch.Tensor,
        k: int,
        src_xyz: torch.Tensor,
        dst_xyz: torch.Tensor,
        cross=False,
        label=None):
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
            # y = torch.cat([cross, dst_nn_in_src - dst_f_padded, dst_nn_in_src - dst_f_padded], 1)
        # else:
        #     y = torch.cat([dst_nn_in_src - dst_f_padded, dst_nn_in_src - dst_f_padded], 1)

        # label_map = torch.gather(label.unsqueeze(-1).expand(-1, -1, N_dst), dim=1, index=knn_idx)#[bs, 1024, 16]
        return y#, label_map  # B,C*2,3,N,K

    def down_sample(self, x, f, factor, label=None):
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
        # if label is not None:
        #     label_new = torch.gather(label, dim=-1, index=idx)#[bs, N_new]
        #     f_new = f_new * label_new[:, None, None, :]
        #     return x_new, f_new, label_new
        # else:
        #     return x_new, f_new, None
        return x_new, f_new#, torch.gather(label, dim=-1, index=idx)#[bs, N_new]


    def forward(self, x, label=None):
        B, _, N = x.shape#[bs, 3, 1024]

        if label is None:
            label= torch.ones_like(x)[:, 0, :]

        feat_list, xyz_list = [], []

        # src means the larger point cloud, dst means the point to query
        # src_xyz, src_f = x.unsqueeze(1), (x*label.transpose(1, 2)).unsqueeze(1)#x.unsqueeze(1)
        src_xyz, src_f = x.unsqueeze(1), x.unsqueeze(1)
        src_label = label
        feat_list.append(src_f), xyz_list.append(src_xyz)

        for i in range(self.num_layers):
            # * down sampling
            if i in self.down_sample_layers:
                feat_list.append(src_f), xyz_list.append(src_xyz)
                # dst_xyz, dst_f, dst_label = self.down_sample(src_xyz, src_f, self.down_sample_factor[self.down_sample_layers.index(i)], src_label)
                dst_xyz, dst_f = self.down_sample(src_xyz, src_f, self.down_sample_factor[self.down_sample_layers.index(i)])

            else:
                dst_xyz, dst_f, dst_label = src_xyz, src_f, src_label

            # * First get KNN feat
            num_knn = self.k if i > self.k_early_layers else self.k_early
            # src_nn_f, label_map = self.get_graph_feature(src_f=src_f, dst_f=dst_f, k=num_knn, src_xyz=src_xyz, dst_xyz=dst_xyz, cross=i == 0, label=src_label)  # only has N_dst of K nn
            src_nn_f = self.get_graph_feature(src_f=src_f, dst_f=dst_f, k=num_knn, src_xyz=src_xyz, dst_xyz=dst_xyz, cross=i == 0)  # only has N_dst of K nn

            # * Apply MSG Passing
            if i < self.atten_start_layer:
                # message passing is through pooling
                dst_f = self.pool(self.V_list[i](src_nn_f))
                # dst_f = (self.V_list[i](src_nn_f)*label_map[:, None, None, :, :]).sum(dim=-1, keepdim=False)/(label_map[:, None, None, :, :].sum(dim=-1, keepdim=False)+1e-12)
            else:  # message passing is through QKV
                k = self.K_list[i](src_nn_f)  # B,C,3,N,K
                q = self.Q_list[i](dst_f)
                v = self.V_list[i](src_nn_f)
                ##
                # k = (k * label_map[:, None, None, :, :])#, (q * label_map[:, None, None, :, :])
                ##
                k = channel_equi_vec_normalize(k)
                q = channel_equi_vec_normalize(q)
                qk = (k * q[..., None]).sum(2)  # B,C,N,K
                B, C, N, K = qk.shape
                N_head = C // self.atten_multi_head_c
                qk_c = qk.reshape(B, N_head, self.atten_multi_head_c, N, K)
                atten = qk_c.sum(2, keepdim=True) / np.sqrt(3 * self.atten_multi_head_c)
                ##
                atten = torch.softmax(atten, dim=-1)
                atten = atten.expand(-1, -1, self.atten_multi_head_c, -1, -1)
                atten = atten.reshape(qk.shape).unsqueeze(2)  # B,C,1,N,K
                ##
                # atten = atten*label_map[:, None, None, :, :]
                # atten  = atten/(atten.sum(dim=-1, keepdim=True)+1e-12)
                # v = (v*label_map[:, None, None, :, :])#/(label_map[:, None, None, :, :].sum(dim=-1, keepdim=False)+1e-10)
                ##
                dst_f = (atten * v).sum(-1)

            # * If use global point net, combine the global info
            if self.use_res_global_conv and i >= self.res_global_start_layer:
                dst_f_global = self.pool(dst_f)
                dst_f = torch.cat([dst_f, dst_f_global[..., None].expand_as(dst_f)], 1)
                dst_f = self.global_conv_list[i - self.res_global_start_layer](dst_f)

            # feat_list.append(dst_f), xyz_list.append(dst_xyz)

            src_xyz, src_f, src_label = dst_xyz, dst_f, dst_label
            # if i in self.down_sample_layers or i==1:
        # feat_list.append(src_f), xyz_list.append(src_xyz)

        #############################
        # mean = self.mean(dst_f).mean(dim=-1, keepdim=False)
        # logvar = self.logvar(dst_f).mean(dim=-1, keepdim=False)
        #
        #
        # so3_mean = self.fc_inv_mean(mean[..., None]).squeeze(-1)#[bs, C, 3]
        # scale_mean = self.fc_scale_mean(mean[..., None]).squeeze(-1).pow(2).sum(dim=-1).mean(1, keepdim=True)
        # inv_mean = (mean * so3_mean).sum(-1)/scale_mean
        #
        # so3_logvar = self.fc_inv_logvar(logvar[..., None]).squeeze(-1)#[bs, C, 3]
        # scale_logvar = self.fc_scale_logvar(logvar[..., None]).squeeze(-1).pow(2).sum(dim=-1).mean(1, keepdim=True)
        # inv_logvar = (logvar * so3_logvar).sum(-1)/scale_logvar
        #
        # return inv_mean, inv_logvar


        x = self.conv_c(dst_f)
        y = x.mean(dim=-1, keepdim=False)
        # y = (x * src_label[:, None, None, :]).sum(dim=-1, keepdim=False) / src_label[:, None, None, :].sum(dim=-1, keepdim=False)
        so3 = channel_equi_vec_normalize(y)
        ###############
        if self.z_so3_as_Omtx_flag:
            z_so3 = self.fc_O(so3)  # B,Basis,3
            R_pred = z_so3.transpose(2, 1)  # B,3,num_basis
            _U, _, _Vh = torch.linalg.svd(R_pred.double())
            so3 = (_U @ _Vh).transpose(2, 1).type(R_pred.dtype)  # B,num_basis,3
        ##################
        scale = (y.norm(dim=-1)).mean(1) * self.scale_factor
        # print(scale)
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
        if self.sem_pred:
            sem = self.sem(y)  # .mean(dim=-1, keepdim=False)
            so3_sem = self.fc_inv_sem(sem[..., None]).squeeze(-1)  # [bs, C, 3]
            inv_sem = (sem * so3_sem).sum(-1)
            return inv_mean, inv_logvar, so3, scale, center, self.sem_classifier(inv_sem).squeeze()
        else:
            return inv_mean, inv_logvar, so3, scale, center


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    torch.set_default_dtype(torch.float64)

    device = torch.device("cuda")
    B, N = 16, 512
    pcl = torch.rand(B, 3, N).to(device)  # .double()
    net = VecDGCNN_att(
        num_layers=7,
        feat_dim=[32, 32, 64, 64, 128, 256, 512],
        down_sample_layers=[2, 4, 5],
        down_sample_factor=[4, 4, 4],
        atten_start_layer=2,
        atten_multi_head_c=16,
        use_res_global_conv=True,
        res_global_start_layer=2,
        use_dg=True,
    ).to(device)
    net.eval()

    with torch.no_grad():
        scale, so3_feat, inv_feat = net(pcl)

        for _ in range(10):
            t = torch.rand(B, 3, 1).to(device) - 0.5
            t = t * 0
            R = [torch.from_numpy(Rotation.random().as_matrix()) for _ in range(B)]
            R = torch.stack(R, 0).to(device).type(scale.dtype)
            s = torch.rand(B).to(device)
            # s = torch.ones(B).to(device)
            aug_pcl = torch.einsum("bij,bjn->bin", R.clone(), pcl * s[:, None, None]) + t

            aug_scale_hat, aug_so3_feat_hat, aug_inv_feat_hat = net(aug_pcl)

            aug_scale = scale * s
            aug_so3_feat = torch.einsum("bij,bnj->bni", R.clone(), so3_feat)

            error_so3_feat = torch.einsum("bij,bkj->bik", aug_so3_feat, aug_so3_feat_hat)
            error_so3_feat = (
                torch.acos(
                    torch.clamp(
                        (
                            error_so3_feat[:, 0, 0]
                            + error_so3_feat[:, 1, 1]
                            + error_so3_feat[:, 2, 2]
                            - 1.0
                        )
                        / 2.0
                        - 1.0,
                        1.0,
                    )
                )
                / np.pi
                * 180.0
            )
            error_so3_feat = error_so3_feat.max()

            error_inv_feat = (abs(aug_inv_feat_hat - inv_feat)).max()

            error_scale = abs(aug_scale - aug_scale_hat).max()

            print(f"so3_feat error {error_so3_feat} deg")
            print(f"Inv_feat error {error_inv_feat}")
            print(f"Scale error {error_scale}")
            print("-" * 20)
