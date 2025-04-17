import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.pointnet2 import *
from pointnet2.nn_util import SharedMLP


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc_xyz = nn.Conv2d(10, 64, kernel_size=1)
        self.fc = nn.Conv2d(d_in+64, d_in+64, kernel_size=1, bias=False)
        self.mlp = nn.Conv2d(d_in+64, d_out, kernel_size=1)
        self.bn = nn.BatchNorm2d(d_out)

    def forward(self, feature_set, xyz_set):
        #feature_set: ## [bs, C, N, K] [bs, 10, N, K]
        feature_set = torch.cat((self.fc_xyz(xyz_set), feature_set), dim=1)
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)## [bs, C, N, 1]
        f_agg = self.mlp(f_agg)
        f_agg = self.bn(f_agg)
        f_agg = F.leaky_relu(f_agg)
        return f_agg


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.fuse = None
        self.attentive_pooling = None

    def forward(self, xyz, features=None, return_inds=False):
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            new_inds = furthest_point_sample(xyz, self.npoint).long()
        new_xyz = (gather_nd(xyz_flipped, new_inds, t=True).transpose(1, 2).contiguous()
            if self.npoint is not None
            # else xyz_flipped.new_zeros((xyz_flipped.size(0), 1, 3))     # This matches original implementation.
            else None     # This matches original implementation.
        )

        for i in range(len(self.groupers)):
            new_features, neighbor_xyz = self.groupers[i](xyz, new_xyz, features)#[0]  # (B, C, npoint, nsample), [B, 3, N, K]

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # max_new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            # avg_new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            #
            # new_features = torch.cat((max_new_features, avg_new_features), dim=1)
            relative_xyz = neighbor_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
            dist = relative_xyz.norm(p=2, dim=1, keepdim=True) #[B, 1, N, K]
            xyz_set = torch.cat([dist, relative_xyz, new_xyz.transpose(1, 2).unsqueeze(-1).repeat(1, 1, 1, relative_xyz.shape[-1]), neighbor_xyz], dim=1)  # [bs, 10, N, K]
            new_features = self.attentive_pooling(new_features, xyz_set)

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features = self.fuse(new_features.transpose(1, 2)).transpose(1, 2)

            new_features_list.append(new_features)

        if return_inds:
            return new_xyz, torch.cat(new_features_list, dim=1), new_inds
        else:
            return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    def __init__(self, npoint, radii, nsamples, mlps, bn, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(mlp_spec, bn=bn))
            # self.fuse = nn.Linear(mlp_spec[-1]*2, mlp_spec[-1])
            self.fuse = nn.Linear(mlp_spec[-1], mlp_spec[-1])
            # if mlp_spec[0] == mlp_spec[-1]:
            #     self.shortcut = nn.Identity()
            # else:
            #     self.shortcut = nn.Linear(mlp_spec[0], mlp_spec[-1])
            self.attentive_pooling = Att_pooling(mlp_spec[-1], mlp_spec[-1])


class PointnetSAModule(PointnetSAModuleMSG):
    def __init__(
        self, mlp, npoint, radius, nsample, bn, use_xyz=True
            # Note: if npoint=radius=nsample=None then will be gather all operation.
    ):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    def __init__(self, mlp, bn):
        super(PointnetFPModule, self).__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        if known is not None:
            dist, idx = three_nn(unknown.contiguous(), known.contiguous())
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats.contiguous(), idx.contiguous(), weight.contiguous()
            )
        else:
            interpolated_feats = known_feats.expand(
                *(list(known_feats.size()[0:2]) + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)