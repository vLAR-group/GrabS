# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.pointnet2_util import PointnetSAModule, PointnetFPModule

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=16):
        super().__init__()
        bn = {"class": "BatchNorm"}
        self.SA_modules = nn.ModuleList()
        self.Att_modules = nn.ModuleList()
        self.glob_modules = nn.ModuleList()

        self.fc0 = nn.Linear(3, input_feature_dim)
        self.bn0 = nn.BatchNorm1d(input_feature_dim)
        self.SA_modules.append(PointnetSAModule(npoint=512, radius=0.2, nsample=32, mlp=[input_feature_dim, 64], bn=bn, use_xyz=True))
        # self.Att_modules.append(SelfAttentionLayer(64, 8))
        # self.glob_modules.append(nn.Sequential(nn.Linear(64*2, 64), nn.LeakyReLU()))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=128, radius=0.4, nsample=32, mlp=[64, 128], bn=bn, use_xyz=True))
        self.Att_modules.append(SelfAttentionLayer(128, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(128*2, 128), nn.LeakyReLU()))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=32, radius=0.8, nsample=16, mlp=[128, 256], bn=bn, use_xyz=True))
        self.Att_modules.append(SelfAttentionLayer(256, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(256*2, 256), nn.LeakyReLU()))

        ####
        self.SA_modules.append(PointnetSAModule(npoint=16, radius=1.2, nsample=16, mlp=[256, 512//2], bn=bn, use_xyz=True))
        self.Att_modules.append(SelfAttentionLayer(512//2, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(512*1, 512//2), nn.LeakyReLU()))

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 128, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], bn=bn))


    def forward(self, xyz, features):
        features = F.leaky_relu(self.bn0(self.fc0(features).transpose(1, 2)).transpose(1, 2)) ## [B, N, C]
        l_pc, l_feats = [xyz], [features.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_pc, li_feats = self.SA_modules[i](l_pc[i], l_feats[i])
            if i >=1:
                li_feats = li_feats + self.Att_modules[i-1](li_feats.transpose(2, 1).contiguous()).transpose(2, 1).contiguous() ## [B, N, C]
                li_feats = self.glob_modules[i-1](torch.cat((li_feats.transpose(2, 1).contiguous(), li_feats.transpose(2, 1).contiguous().mean(1, keepdim=True).repeat(1, li_feats.shape[2], 1)), dim=-1)).transpose(2, 1).contiguous()
            l_pc.append(li_pc)
            l_feats.append(li_feats)
        return li_feats.transpose(1, 2).contiguous(), li_pc


class Pointnet2Backbone_tiny(nn.Module):
    def __init__(self, input_feature_dim=16):
        super().__init__()
        bn = {"class": "BatchNorm"}
        self.SA_modules = nn.ModuleList()
        self.Att_modules = nn.ModuleList()
        self.glob_modules = nn.ModuleList()

        self.fc0 = nn.Linear(3, input_feature_dim)
        self.bn0 = nn.BatchNorm1d(input_feature_dim)
        self.SA_modules.append(PointnetSAModule(npoint=256, radius=0.2, nsample=32, mlp=[input_feature_dim, 64], bn=bn, use_xyz=True))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=64, radius=0.4, nsample=32, mlp=[64, 128], bn=bn, use_xyz=True))
        self.Att_modules.append(SelfAttentionLayer(128, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(128 * 2, 128), nn.LeakyReLU()))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=16, radius=0.8, nsample=16, mlp=[128, 256], bn=bn, use_xyz=True))
        self.Att_modules.append(SelfAttentionLayer(256, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(256 * 2, 256), nn.LeakyReLU()))

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 128, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], bn=bn))

    def forward(self, xyz, features):
        features = F.leaky_relu(self.bn0(self.fc0(features).transpose(1, 2)).transpose(1, 2))  ## [B, N, C]
        l_pc, l_feats = [xyz], [features.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_pc, li_feats = self.SA_modules[i](l_pc[i], l_feats[i])
            if i >= 1:
                li_feats = li_feats + self.Att_modules[i - 1](li_feats.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()  ## [B, N, C]
                li_feats = self.glob_modules[i - 1](torch.cat((li_feats.transpose(2, 1).contiguous(),
                        li_feats.transpose(2, 1).contiguous().mean(1,keepdim=True).repeat(1, li_feats.shape[2], 1)), dim=-1)).transpose(2, 1).contiguous()
            l_pc.append(li_pc)
            l_feats.append(li_feats)
        return li_feats.transpose(1, 2).contiguous(), li_feats.transpose(1, 2).contiguous(), li_pc


class Pointnet2Backbone_tiny_noatten(nn.Module):
    def __init__(self, input_feature_dim=16):
        super().__init__()
        bn = {"class": "BatchNorm"}
        self.SA_modules = nn.ModuleList()
        self.Att_modules = nn.ModuleList()
        self.glob_modules = nn.ModuleList()

        self.fc0 = nn.Linear(3, input_feature_dim)
        self.bn0 = nn.BatchNorm1d(input_feature_dim)
        self.SA_modules.append(PointnetSAModule(npoint=256, radius=0.2, nsample=32, mlp=[input_feature_dim, 64], bn=bn, use_xyz=True))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=64, radius=0.4, nsample=32, mlp=[64, 128], bn=bn, use_xyz=True))
        # self.Att_modules.append(SelfAttentionLayer(128, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(128 * 2, 128), nn.LeakyReLU()))

        ###
        self.SA_modules.append(PointnetSAModule(npoint=16, radius=0.8, nsample=16, mlp=[128, 256], bn=bn, use_xyz=True))
        # self.Att_modules.append(SelfAttentionLayer(256, 8))
        self.glob_modules.append(nn.Sequential(nn.Linear(256 * 2, 256), nn.LeakyReLU()))

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 128, 128], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], bn=bn))

    def forward(self, xyz, features):
        features = F.leaky_relu(self.bn0(self.fc0(features).transpose(1, 2)).transpose(1, 2))  ## [B, N, C]
        l_pc, l_feats = [xyz], [features.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_pc, li_feats = self.SA_modules[i](l_pc[i], l_feats[i])
            if i >= 1:
                li_feats = self.glob_modules[i - 1](torch.cat((li_feats.transpose(2, 1).contiguous(),
                        li_feats.transpose(2, 1).contiguous().mean(1,keepdim=True).repeat(1, li_feats.shape[2], 1)), dim=-1)).transpose(2, 1).contiguous()
            l_pc.append(li_pc)
            l_feats.append(li_feats)
        return li_feats.transpose(1, 2).contiguous(), li_feats.transpose(1, 2).contiguous(), li_pc

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
