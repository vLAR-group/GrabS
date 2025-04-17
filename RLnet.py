import torch.nn as nn
import torch
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling, MinkowskiMaxPooling
from mask3d_models.position_embedding import PositionEmbeddingCoordsSine
from torch.cuda.amp import autocast

pos_func = PositionEmbeddingCoordsSine(pos_type="sine", d_pos=128, gauss_scale=1.0, normalize=True).cuda()

class PPO_actor(nn.Module):
    def __init__(self, outputs=[5, 3]):
        super(PPO_actor, self).__init__()
        self.layer0 = nn.Linear(32 * 128, 128)

        self.layer1 = nn.Sequential(nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128), nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128), nn.ReLU())

        self.layer3_moving = nn.Linear(in_features=128, out_features=outputs[0])
        self.layer3_scale = nn.Linear(in_features=128, out_features=outputs[1])

        self.pooling = nn.Sequential(MinkowskiAvgPooling(kernel_size=1, stride=1, dimension=3))

        ### detector
        # self.point_features_head = nn.Linear(3, 128)#nn.Linear(256, 128)
        self.point_features_head = nn.Linear(256, 128)
        self.cross_atten = CrossAttentionLayer(d_model=128, nhead=8)
        self.self_atten = SelfAttentionLayer(d_model=128, nhead=8)
        self.ffa = FFNLayer(d_model=128, dim_feedforward=128)
        self.query = nn.Embedding(32, 128)
        self.query_pos = nn.Embedding(32, 128)
        ### if use sincos pos_encoding, need to proj by a layer
        self.proj_query_pos = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        #
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine", d_pos=128, gauss_scale=1.0, normalize=True).cuda()

    def forward(self, sampled_env_xyz, R, sampled_env_feats, curpos, history, non_parametric_queries=False):
        ### the sampled env_xyz should not be cnetered at curpos, so center it firstly
        sampled_env_xyz -= curpos
        with autocast(enabled=False):
            mins, maxs = sampled_env_xyz.min(dim=1)[0], sampled_env_xyz.max(dim=1)[0] ##[K, 3], [K, 3]
            maxs[maxs==mins]+=0.1
            ## sometimes, some max value will equals to their min value, like moved to a large ground, causing the later normalization be nan
            point_pos = self.pos_enc(sampled_env_xyz.float(), input_range=[mins, maxs]).permute((0, 2, 1))  # Batch, Dim, queries
        ### pos encoding if needed
        if non_parametric_queries:
            query_pos = self.pos_enc(torch.zeros_like(curpos).float(), input_range=[mins, maxs]).squeeze(2) # Batch, Dim, queries
            query_pos = self.proj_query_pos(query_pos)
        else:
            query_pos = self.query_pos.weight.unsqueeze(0).repeat(curpos.shape[0], 1, 1)

        # point_feats = self.point_features_head(sampled_env_xyz) ##[bs, N, 128]
        point_feats = self.point_features_head(sampled_env_feats)  ##[bs, N, 128]
        query = self.query.weight.unsqueeze(0).repeat(curpos.shape[0], 1, 1)
        ###
        output = self.cross_atten(query, point_feats, pos=point_pos, query_pos=query_pos)  ## new query features
        output = self.self_atten(output, query_pos=query_pos)  ## not clear whether to give query pos ???
        output = self.ffa(output).view(output.shape[0], -1)  ### [bs, 128] global feature of states
        output = self.layer0(output)
        ###
        # x = self.layer1(torch.cat((output, history), dim=-1))
        x = self.layer1(output)
        y = self.layer2(x)
        return self.layer3_moving(y), self.layer3_scale(y), x, output


class PPO_critic(nn.Module):
    def __init__(self):
        super(PPO_critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features=128, out_features=128), nn.BatchNorm1d(128), nn.ReLU())
        self.layer2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


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


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


