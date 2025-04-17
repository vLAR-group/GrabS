import torch
import torch.nn as nn
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.VN_lib.vec_sim3.vec_dgcnn_atten import VecDGCNN_att
from models.VN_lib.implicit_func.onet_decoder import DecoderCat, DecoderCatBN
from models.backbone_module import Pointnet2Backbone, Pointnet2Backbone_tiny#, Pointnet2Backbone_tiny_noatten
import math


class Diffusion_war(nn.Module):
    def __init__(self, diffusion_net, cond_net, VAE):
        super(Diffusion_war, self).__init__()
        self.diffusion_net = diffusion_net
        self.cond_net = cond_net
        self.VAE = VAE


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
        return angles, R



class PointNet2_wpos(nn.Module):
    def __init__(self, hidden_dim=256, res=64):
        super(PointNet2_wpos, self).__init__()
        self.res = res
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
        z = z + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        if vae_training:
            return z, mu, log_var, pred_R_inv
        else:
            return z, pred_R_inv

    # def decode(self, p, features, mu, log_var, pred_R_inv=torch.eye(3)[None, ...].cuda()):
    def decode(self, p, features, pred_R_inv=None):
        if pred_R_inv is None:
            pred_R_inv = torch.eye(3)[None, ...].cuda().repeat(features.shape[0], 1, 1)
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
    def __init__(self, hidden_dim=256):
        super(Diffusion_cond, self).__init__()
        self.so3 = PoseNet()

        self.backbone = Pointnet2Backbone(input_feature_dim=16)
        self.mu = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))
        self.log_var = nn.Sequential(nn.Linear(512//2, 512//2), nn.LeakyReLU(), nn.Linear(512//2, 512//2))

        self.temp_feature = nn.Embedding(1, 512//2)
        self.decode_mlp = DecoderCat(input_dim=hidden_dim+3, hidden_size=hidden_dim)

    def forward(self, x):# [bs, N, 3]
        global_feature = self.backbone(x, x)[0].mean(1) ##[B, N, C]
        z = self.mu(global_feature) + self.temp_feature.weight.repeat(global_feature.shape[0], 1)
        return z


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
    def __init__(self, time_embedding_dim=256, max_period=10000):
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


class Diffusion_net_sum(nn.Module):
    def __init__(self, time_embedding_dim=256, max_period=10000):
        super(Diffusion_net_sum, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.max_period = max_period
        self.time_embedding = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256),
                                nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.fusion = nn.Sequential(nn.Linear(256, 256, bias=True), nn.LeakyReLU(), nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(), nn.Linear(256, 256, bias=False))

    def forward(self, noisy_feat, cond_feat, t):
        batchsize = noisy_feat.shape[0]
        time_embedding = timestep_embedding(t, self.time_embedding_dim, self.max_period).cuda()
        return self.fusion(noisy_feat+cond_feat+self.time_embedding(time_embedding))

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