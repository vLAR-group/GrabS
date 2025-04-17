import models.obj_model as model
import models.datasets.segnet_sys as voxelized_data
from models import training_ddpmseg_sys
import torch
import logging
import argparse
import os
from glob import glob
import numpy as np
from RLnet import PPO_actor, PPO_critic
from omegaconf import DictConfig, OmegaConf
from mask3d_models import Res16UNet18A, Mask3D, Res16UNet14, Custom30M
import warnings
warnings.filterwarnings('ignore')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))

def config_parser():
    parser = argparse.ArgumentParser(description='sys')
    parser.add_argument("--data_dir", type=str, default='./data/sys_scene/processed/')
    parser.add_argument("--objnet_dir", type=str, default='./objnet/multi-cate/')
    parser.add_argument("--sp_dir", type=str, default='./data/sys_scene/SPG_0.01')
    # parser.add_argument("--save_path", type=str, default='./ckpt_safe_check/unsup_diff/ckpt_10query_SPG_0.01_cylinder_rintial0.35_CD0.1/')
    parser.add_argument("--save_path", type=str, default='./segnet/sys_scene_ddpm')

    # Training Data Parameters
    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--use_norm", type=bool, default=False)
    parser.add_argument("--env_num", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=False)# print training details
    ##
    parser.add_argument("--max_diff_steps", type=int, default=1000)
    return parser.parse_args()

def main(cfg, logger):
    '''Prepare Data'''
    with open('./mask3d_models/mask3d_sys.yaml', 'r') as file:
        model_cfg = OmegaConf.load(file)
    mask3d = mask3d_loading(model_cfg)


    vae_checkpoints = glob(os.path.join(cfg.objnet_dir, 'vae') + '/*tar')
    vae_checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in vae_checkpoints]
    vae_checkpoints = np.array(vae_checkpoints, dtype=int)
    vae_checkpoints = np.sort(vae_checkpoints)
    vae_path = os.path.join(os.path.join(cfg.objnet_dir, 'vae'), 'checkpoint_{}.tar'.format(vae_checkpoints[-1]))
    print('Loaded vae checkpoint from: {}'.format(vae_path))

    diff_checkpoints = glob(os.path.join(cfg.objnet_dir, 'ddpm') + '/*tar')
    diff_checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in diff_checkpoints]
    diff_checkpoints = np.array(diff_checkpoints, dtype=int)
    diff_checkpoints = np.sort(diff_checkpoints)
    diff_path = os.path.join(os.path.join(cfg.objnet_dir, 'ddpm'), 'checkpoint_{}.tar'.format(diff_checkpoints[-1]))
    print('Loaded diff checkpoint from: {}'.format(diff_path))

    cond_net = model.Diffusion_cond().eval()
    cond_net.load_state_dict(torch.load(diff_path)['cond_net_state_dict'])
    diffuse_net = model.Diffusion_net(max_period=cfg.max_diff_steps).eval()
    diffuse_net.load_state_dict(torch.load(diff_path)['diffuse_net_state_dict'])
    VAE = model.PointNet2_wpos().eval()
    VAE.load_state_dict(torch.load(vae_path)['model_state_dict'])

    objnet = model.Diffusion_war(diffusion_net=diffuse_net, cond_net=cond_net, VAE=VAE).eval().cuda()

    n_actions = [4 + 1, 2 + 1]
    actor = PPO_actor(n_actions).cuda()
    critic = PPO_critic().cuda()
    #########################
    train_dataset = voxelized_data.VoxelizedDataset('train', cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size, num_workers=12, voxel_size=cfg.voxel_size)
    test_RL_dataset = voxelized_data.VoxelizedDataset('test', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=8, voxel_size=cfg.voxel_size, RL=True)
    test_dataset = voxelized_data.VoxelizedDataset('test', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=8, voxel_size=cfg.voxel_size)
    #########################
    # trainer = training_supseg_safe.Trainer(mask3d, logger, train_dataset, test_dataset, cfg.save_path, cfg, use_label=False)
    trainer = training_ddpmseg_sys.Trainer(mask3d, objnet, actor, critic, logger, train_dataset, test_dataset, test_RL_dataset, cfg.save_path, cfg, use_norm=cfg.use_norm, use_label=False)
    trainer.train_model(cfg.num_epochs)
    # trainer.validation_RL(vis=True, log=False)
    # trainer.validation(vis=False, log=False)
    # trainer.validation_pseudo(vis=True, log=False)


def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=3, out_channels=model_cfg.num_classes, out_fpn=True, config=model_cfg.config.backbone.config)
    # backbone = Res16UNet18A(in_channels=6, out_channels=model_cfg.num_classes, out_fpn=True, config=model_cfg.config.backbone.config)
    relevant_params = {key: value for key, value in model_cfg.items() if key in Mask3D.__init__.__code__.co_varnames}
    mask3d = Mask3D(backbone, **relevant_params)
    return mask3d

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    cfg = config_parser()

    '''Setup logger'''
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    logger = set_logger(os.path.join(cfg.save_path, 'train.log'))
    # #
    os.system(f"cp {__file__} {cfg.save_path}")
    os.system(f"cp -r {'./models/'} {cfg.save_path}")
    os.system(f"cp -r {'./mask3d_models/'} {cfg.save_path}")
    os.system(f"cp {'./RLnet.py'} {cfg.save_path}")

    main(cfg, logger)