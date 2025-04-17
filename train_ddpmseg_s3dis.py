import models.obj_model as model
import models.datasets.segnet_s3dis as voxelized_data
from models import training_ddpmseg_s3dis
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
    parser = argparse.ArgumentParser(description='s3dis')
    parser.add_argument("--data_dir", type=str, default='./data/s3dis_align/processed/')
    parser.add_argument("--objnet_dir", type=str, default='./objnet/chair/')
    parser.add_argument("--sp_dir", type=str, default='./data/s3dis_align/SPG_0.05')
    parser.add_argument("--save_path", type=str, default='./segnet/s3dis_ddpm_chair')

    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--use_norm", type=bool, default=False)
    parser.add_argument("--env_num", type=int, default=50)
    parser.add_argument("--verbose", type=bool, default=False)
    ##
    parser.add_argument("--max_diff_steps", type=int, default=1000)
    return parser.parse_args()

def main(cfg, logger):
    '''Prepare Data'''
    all_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    test_areas = ['Area_5']
    training_areas = sorted(list(set(all_areas) - set(test_areas)))
    logger.info('Training Areas: %s', training_areas)

    with open('./mask3d_models/mask3d_s3dis.yaml', 'r') as file:
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
    train_dataset = voxelized_data.VoxelizedDataset('train', training_areas, cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size, num_workers=8, voxel_size=cfg.voxel_size)
    val_RL_dataset = voxelized_data.VoxelizedDataset('validation', test_areas, cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size, RL=True)
    val_dataset = voxelized_data.VoxelizedDataset('validation', test_areas, cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size)
    #########################
    trainer = training_ddpmseg_s3dis.Trainer(mask3d, objnet, actor, critic, logger, train_dataset, val_dataset, val_RL_dataset, cfg.save_path, cfg, use_norm=cfg.use_norm, use_label=False)
    trainer.train_model(cfg.num_epochs)
    # trainer.validation_RL(vis=True, log=False)
    # trainer.validation(vis=False, log=False)
    # trainer.validation_pseudo(vis=True, log=False)


def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=6, out_channels=model_cfg.num_classes, out_fpn=True, config=model_cfg.config.backbone.config)
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