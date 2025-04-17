import models.obj_model as model
import models.datasets.segnet_s3dis as voxelized_data
from models import training_vaeseg_s3dis
import logging
import argparse
import os
import torch
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
    parser.add_argument("--save_path", type=str, default='./segnet/s3dis_vae_chair')

    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--use_norm", type=bool, default=False)
    parser.add_argument("--env_num", type=int, default=50)
    parser.add_argument("--verbose", type=bool, default=False)# print training details
    ##
    parser.add_argument("--cross_test", type=bool, default=True)# if scannet2s3dis test
    parser.add_argument("--cross_test_ckpt", type=str, default='ckpts/segnet/scannet_to_s3dis/diffseg/checkpoint_-1.tar')# path of ckpt trained on ScanNet
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

    objnet = model.PointNet2_wpos().eval().cuda()
    if not cfg.cross_test:
        objnet_checkpoints = glob(os.path.join(cfg.objnet_dir, 'vae') + '/*tar')
        objnet_checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in objnet_checkpoints]
        objnet_checkpoints = np.array(objnet_checkpoints, dtype=int)
        objnet_checkpoints = np.sort(objnet_checkpoints)
        path = os.path.join(os.path.join(cfg.objnet_dir, 'vae'), 'checkpoint_{}.tar'.format(objnet_checkpoints[-1]))
        print('Loaded checkpoint from: {}'.format(path))
        objnet.load_state_dict(torch.load(path)['model_state_dict'])

    n_actions = [4 + 1, 2 + 1]
    actor = PPO_actor(n_actions).cuda()
    critic = PPO_critic().cuda()
    #########################
    train_dataset = voxelized_data.VoxelizedDataset('train', training_areas, cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size, num_workers=8, voxel_size=cfg.voxel_size)
    val_RL_dataset = voxelized_data.VoxelizedDataset('validation', test_areas, cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size, RL=True)
    val_dataset = voxelized_data.VoxelizedDataset('validation', test_areas, cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size)
    #########################
    # trainer = training_supseg_s3dis.Trainer(mask3d, logger, train_dataset, val_dataset, cfg.save_path, cfg, use_label=False)
    trainer = training_vaeseg_s3dis.Trainer(mask3d, objnet, actor, critic, logger, train_dataset, val_dataset, val_RL_dataset, cfg.save_path, cfg, use_norm=cfg.use_norm, use_label=False)
    if not cfg.cross_test:
        trainer.train_model(cfg.num_epochs)
    else:
        trainer.validation(vis=False, log=False, ckpt_path=cfg.cross_test_ckpt)

    # trainer.validation_RL(vis=False, log=False)
    # trainer.validation(vis=False, log=False)
    # trainer.validation_pseudo(vis=True,s log=False)


def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=6, out_channels=model_cfg.num_classes, out_fpn=True, config=model_cfg.config.backbone.config)
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
