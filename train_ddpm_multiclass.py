import models.obj_model as model
import models.datasets.objnet_multiclass as voxelized_data
from models import training_ddpm
import torch
import logging
import argparse
import os
from glob import glob
import numpy as np

def config_parser():
    parser = argparse.ArgumentParser(description='sdf')
    parser.add_argument("--exp_name", type=str, default='shapenet_multiclass')
    parser.add_argument("--data_dir", type=str, default='./GAPS_SDF/')
    parser.add_argument("--save_path", type=str, default='./objnet/multi-cate/')
    parser.add_argument("--split_file_folder", type=str, default='./data/shapenet_splits/data/')
    parser.add_argument("--mix_file_dir", type=str, default='./data/other_cls_data/')
    parser.add_argument("--mix_split_file", type=str, default='./data/shapenet_splits/data/')
    # Training Data Parameters
    parser.add_argument("--num_sample_points_training", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_diff_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.') ### notice the EFEMSDF lr is 1e-4
    parser.add_argument("--stage", type=str, default='ddpm')
    return parser.parse_args()

def main(cfg, logger):
    vae_checkpoints = glob(os.path.join(cfg.save_path, 'vae') + '/*tar')
    vae_checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in vae_checkpoints]
    vae_checkpoints = np.array(vae_checkpoints, dtype=int)
    vae_checkpoints = np.sort(vae_checkpoints)
    path = os.path.join(os.path.join(cfg.save_path, 'vae'), 'checkpoint_{}.tar'.format(vae_checkpoints[-1]))
    print('Loaded checkpoint from: {}'.format(path))

    cond_net = model.Diffusion_cond()
    cond_net.load_state_dict(torch.load(path)['model_state_dict'])

    diffuse_net = model.Diffusion_net(max_period=cfg.max_diff_steps)

    VAE = model.PointNet2_wpos()
    VAE.load_state_dict(torch.load(path)['model_state_dict'])

    cfg.save_path = os.path.join(cfg.save_path, 'ddpm')
    train_dataset = voxelized_data.VoxelizedDataset('train', data_path=cfg.data_dir, split_file_folder=cfg.split_file_folder,
                    batch_size=cfg.batch_size, num_sample_points=cfg.num_sample_points_training, num_workers=8, mix_file_dir=cfg.mix_file_dir, mix_splits=cfg.mix_split_file)
    val_dataset = voxelized_data.VoxelizedDataset('val', data_path=cfg.data_dir, split_file_folder=cfg.split_file_folder,
                    batch_size=cfg.batch_size, num_sample_points=cfg.num_sample_points_training, num_workers=4, mix_file_dir=cfg.mix_file_dir, mix_splits=cfg.mix_split_file)

    trainer = training_ddpm.Trainer(cond_net, diffuse_net, VAE, logger, train_dataset, val_dataset, cfg.save_path, cfg.lr, cfg, cfg.max_diff_steps)
    trainer.train_model(cfg.num_epochs)


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
    if not os.path.exists(os.path.join(cfg.save_path, cfg.stage)):
        os.makedirs(os.path.join(cfg.save_path, cfg.stage))
    logger = set_logger(os.path.join(cfg.save_path, cfg.stage, 'train.log'))

    os.system(f"cp {__file__} {os.path.join(cfg.save_path, cfg.stage)}")
    os.system(f"cp -r {'./models/'} {os.path.join(cfg.save_path, cfg.stage)}")

    main(cfg, logger)
