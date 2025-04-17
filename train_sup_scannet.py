import models.datasets.segnet_scannet_sup as voxelized_data
from models import training_mask3d_scannet
import logging
import argparse
import os
from omegaconf import DictConfig, OmegaConf
from mask3d_models import Res16UNet18A, Mask3D, Res16UNet14, Custom30M
import warnings
warnings.filterwarnings('ignore')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
### This code is to train all class segmentation on ScanNet, 18 classes

def config_parser():
    parser = argparse.ArgumentParser(description='scannet')
    parser.add_argument("--data_dir", type=str, default='./data/scannet/processed/')
    parser.add_argument("--save_path", type=str, default='./ckpt_sup_scannet_tmp/')
    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--optimizer", type=str, default='Adam', help='Optimizer used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--use_sp", type=bool, default=True)
    return parser.parse_args()

def main(cfg, logger):
    with open('./mask3d_models/mask3d.yaml', 'r') as file:
        model_cfg = OmegaConf.load(file)
    mask3d = mask3d_loading(model_cfg)
    #########################
    train_dataset = voxelized_data.VoxelizedDataset('train', cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size,
                                                    num_workers=8, voxel_size=cfg.voxel_size)
    val_dataset = voxelized_data.VoxelizedDataset('validation', cfg, data_path=cfg.data_dir, batch_size=1,
                                                  num_workers=4, voxel_size=cfg.voxel_size)
    #########################
    trainer = training_mask3d_scannet.Trainer(mask3d, logger, train_dataset, val_dataset, cfg.save_path, cfg)
    trainer.train_model(cfg.num_epochs)
    # trainer.validation(vis=True, log=False)


# @hydra.main(config_path="mask3d_models", config_name="mask3d.yaml")
def mask3d_loading(model_cfg: DictConfig):
    # mask3d = hydra.utils.instantiate(model_cfg)
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

    # os.system(f"cp {__file__} {cfg.save_path}")
    # os.system(f"cp -r {'./models/'} {cfg.save_path}")
    # os.system(f"cp -r {'./mask3d_models/'} {cfg.save_path}")

    main(cfg, logger)


# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.142   0.239   0.414 |   0.243   0.369   0.547 |   0.179   0.310   0.506
# ----------------------------------------------------------------------------------------------------
# average        :   0.142   0.239   0.414 |   0.243   0.369   0.547 |   0.179   0.310   0.506
#
#
# ####################################################################################################
# what           :      AP  AP_50%  AP_25% |      RC  RC_50%  RC_25% |      PR  PR_50%  PR_25%
# ####################################################################################################
# class_agnostic :   0.142   0.239   0.414 |   0.243   0.369   0.547 |   0.179   0.310   0.506
# ----------------------------------------------------------------------------------------------------
# average        :   0.142   0.239   0.414 |   0.243   0.369   0.547 |   0.179   0.310   0.506
# ####################################################################################################