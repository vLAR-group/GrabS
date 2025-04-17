from torch.utils.data import Dataset
import os
import numpy as np
import torch
from glob import glob
from lib.helper_ply import write_ply, read_ply
import MinkowskiEngine as ME
from lib.aug_tools import rota_coords, scale_coords, trans_coords
import scipy
import yaml
from preprocessing.scannet200_constants import VALID_CLASS_IDS_20

class VoxelizedDataset(Dataset):
    def __init__(self, mode, cfg, data_path , batch_size, num_workers, voxel_size):
        self.path = data_path
        self.mode = mode
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_size = voxel_size
        self.ignore_label = -1#invalid semantic label
        self.limit_numpoints = 1500000
        self.label_offset = 2
        self.filter_out_classes = [0, 1]


        self.data = []
        scene_list = sorted(glob(os.path.join(data_path, mode, '*.npy')))
        for scene_name in scene_list:
            self.data.append(scene_name)

        mean_std = self.load_yaml(os.path.join(self.path, 'color_mean_std.yaml'))
        self.mean, self.std, self.max_pixel_value = mean_std['mean'], mean_std['std'], 255

        # self.trans_coords = trans_coords(shift_ratio=2)
        self.rota_coords = rota_coords(rotation_bound = ((-np.pi/320, np.pi/320), (-np.pi/320, np.pi/320), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))

    def __len__(self):
        return len(self.data)

    def load_yaml(self, filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader = yaml.FullLoader)
        return file

    def map_NYU_label(self, labels):
        labels[~np.isin(labels, list(VALID_CLASS_IDS_20))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(VALID_CLASS_IDS_20):
            labels[labels == k] = i
        return labels

    def remap_model_output(self, output):
        output = np.array(output)
        output_remapped = output.copy()
        for i, k in enumerate(VALID_CLASS_IDS_20):
            output_remapped[output == i] = k
        return output_remapped

    def elastic_distortion(self, pointcloud, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords = pointcloud[:, :3]
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim)]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    def __getitem__(self, idx):
        path = self.data[idx]
        points = np.load(path)
        pc, color, normals, sp_idx, semantic, instance = points[:, :3], points[:, 3:6], points[:, 6:9], points[:, 9], points[:, 10:11], points[:, 11:12]
        # # read efem gt
        # if self.mode=='train':
            # efemgt_file = '/home/zihui/SSD/MyEFEM/log/scannet_train/results_eval/scannet_format/scene'+ path.split('/')[-1][: -4] + '.txt'
            # instance = -np.ones_like(instance)
            # with open(efemgt_file, 'r') as f:
            #     data = f.readlines()
            # for mask_id, mask_file in enumerate(data):
            #     mask_file = mask_file.split('txt')[0]+'txt'
            #     mask_file = os.path.join('/home/zihui/SSD/MyEFEM/log/scannet_train/results_eval/scannet_format/', mask_file)
            #     mask = np.loadtxt(mask_file)
            #     instance[mask==1] = (mask_id+1)

            # instance = -np.ones_like(instance)
            # exist_mask_file = '/home/zihui/SSD/GOPS/ckpt_scannet_obj/VAE_wpos_shape2scan/5classSDF_CD0.14_maxh2.2_maxr1.5_IniCyl_max0.6/exist_pseudo/scene'+ path.split('/')[-1][: -4] + '.pickle'
            # with open(exist_mask_file, 'rb') as f:
            #     data = pickle.load(f)
            # mask = data[0].numpy()
            # for mask_id in range(mask.shape[1]):
            #     instance[mask[:, mask_id]==1] = (mask_id+1)

        pc = pc - pc.min(0)
        if self.mode == 'train':
            pc = pc + (np.random.uniform(pc.min(0), pc.max(0))/ 2)
            for i in (0, 1):
                if np.random.random() < 0.5:
                    pc_max = np.max(pc[:, i])
                    pc[:, i] = pc_max - pc[:, i]
            pc = self.scale_coords(pc)
            pc = self.rota_coords(pc)
            # if np.random.random() < 0.95:
            #     for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
            #         pc = self.elastic_distortion(pc, granularity, magnitude)

        # normalize color information
        mean = np.array(self.mean)*self.max_pixel_value
        std = np.array(self.std)*self.max_pixel_value
        color = (color - mean)*np.reciprocal(std)

        # prepare labels and map from 0 to 20(40)
        semantic = self.map_NYU_label(semantic)
        for filter_class in self.filter_out_classes:
            if (semantic==filter_class).sum()>0:
                semantic[semantic == filter_class] = self.ignore_label
        semantic[semantic!=self.ignore_label] = np.clip(semantic[semantic!=self.ignore_label] - self.label_offset, a_min=0, a_max=None)### semantic are 0-17,-1, -1 is bg 1-17 are categories
        # semantic = np.clip(semantic - self.label_offset, a_min=0, a_max=None)### semantic are 0-17,-1, -1 is bg 1-17 are categories

        feature = np.concatenate([color, pc], 1)
        coords, feature, semantic, instance, unique_map, inverse_map = self.voxelize(pc, feature, semantic, instance)

        feature, semantic, instance, pc = torch.from_numpy(feature), torch.from_numpy(semantic), torch.from_numpy(instance), torch.from_numpy(pc)
        normals = torch.from_numpy(normals)
        scene_name = 'scene' + self.data[idx].split('/')[-1][0:-4]
        return coords, feature, normals, semantic, instance, inverse_map, scene_name, pc, torch.from_numpy(sp_idx[unique_map]).long(), torch.from_numpy(sp_idx).long()

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=shuffle, worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def voxelize(self, coords, feature, semantic, instance):
        scale = 1 / self.voxel_size
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), ignore_label=-1, return_index=True, return_inverse=True)
        return coords.numpy(), feature[unique_map], semantic[unique_map], instance[unique_map], unique_map, inverse_map

    def collate_fn(self, batch):
        coords, feature, normals, semantic, instance, inverse_map, scene_name, pc, sp_idx, sp_idx_full = list(zip(*batch))
        coords_batch, feature_batch, instance_batch, pc_batch, sp_batch, sp_batch_full = [], [], [], [], [], []
        target = []
        semantic_batch =[]
        normals_batch = []
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                print(f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points ' f'limit. '
                      f'Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.')
                break
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feature_batch.append((feature[batch_id]))
            normals_batch.append(normals[batch_id].unsqueeze(0))
            pc_batch.append(pc[batch_id])#.unsqueeze(0))
            sp_batch.append(sp_idx[batch_id])
            sp_batch_full.append(sp_idx_full[batch_id])

            #### sem label 0 is bg
            instance_batch.append((instance[batch_id] + (semantic[batch_id])*1000))## valid instance_id starts from 0001
            semantic_batch.append((semantic[batch_id].unsqueeze(0)))
            target.append(dict())
            masks, labels, segment_masks = [], [], []
            ### if use sp
            _, ret_index, ret_inv = np.unique(sp_idx[batch_id].numpy(), return_index=True,return_inverse=True)
            sp_instance_label = instance[batch_id][ret_index]

            ####
            for instance_id in torch.unique(instance[batch_id]):
                if instance_id == -1:
                    continue

                mask = (instance[batch_id] == instance_id).bool()
                tmp_semantic = semantic[batch_id]
                label = torch.mode(tmp_semantic[mask]).values

                # efem gt
                masks.append(mask.unsqueeze(0))
                labels.append(label.unsqueeze(0).long())
                segment_masks.append((sp_instance_label == instance_id).bool().unsqueeze(0))
                #
            if len(masks)>0:
                target[batch_id]['labels'] = torch.cat(labels)
                target[batch_id]['masks'] = torch.cat(masks, dim=0).squeeze(-1)
                target[batch_id]['segment_mask'] = torch.cat(segment_masks, dim=0).squeeze(-1)
            else:
                target[batch_id]['labels'] = []
                target[batch_id]['masks'] = []
                target[batch_id]['segment_mask'] = []

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feature_batch = torch.cat(feature_batch, 0).float()
        # normals_batch = torch.cat(normals_batch, 0).float()
        # semantic_batch = torch.cat(semantic_batch, 0).float()
        # instance_batch = torch.cat(instance_batch, 0).float()
        # pc_batch = torch.cat(pc_batch, 0).float()

        return coords_batch, feature_batch, normals_batch, target, scene_name, semantic_batch, instance_batch, inverse_map, pc_batch, sp_batch, sp_batch_full
