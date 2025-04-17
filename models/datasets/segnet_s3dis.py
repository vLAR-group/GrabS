from torch.utils.data import Dataset
import os
import numpy as np
import torch
from glob import glob
import MinkowskiEngine as ME
from lib.aug_tools import rota_coords, scale_coords, trans_coords
import yaml
import pickle
import torch.nn.functional as F
import scipy

class VoxelizedDataset(Dataset):
    def __init__(self, mode, areas, cfg, data_path , batch_size, num_workers, voxel_size, RL=False):
        self.path = data_path
        self.mode = mode
        self.areas = areas
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_size = voxel_size
        self.ignore_label = -1#invalid semantic label
        self.limit_numpoints = 1200000

        self.rota_coords = rota_coords(rotation_bound = ((-0, 0), (-0, 0), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound = (0.9, 1.1))
        self.random_cuboid = RandomCuboid(min_points=20000, crop_length=6, version1=True)

        self.data = []
        ### here are total 6 areas
        for area in areas:
            scene_list = sorted(glob(os.path.join(data_path, area, '*.npy')))
            mean_std = self.load_yaml(os.path.join(self.path, area + '_color_mean_std.yaml'))
            for scene_name in scene_list:
                points = np.load(scene_name)
                semantic = points[:, 10:11]
                if RL and self.mode=='validation':
                    if (semantic==9).sum()>0:
                        self.data.append((scene_name, (mean_std['mean'], mean_std['std'], 255)))
                else:
                    self.data.append((scene_name, (mean_std['mean'], mean_std['std'], 255)))

        self.above_num = 0
        self.total_num = 0

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

    def __len__(self):
        return len(self.data)

    def load_yaml(self, filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader = yaml.FullLoader)
        return file

    def __getitem__(self, idx):
        path, (mean, std, max_pixel_value) = self.data[idx]
        points = np.load(path)
        if self.mode == 'train':
            inds = self.random_cuboid(points)
            points2 = points[inds]
            points = points2
        ### this sp_idx is invalid, and the intance is not scannet format, only puerly instance id, not containg category information (from 0-?)
        ### also the normal vector is also invalid
        pc, color, _, sp_idx, semantic, instance = points[:, :3], points[:, 3:6], points[:, 6:9], points[:, 9], points[:, 10:11], points[:, 11:12]
        pc = pc - pc.min(0)

        if self.mode == 'train':
            pc[:, 0:2] = pc[:, 0:2] + (np.random.uniform(pc.min(0), pc.max(0))/ 2)[0:2][None, ...]
            for i in (0, 1):
                if np.random.random() < 0.5:
                    pc_max = np.max(pc[:, i])
                    pc[:, i] = pc_max - pc[:, i]
            # if np.random.random() < 0.95:
            #     for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
            #         pc = self.elastic_distortion(pc, granularity, magnitude)
            pc = self.scale_coords(pc)
            pc = self.rota_coords(pc)
        pc = pc.astype(np.float32)

        # normalize color information
        mean = np.array(mean)*max_pixel_value
        std = np.array(std)*max_pixel_value
        color = (color - mean)*np.reciprocal(std)

        feature = np.concatenate([color, pc], 1)
        # feature = pc
        coords, feature, semantic, instance, unique_map, inverse_map = self.voxelize(pc, feature, semantic, instance)
        feature, semantic, instance, pc, pc_full = torch.from_numpy(feature), torch.from_numpy(semantic), torch.from_numpy(instance), torch.from_numpy(pc[unique_map]), torch.from_numpy(pc)
        normals = _
        normals = F.normalize(torch.from_numpy(normals), dim=-1)
        area_name = self.data[idx][0].split('/')[-2]
        scene_name = area_name + '/' +self.data[idx][0].split('/')[-1][0:-4]

        #### my superpoints
        if self.cfg.sp_dir is not None:
            mysp_name = os.path.join(self.cfg.sp_dir, scene_name + '_superpoint.npy')### numpy array [N, 1]
            if self.mode == 'train':
                mysp = np.load(mysp_name)[np.where(inds)]
            else:
                mysp = np.load(mysp_name)
            if len(mysp.shape) == 2:
                mysp = mysp.squeeze(1)
            sp_idx = mysp
        ###

        sp_idx_voxel = sp_idx[unique_map]
        sp_idx_voxel_copy = -np.ones_like(sp_idx_voxel)
        valid_sp_idx = sp_idx_voxel[sp_idx_voxel != -1]
        unique_vals = np.unique(valid_sp_idx)
        unique_vals.sort()
        sp_idx_voxel_copy[sp_idx_voxel != -1] = np.searchsorted(unique_vals, valid_sp_idx)
        sp_idx_voxel = sp_idx_voxel_copy

        ## exist mask
        exist_mask_file = os.path.join(self.cfg.save_path, 'exist_pseudo', scene_name+'.pickle')
        if os.path.exists(exist_mask_file):
            try:
                with open(exist_mask_file, 'rb') as f:
                    data = pickle.load(f)
                exist_mask = [data[0][unique_map], data[1], inds]
            except:
                print('removing: ', exist_mask_file)
                os.system(f"rm -r {exist_mask_file}")
                exist_mask = [torch.zeros((len(semantic), 1)).bool(), torch.tensor(0).unsqueeze(-1)]
        else:
            exist_mask = [torch.zeros((len(semantic), 1)).bool(), torch.tensor(0).unsqueeze(-1)]
        return coords, feature, normals, semantic.squeeze(), instance.squeeze(), inverse_map, unique_map, scene_name, pc, pc_full, torch.from_numpy(sp_idx_voxel).long(), torch.from_numpy(sp_idx).long(), exist_mask

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
        coords, feature, normals, semantic, instance, inverse_map, unique_map, scene_name, pc, pc_full, sp_idx, sp_idx_full, exist_mask = list(zip(*batch))
        coords_batch, feature_batch, instance_batch, pc_batch, pc_batch_full, sp_batch, sp_batch_full = [], [], [], [], [], [], []
        target = []
        semantic_batch =[]
        normals_batch = []
        exist_mask_batch = []
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
            # print(batch_num_points-num_points)
            coords_batch.append(torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1))
            feature_batch.append((feature[batch_id]))
            normals_batch.append(normals[batch_id])
            pc_batch.append(pc[batch_id])
            pc_batch_full.append(pc_full[batch_id])
            sp_batch.append(sp_idx[batch_id])
            sp_batch_full.append(sp_idx_full[batch_id])
            exist_mask_batch.append(exist_mask[batch_id])

            #### sem label 0 is bg
            instance_batch.append((instance[batch_id] + (semantic[batch_id])*1000))
            ### only use for chair(sem_label:0)
            semantic_batch.append((semantic[batch_id]))#.unsqueeze(0)))
            target.append(dict())
            masks, labels, segment_masks = [], [], []
            ##### if use sp
            valid_sp_mask = sp_idx[batch_id]!=-1
            _, ret_index, ret_inv = np.unique(sp_idx[batch_id][valid_sp_mask].numpy(), return_index=True, return_inverse=True)
            sp_instance_label = instance[batch_id][ret_index]
            ####
            for instance_id in torch.unique(instance[batch_id]):
                if instance_id == -1:
                    continue

                mask = (instance[batch_id] == instance_id).bool()
                tmp_semantic = semantic[batch_id]
                label = torch.mode(tmp_semantic[mask]).values

                if label ==8:
                    masks.append(mask.unsqueeze(0))
                    # labels.append(label.unsqueeze(0).long())
                    labels.append(torch.zeros_like(label.unsqueeze(0).long()))
                    segment_masks.append((sp_instance_label == instance_id).bool().unsqueeze(0))

            if len(masks)>0:
                target[batch_id]['labels'] = torch.cat(labels)
                target[batch_id]['masks'] = torch.cat(masks, dim=0).squeeze(-1)
                target[batch_id]['segment_mask'] = torch.cat(segment_masks, dim=0).squeeze(-1)
                # self.above_num += (torch.cat(masks, dim=0).squeeze(-1).sum(-1)>100).sum()
                # self.total_num += torch.cat(masks, dim=0).squeeze(-1).shape[0]
                # print('cccccccccccccccccccccc', self.above_num/(self.total_num))
            else:
                target[batch_id]['labels'] = []
                target[batch_id]['masks'] = torch.zeros_like(instance[batch_id])[None, :]
                target[batch_id]['segment_mask'] = []

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        feature_batch = torch.cat(feature_batch, 0).float()

        return coords_batch, feature_batch, normals_batch, target, scene_name, semantic_batch, instance_batch, inverse_map, unique_map, pc_batch, pc_batch_full, sp_batch, sp_batch_full, exist_mask_batch





class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    We slightly modify this operation to account for object detection.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

    def __init__(
        self,
        min_points,
        # aspect=0.8,
        crop_length=6.0,
        version1=True,
    ):
        # self.aspect = aspect
        self.crop_length = crop_length
        self.min_points = min_points
        self.version1 = version1

    def __call__(self, point_cloud):
        if point_cloud.shape[0] < self.min_points:
            print("too small pcd")
            return np.ones(point_cloud.shape[0], dtype=np.bool)

        range_xyz = np.max(point_cloud[:, :2], axis=0) - np.min(
            point_cloud[:, :2], axis=0
        )

        for _ in range(100):

            sample_center = point_cloud[:, :2].min(axis=0) + range_xyz / 2

            if self.version1:
                offset_x = np.random.uniform(
                    -range_xyz[0] / 4, range_xyz[0] / 4
                )
                offset_y = np.random.uniform(
                    -range_xyz[1] / 4, range_xyz[1] / 4
                )
            else:
                offset_x = np.random.uniform(
                    -(range_xyz[0] / 2) + self.crop_length / 4,
                    +(range_xyz[0] / 2) - self.crop_length / 4,
                )
                offset_y = np.random.uniform(
                    -(range_xyz[1] / 2) + self.crop_length / 4,
                    +(range_xyz[1] / 2) - self.crop_length / 4,
                )

            # sample_center[0] = sample_center[0] + offset_x
            # sample_center[1] = sample_center[1] + offset_y

            min_xy = sample_center - self.crop_length / 2
            max_xy = sample_center + self.crop_length / 2

            upper_idx = (
                np.sum((point_cloud[:, :2] <= max_xy).astype(np.int32), 1) == 2
            )
            lower_idx = (
                np.sum((point_cloud[:, :2] >= min_xy).astype(np.int32), 1) == 2
            )

            new_pointidx = (upper_idx) & (lower_idx)

            if np.sum(new_pointidx) < self.min_points:
                print("TOO SMALL")
                continue

            return new_pointidx

        # fallback
        print("FALLBACK")
        return np.ones(point_cloud.shape[0], dtype=np.bool)
