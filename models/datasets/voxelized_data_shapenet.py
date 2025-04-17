from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback
import os.path as osp
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat
from scipy.spatial import cKDTree as KDTree

class VoxelizedDataset(Dataset):
    def __init__(self, mode, res, pointcloud_samples, data_path, split_file ,
                 batch_size, num_sample_points, num_workers, sample_distribution, sample_sigmas):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        # self.split = np.load(split_file)

        self.mode = mode
        split_file = osp.join(split_file, '%s.lst' % (self.mode))
        with open(split_file, 'r') as f:
            model_files = f.read().split('\n')
            model_files = list(filter(lambda x: len(x) > 0, model_files))

        # with open('/home/user/SSD/ndf/02958343_names.txt', 'r') as file:
        #     model_files = file.read().split('\n')

        self.data = model_files#self.split[mode]
        # self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples

        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path = os.path.join(self.path, self.data[idx])
        # path = self.data[idx]
        input_path = path
        samples_path = path

        voxel_path = input_path + '/SDF_voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples)
        point_cloud = np.load(voxel_path)['point_cloud']
        if len(point_cloud) > 1024:
            sample_idx = np.random.choice(len(point_cloud), 1024, replace=False)
        else:
            sample_idx = np.random.choice(len(point_cloud), 1024, replace=True)
        point_cloud = point_cloud[sample_idx]

        # occupancies1 = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
        # input = np.reshape(occupancies1, (self.res,)*3)

        # if self.mode == 'test':
        #     # grid_points = self.create_grid_points_from_bounds(res=self.res)
        #     # kdtree = KDTree(grid_points)
        #     # occupancies = np.zeros(len(grid_points), dtype=np.int8)
        #     # _, idx = kdtree.query(point_cloud)
        #     # occupancies[idx] = 1
        #     # compressed_occupancies = np.reshape(occupancies, (self.res,) * 3)
        #     return {'inputs': np.array(input, dtype=np.float32), 'path' : path}#, 'occ': compressed_occupancies}

        points = []
        coords = []
        df = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = samples_path + '/PCU_boundary_{}_samples_100000.npz'.format( self.sample_sigmas[i])
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_df = boundary_samples_npz['df']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            df.extend(boundary_sample_df[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(df) == self.num_sample_points
        assert len(coords) == self.num_sample_points


        points = np.array(points, dtype=np.float32)
        df = np.array(df, dtype=np.float32)

        point_cloud += 0.005 * np.random.randn(*point_cloud.shape).astype(np.float32)
        # augmentation on the center
        centroid = point_cloud.mean(0).astype(np.float32)  # B,3
        noise = np.random.normal(0, 0.005, size=centroid.shape).astype(np.float32)
        centroid = centroid + noise
        point_cloud = point_cloud - centroid
        points = points - centroid

        # random_R = Rotation.random().as_matrix().astype(np.float32)
        # point_cloud = np.dot(point_cloud, random_R)
        # points = np.dot(points, random_R)
        #
        point_cloud_min, point_cloud_max = point_cloud.min(0, keepdims=True), point_cloud.max(0, keepdims=True)
        center, scale = (point_cloud_min + point_cloud_max)/2, (point_cloud_max - point_cloud_min).max()+1e-6 ## [bs, 1]

        point_cloud = (point_cloud-center)/scale
        points = (points-center)/scale
        df = df /scale

        # grid_points = self.create_grid_points_from_bounds(res=self.res)
        # kdtree = KDTree(grid_points)
        # occupancies = np.zeros(len(grid_points), dtype=np.int8)
        # _, idx = kdtree.query(point_cloud)
        # occupancies[idx] = 1
        # compressed_occupancies = np.reshape(occupancies, (self.res,) * 3)

        # return {'grid_coords':np.array(coords, dtype=np.float32),'df': np.array(df, dtype=np.float32), 'points':np.array(points, dtype=np.float32),
        #         'inputs': np.array(input, dtype=np.float32), 'point_cloud': np.array(point_cloud, dtype=np.float32), 'path' : path}#, 'occ': compressed_occupancies}
        return {'df': np.array(df, dtype=np.float32), 'points':np.array(points, dtype=np.float32), 'point_cloud': np.array(point_cloud, dtype=np.float32)}#, 'occ': compressed_occupancies}


    def get_loader(self, shuffle =True):
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def create_grid_points_from_bounds(self, minimun=-0.5, maximum=0.5, res=256):
        x = np.linspace(minimun, maximum, res)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        Z = Z.reshape((np.prod(Z.shape),))

        points_list = np.column_stack((Z, Y, X))
        del X, Y, Z, x
        return points_list
