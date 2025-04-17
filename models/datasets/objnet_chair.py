from torch.utils.data import Dataset
import os
import numpy as np
import torch
import os.path as osp
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat
from lib.helper_ply import write_ply
import glob


class VoxelizedDataset(Dataset):
    def __init__(self, mode, data_path, split_file , batch_size, num_sample_points, num_workers, mix_file_dir, mix_splits):

        self.path = data_path

        self.mode = mode
        split_file = osp.join(split_file, '%s.lst' % (self.mode))
        with open(split_file, 'r') as f:
            model_files = f.read().split('\n')
            model_files = list(filter(lambda x: len(x) > 0, model_files))

        self.count = len(model_files)
        self.data = model_files
        self.mix_file_dir = mix_file_dir
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_config()

        if mix_file_dir is not None:
            mix_data = {}
            mix_classes = os.listdir(mix_file_dir)
            for mix_class in mix_classes:
                if mix_class in ['02828884', '02933112', '03211117', '03636649', '04256520', '04379243']:
                    mix_split_file = osp.join(os.path.join(mix_splits, mix_class), '%s.lst' % (self.mode))
                    with open(mix_split_file, 'r') as f:
                        mix_model_files = f.read().split('\n')
                        mix_model_files = list(filter(lambda x: len(x) > 0, mix_model_files))
                        mix_data[mix_class] = mix_model_files
            self.mix_data = mix_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.path, self.data[idx])
        samples_path = path

        voxel_path = path + '/pointcloud.npz'
        clean_surface_points = np.load(voxel_path)['points'].astype(np.float32)

        if self.mode == 'train':
            # on_surface_points = np.load(voxel_path)['points'].astype(np.float32)
            occ_path = os.path.join("/".join(self.path.split('/')[:-2]), self.path.split('/')[-2] + '_dep', self.data[idx])
            files = sorted(glob.glob(os.path.join(occ_path, '*npz')))
            pcl_list = []
            n_views = np.random.randint(low=1, high=13)

            for file_idx in np.random.choice(12, n_views):
                file = files[int(file_idx)]
                pcl_list.append(np.load(file, allow_pickle=True)['p_w'].astype(np.float32))
            on_surface_points = np.concatenate(pcl_list, 0)
            if len(on_surface_points)>1024:
                sample_idx = np.random.choice(len(on_surface_points), 1024, replace=True)
            else:
                sample_idx = np.random.choice(len(on_surface_points), 1024, replace=True)
            on_surface_points = on_surface_points[sample_idx]

        nss_samples_path = samples_path + '/points_nss.npz'
        nss_samples = np.load(nss_samples_path)['points'].astype(np.float32)
        uni_samples_path = samples_path + '/points_uni.npz'
        uni_samples = np.load(uni_samples_path)['points'].astype(np.float32)

        # nss_indices, uni_indices = np.random.choice(len(nss_samples), self.num_sample_points//2), np.random.choice(len(uni_samples), self.num_sample_points//2)
        nss_indices, uni_indices = np.random.choice(len(nss_samples), int(self.num_sample_points*0.9)), np.random.choice(len(uni_samples), int(self.num_sample_points*0.1))
        off_surface_points = [nss_samples[nss_indices][:, 0:3], uni_samples[uni_indices][:, 0:3]]
        df = [nss_samples[nss_indices][:, 3], uni_samples[uni_indices][:, 3]]

        off_surface_points = np.concatenate(off_surface_points, axis=0)
        df = np.concatenate(df, axis=0)


        if self.mode == 'val':
            on_surface_points = np.load(voxel_path)['points'].astype(np.float32)
            if len(on_surface_points) > 1024:
                sample_idx = np.random.choice(len(on_surface_points), 1024, replace=True)
            else:
                sample_idx = np.random.choice(len(on_surface_points), 1024, replace=True)
            on_surface_points = on_surface_points[sample_idx]
            centroid = on_surface_points.mean(0)  # np.zeros_like(on_surface_points.mean(0))
            # centroid = clean_surface_points.mean(0).astype(np.float32)  # B,3
            on_surface_points = on_surface_points - centroid
            off_surface_points = off_surface_points - centroid
            clean_surface_points = clean_surface_points[np.random.choice(len(clean_surface_points), 1024, replace=True)].astype(np.float32) - centroid

            on_surface_points_min, on_surface_points_max = on_surface_points.min(0, keepdims=True), on_surface_points.max(0, keepdims=True)
            _, scale = (on_surface_points_min + on_surface_points_max) / 2, (on_surface_points_max - on_surface_points_min).max() + 1e-6  ## [bs, 1]
            on_surface_points, off_surface_points, df, clean_surface_points = on_surface_points / scale, off_surface_points / scale, df / scale, clean_surface_points/scale

            random_R_z = np.random.uniform(-1*np.pi, 1*np.pi)
            random_R_x = np.random.uniform(-np.pi / 12, np.pi / 12)
            random_R_y = np.random.uniform(-np.pi / 12, np.pi / 12)
            random_R = np.array([random_R_z, random_R_y, random_R_x]).astype(np.float32)
            random_R2 = Rotation.from_euler('zyx', random_R).as_matrix().astype(np.float32)
            on_surface_points, off_surface_points, clean_surface_points = on_surface_points[:, [0, 2, 1]], off_surface_points[:, [0, 2, 1]], clean_surface_points[:, [0, 2, 1]]

            on_surface_points = np.dot(on_surface_points, random_R2)
            off_surface_points = np.dot(off_surface_points, random_R2)

            return {'on_surface_points': on_surface_points, 'df': df, 'off_surface_points': off_surface_points, 'path': path, 'clean_surface_points':clean_surface_points,
                    'centroid': np.array(centroid, dtype=np.float32), 'scale': np.array(scale, dtype=np.float32), 'so3': np.array(random_R, dtype=np.float32), 'R': random_R2}

        on_surface_points += 0.005 * np.random.randn(*on_surface_points.shape).astype(np.float32)
        on_surface_points, fg_labels = self.augment_v1(on_surface_points, off_surface_points, df, bottom_y=on_surface_points.min(0)[1])
        on_surface_points, fg_labels = on_surface_points.astype(np.float32), fg_labels.astype(np.float32)
        #
        # augmentation on the center
        centroid = on_surface_points.mean(0).astype(np.float32)  # B,3
        # centroid = clean_surface_points.mean(0).astype(np.float32)  # B,3
        noise = np.random.normal(0, 0.005, size=centroid.shape).astype(np.float32)
        centroid = centroid + noise
        on_surface_points = on_surface_points - centroid
        off_surface_points = off_surface_points - centroid
        clean_surface_points = clean_surface_points[np.random.choice(len(clean_surface_points), 1024, replace=True)].astype(np.float32) - centroid

        on_surface_points_min, on_surface_points_max = on_surface_points.min(0, keepdims=True), on_surface_points.max(0, keepdims=True)
        _, scale = (on_surface_points_min + on_surface_points_max)/2, (on_surface_points_max - on_surface_points_min).max()+1e-6 ## [bs, 1]
        #
        on_surface_points, off_surface_points, df, clean_surface_points = on_surface_points/scale, off_surface_points/scale, df/scale, clean_surface_points/scale

        random_R_z = np.random.uniform(-1 * np.pi, 1 * np.pi)
        random_R_x = np.random.uniform(-np.pi / 12, np.pi / 12)
        random_R_y = np.random.uniform(-np.pi / 12, np.pi / 12)
        random_R = np.array([random_R_z, random_R_y, random_R_x]).astype(np.float32)
        random_R2 = Rotation.from_euler('zyx', random_R).as_matrix().astype(np.float32)
        on_surface_points, off_surface_points, clean_surface_points = on_surface_points[:, [0, 2, 1]], off_surface_points[:, [0, 2, 1]], clean_surface_points[:, [0, 2, 1]]

        on_surface_points = np.dot(on_surface_points, random_R2)
        off_surface_points = np.dot(off_surface_points, random_R2)

        return {'on_surface_points': on_surface_points,'df': df, 'off_surface_points': off_surface_points, 'path': path, 'clean_surface_points':clean_surface_points,
                'centroid': np.array(centroid, dtype=np.float32), 'scale': np.array(scale, dtype=np.float32), 'so3': np.array(random_R, dtype=np.float32), 'R': random_R2}

    def get_loader(self, shuffle =True):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def aug_config(self):
        self.aug_ratio = 0.6
        self.random_object_prob = 0.7
        self.random_object_radius = 0.15
        self.random_object_radius_std = 0.07
        self.random_object_center_near_surface = True
        self.random_object_center_L = 0.15
        self.random_object_scale = [0.5, 1.5]
        self.random_plane_prob = 0.5
        self.random_plane_vertical_prob = 0.5
        self.random_plane_vertical_scale = [0.05, 0.5]
        self.random_plane_vertical_height_range = [0.4, 1.0]
        self.random_plane_vertical_horizon_range = [0.4, 0.5]
        self.random_plane_ground_scale = [0.4, 1.0]
        self.random_plane_ground_range = 0.2
        self.random_ball_removal_prob = 0.6
        self.random_ball_removal_max_k = 50
        self.random_ball_removal_noise_std = 0.05

    def ball_removal(self, pcl, n, noise_std):
        anchor = pcl[np.random.choice(len(pcl), 1)]
        d = ((pcl - anchor) ** 2).sum(-1)
        d_noise = np.random.normal(0.0, noise_std, size=len(d))
        d += d_noise
        idx = d.argsort()[:n]
        return idx

    def ball_crop(self, pcl, radius):
        seed = np.random.choice(len(pcl), 1)
        ball_d = np.linalg.norm(pcl - pcl[seed], axis=-1)
        ball_pts = pcl[ball_d < radius]
        return ball_pts

    def augment_v1(self, pcl, points, points_sdf, bottom_y=None):
        N = pcl.shape[0]
        N_aug_max = int(self.aug_ratio * N)
        N_aug = int(np.random.rand() * N_aug_max)
        if N_aug == 0:
            return pcl, np.ones(N)

        random_seed = np.random.rand(3)
        aug_mask = random_seed <= np.array([self.random_object_prob, self.random_plane_prob, self.random_ball_removal_prob])
        if not aug_mask.any():
            return pcl, np.ones(N)

        flag_obj, flag_ground, flag_ball = aug_mask
        # flag_obj = False
        if bottom_y is None:
            bottom_y = pcl[:, 1].min()

        total_remove = N_aug
        N_random_noise_fallback = 0
        if flag_obj and flag_ground:
            N_other_obj = int(np.random.rand() * N_aug)
            N_ground = N_aug - N_other_obj
        elif flag_obj:
            N_other_obj = N_aug
            N_ground = 0
        elif flag_ground:
            N_other_obj = 0
            N_ground = N_aug
        else:
            N_other_obj = 0
            N_ground = 0
            N_random_noise_fallback = N_aug

        aug_main_pcl = pcl

        # * random crop out some point cloud
        if flag_ball:
            # remove some ball from the point cloud
            remove_ratio = np.random.rand()
            if remove_ratio>0.5:
                remove_ratio=0.5
            n_ball_removal = int(remove_ratio * N_aug)
            cnt_ball_removed = 0
            while cnt_ball_removed < n_ball_removal:
                removal_idx = self.ball_removal(aug_main_pcl, min(self.random_ball_removal_max_k, n_ball_removal - cnt_ball_removed), self.random_ball_removal_noise_std)
                cnt_ball_removed += len(removal_idx)
                aug_main_pcl = np.delete(aug_main_pcl, removal_idx, axis=0)
            total_remove -= cnt_ball_removed
        # remove other points
        removal_idx = np.random.choice(len(aug_main_pcl), total_remove, replace=False)
        aug_main_pcl = np.delete(aug_main_pcl, removal_idx, axis=0)

        # * random add some other object's part near outside the object
        ########################33 cost loading time #######################
        AUG_LIST = []
        if N_other_obj > 0:
            cnt_object_added = 0
            other_obj_aug_list = []
            while cnt_object_added < N_other_obj:
                other_cls_id = int(np.random.choice(len(self.mix_data.keys()), 1))
                other_cls = list(self.mix_data.keys())[other_cls_id]
                other_obj_id = int(np.random.choice(len(self.mix_data[other_cls]), 1))
                other_obj = self.mix_data[other_cls][other_obj_id]
                pointcloud = np.load(os.path.join(self.mix_file_dir, other_cls, other_obj) + '/pointcloud.npy')#.astype(np.float32)

                other_pcl = self.ball_crop(pointcloud[np.random.choice(len(pointcloud), len(pcl))],
                    radius=max(self.random_object_radius + np.random.normal(0.0, self.random_object_radius_std), 0.01))
                other_pcl = other_pcl - other_pcl.mean(0)[None, ...]
                random_scale = (np.random.rand() * (self.random_object_scale[1] - self.random_object_scale[0]) + self.random_object_scale[0])
                other_pcl = random_scale * other_pcl
                other_pcl_r = np.linalg.norm(other_pcl, axis=-1).max()

                # for _i in range(10):
                #     if self.random_object_center_near_surface:
                #         random_center = aug_main_pcl[np.random.choice(len(aug_main_pcl), 1)] + np.random.normal(loc=0.0, scale=self.random_object_center_L, size=(3))
                #     else:
                #         random_center = (np.random.rand(3) - 0.5) * 2 * self.random_object_center_L
                #         random_center = random_center[None, ...]
                #     random_center_d = np.linalg.norm(points - random_center, axis=-1)
                #     random_center_nearest_values = points_sdf[random_center_d.argmin()]
                #     if random_center_nearest_values > other_pcl_r:
                #         break
                # random_center = random_center.squeeze(0)
                # random_rotation = Rotation.random().as_matrix()
                iter = 10
                if self.random_object_center_near_surface:
                    random_center = aug_main_pcl[np.random.choice(len(aug_main_pcl), iter)] + np.random.normal(loc=0.0, scale=self.random_object_center_L, size=(iter, 3))
                else:
                    random_center = (np.random.rand(iter, 3) - 0.5) * 2 * self.random_object_center_L
                    random_center = random_center[None, ...]
                random_center_d = np.linalg.norm(points[:, None, :] - random_center[None, :, :], axis=-1)#[5000, iter]
                random_center_nearest_values = points_sdf[random_center_d.argmin(0)]#[100]
                index = np.argmax(random_center_nearest_values > other_pcl_r)# using np.argmax find the first value meeting the condition
                random_center = random_center[index]
                random_rotation = Rotation.random().as_matrix()

                other_pcl = other_pcl @ random_rotation + random_center[None, :]
                other_obj_aug_list.append(other_pcl)
                cnt_object_added += len(other_pcl)
            AUG_LIST.append(np.concatenate(other_obj_aug_list, 0)[:N_other_obj])
        ##############################################################################################
        # * random add the ground or other planes
        if N_ground > 0:
            n_ground = N_ground
            if np.random.rand() < self.random_plane_vertical_prob:
                # if True:
                n_vertical = int(np.random.rand() * N_ground)
                n_ground = N_ground - n_vertical
                random_uv = (np.random.rand(n_vertical * 2).reshape(n_vertical, 2) - 0.5) * 2
                random_scale = (np.random.rand()* (self.random_plane_vertical_scale[1] - self.random_plane_vertical_scale[0]) + self.random_plane_vertical_scale[0])
                random_height = (np.random.rand()* (self.random_plane_vertical_height_range[1] - self.random_plane_vertical_height_range[0])+ self.random_plane_vertical_height_range[0])
                vertical_pcl = np.zeros((n_vertical, 3))
                vertical_pcl[:, :2] = random_uv * random_scale
                vertical_pcl[:, 1] += random_height + bottom_y
                random_vertical_rotation = euler2mat(np.random.rand() * np.pi * 2, 0.0, 0.0, "syzx")
                vertical_pcl = (random_vertical_rotation[None, ...] @ vertical_pcl[..., None]).squeeze(-1)
                random_vertical_center_r = (np.random.rand()* (self.random_plane_vertical_horizon_range[1]-
                        self.random_plane_vertical_scale[0]) + self.random_plane_vertical_horizon_range[0])
                random_vertical_center_angle = np.random.rand() * np.pi * 2
                vertical_pcl[:, 0] += (np.cos(random_vertical_center_angle) * random_vertical_center_r)
                vertical_pcl[:, 2] += (np.sin(random_vertical_center_angle) * random_vertical_center_r)
                AUG_LIST.append(vertical_pcl)
            if n_ground > 0:
                # add ground
                random_uv = (np.random.rand(n_ground * 2).reshape(n_ground, 2) - 0.5) * 2
                random_scale = (np.random.rand()* (self.random_plane_ground_scale[1] - self.random_plane_ground_scale[0])
                    + self.random_plane_ground_scale[0])
                random_center = (np.random.rand(2) - 0.5) * 2 * self.random_plane_ground_range
                ground_pcl = np.zeros((n_ground, 3))
                ground_pcl[:, 1] += bottom_y
                ground_pcl[:, [0, 2]] = random_uv * random_scale + random_center[None, :]
                AUG_LIST.append(ground_pcl)
        if N_random_noise_fallback > 0:
            AUG_LIST.append(np.random.rand(N_random_noise_fallback * 3).reshape(N_random_noise_fallback, 3)- 0.5)
        AUG_LIST.append(aug_main_pcl)
        aug_pcl = np.concatenate(AUG_LIST, 0)
        # # debug
        # np.savetxt("./debug/aug.txt", aug_pcl)
        assert aug_pcl.shape[0] == N
        fg_label = np.zeros(aug_pcl.shape[0])
        fg_label[-aug_main_pcl.shape[0]:] = 1
        return aug_pcl, fg_label#, N_aug

