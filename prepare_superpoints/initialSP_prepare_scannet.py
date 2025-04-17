from pclpy import pcl
import pclpy
import numpy as np
from scipy import stats, spatial
import os
from os.path import join, exists, dirname, abspath
import sys
from glob import glob
import torch

from lib.helper_ply import read_ply, write_ply
import time
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import Pool
import multiprocessing as mp
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))


colormap = []
for _ in range(10000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)

import functools
from typing import List, Tuple
import colorsys

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))


input_path = './data/scannet/processed/train/'
sp_save_path = './data/scannet/growsp5/'
ignore_label = -1
voxel_size = 0.05

vis = True
if not exists(sp_save_path):
    os.makedirs(sp_save_path)
os.system(f"cp {__file__} {sp_save_path}")

def supervoxel_clustering(coords, rgb=None, normals=None):
    pc = pcl.PointCloud.PointXYZRGBA(coords, rgb)

    if normals is not None:
        normals_np = normals.copy().astype(np.float32)
        normals = pcl.PointCloud.Normal()
        normals.resize(normals_np.shape[0])
        for i in range(normals_np.shape[0]):
            normals.points[i].normal_x = normals_np[i, 0]
            normals.points[i].normal_y = normals_np[i, 1]
            normals.points[i].normal_z = normals_np[i, 2]
    else:
        normals = pc.compute_normals(radius=3, num_threads=2)

    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(voxel_resolution=1, seed_resolution=5)
    vox.setInputCloud(pc)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(0.4)
    vox.setNormalImportance(1)
    vox.setColorImportance(0.2)
    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    return list(output.items())

def region_growing_simple(coords, normals=None):
    pc = pcl.PointCloud.PointXYZ(coords)

    if normals is not None:
        normals_np = normals.copy().astype(np.float32)
        normals = pcl.PointCloud.Normal()
        normals.resize(normals_np.shape[0])
        for i in range(normals_np.shape[0]):
            normals.points[i].normal_x = normals_np[i, 0]
            normals.points[i].normal_y = normals_np[i, 1]
            normals.points[i].normal_z = normals_np[i, 2]
    else:
        normals = pc.compute_normals(radius=3, num_threads=2)

    clusters = pclpy.region_growing(pc, normals=normals, min_size=1, max_size=100000, n_neighbours=15, smooth_threshold=5, curvature_threshold=1, residual_threshold=1)
    return clusters, normals.normals


def instance2masks(sp2gt, instance, semantic, scene_name):
    sp2gt, instance, semantic = sp2gt.squeeze(1), instance.squeeze(1), semantic.squeeze(1)## now only for chair
    ious = []
    has_target_category=False
    for instance_id in torch.unique(instance):
        if instance_id == -1:
            continue

        mask_gt = (instance == instance_id).bool()
        mask_sp2gt = (sp2gt == instance_id).bool()
        label = torch.mode(semantic[mask_gt]).values

        if label == 5:
            has_target_category = True
            inter_area = (mask_gt*mask_sp2gt).sum()
            union_area = mask_gt.sum() + mask_sp2gt.sum() - inter_area
            iou = inter_area / (union_area + 1e-5)
            ious.append(iou)
    if has_target_category:
        return ious
    else:
        print('no target category in: '+scene_name)
        return []


def construct_superpoints(path):
    f = Path(path)
    if os.path.exists(sp_save_path + '/' + f.parts[-1][0:-4] + '_superpoint.npy'):
        scene_name = f.parts[-2] +'/'+ f.parts[-1][0:-4]
        points = np.load(path)
        pc, color, _, _, semantic, instance = points[:, :3], points[:, 3:6], points[:, 6:9], points[:,9], points[:,10:11], points[:,11:12]
        sp2gt = np.load(sp_save_path + '/' + scene_name + '_superpoint.npy')
    else:
        scene_name = f.parts[-2] +'/'+ f.parts[-1][0:-4]
        points = np.load(path)
        pc, color, _, _, semantic, instance = points[:, :3], points[:, 3:6], points[:, 6:9], points[:, 9], points[:,10:11], points[:,11:12]
        pc = pc - pc.mean(0)

        time_start = time.time()
        '''Voxelize'''
        scale = 1 / voxel_size
        coords = np.floor(pc * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), color, labels=instance, ignore_label=-1, return_index=True, return_inverse=True) ### labels is not used later
        coords = coords.numpy().astype(np.float32)

        '''VCCS'''
        out = supervoxel_clustering(coords, feats)#, normals)
        voxel_idx = -np.ones_like(labels)
        voxel_num = 0
        for voxel in range(len(out)):
            if out[voxel][1].voxels_.xyz.shape[0] >= 0:
                for xyz_voxel in out[voxel][1].voxels_.xyz:
                    index_colum = np.where((xyz_voxel == coords).all(1))
                    voxel_idx[index_colum] = voxel_num
                voxel_num += 1

        '''Region Growing'''
        clusters = region_growing_simple(coords)[0]#, normals)[0]
        region_idx = -1 * np.ones_like(labels)
        for region in range(len(clusters)):
            for point_idx in clusters[region].indices:
                region_idx[point_idx] = region
        #
        '''Merging'''
        merged = -np.ones_like(labels)
        voxel_idx[voxel_idx != -1] += len(clusters)
        for v in np.unique(voxel_idx):
            if v != -1:
                voxel_mask = v == voxel_idx
                voxel2region = region_idx[voxel_mask] ### count which regions are appeared in current voxel
                dominant_region = stats.mode(voxel2region)[0][0]
                if (dominant_region == voxel2region).sum() > voxel2region.shape[0] * 0.5:
                    merged[voxel_mask] = dominant_region
                else:
                    merged[voxel_mask] = v

        # merged = region_idx  # voxel_idx

        '''Make Superpoint Labels Continuous'''
        sp_labels = -np.ones_like(merged)
        count_num = 0
        for m in np.unique(merged):
            if m != -1:
                sp_labels[merged == m] = count_num
                count_num += 1

        '''ReProject to Input Point Cloud'''
        out_sp_labels = sp_labels[inverse_map]
        out_coords = pc
        out_labels = instance
        #
        if not os.path.exists(sp_save_path + '/' + f.parts[-2]):
            os.makedirs(sp_save_path + '/' + f.parts[-2])
        np.save(sp_save_path + '/' + scene_name + '_superpoint.npy', out_sp_labels)

        if vis:
            vis_path = sp_save_path +'/vis/'
            if not os.path.exists(vis_path + '/' + f.parts[-2]):
                os.makedirs(vis_path+ '/' + f.parts[-2])
            colors = np.zeros_like(out_coords)
            for p in range(colors.shape[0]):
                colors[p] = 255 * (colormap[out_sp_labels[p].astype(np.int32)].squeeze())[:3]
            colors = colors.astype(np.uint8)
            write_ply(vis_path + '/' + scene_name + '.ply', [out_coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

        sp2gt = -np.ones_like(out_labels)
        for sp in np.unique(out_sp_labels):
            if sp != -1:
                sp_mask = sp == out_sp_labels
                sp2gt[sp_mask] = stats.mode(out_labels[sp_mask])[0][0]
        print('completed scene: {}, used time: {:.2f}s'.format(scene_name, time.time() - time_start))
    # print(sp2gt.shape, instance.shape, )
    ious = instance2masks(torch.from_numpy(sp2gt), torch.from_numpy(instance), torch.from_numpy(semantic), scene_name)## [K, N]
    return (scene_name, ious)


print('start constructing initial superpoints')

path_list = sorted(glob(os.path.join(input_path, '*.npy')))

pool = ProcessPoolExecutor(max_workers=10)
result = list(pool.map(construct_superpoints, path_list))

print('end constructing initial superpoints')
all_iou = []
for (scene_name, ious) in result:
    all_iou += ious
print('Mean IoU on target category', sum(all_iou)/len(all_iou))