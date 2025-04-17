from scipy import stats, spatial
import os
from os.path import join, exists, dirname, abspath
import sys
from glob import glob
import torch
BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c
from partition.graphs import *

from lib.helper_ply import read_ply, write_ply
import time
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from pathlib import Path
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


input_path = './data/s3dis_align/processed/'
sp_save_path = './data/s3dis_align/SPG_0.05/'
ignore_label = -1
voxel_size = 0.05

k_nn_geof = 45
k_nn_adj = 10
lambda_edge_weight = 1.
reg_strength = 0.05

vis = True
if not exists(sp_save_path):
    os.makedirs(sp_save_path)
os.system(f"cp {__file__} {sp_save_path}")


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

        if label == 8:
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
        pc = pc - pc.min(0)

        time_start = time.time()
        '''Voxelize'''
        scale = 1 / voxel_size
        coords = np.floor(pc * scale)
        coords, feats, labels, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords), color, labels=instance, ignore_label=-1, return_index=True, return_inverse=True) ### labels is not used later
        # coords = coords.numpy().astype(np.float32)

        # ''' SPG '''
        # ---compute 10 nn graph-------
        xyz = pc[unique_map]
        rgb = color[unique_map].astype(np.float32) / 255
        graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
        # ---compute geometric features-------
        geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
        del target_fea
        # --compute the partition------
        features = np.hstack((geof, rgb)).astype('float32')  # add rgb as a feature for partitioning
        features[:, 3] = 2. * features[:, 3]  # increase importance of verticality (heuristic)

        graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype='float32')
        components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], reg_strength)
        merged = in_component.astype(np.int32)

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
    print(sp2gt.shape, instance.shape, )
    ious = instance2masks(torch.from_numpy(sp2gt), torch.from_numpy(instance), torch.from_numpy(semantic), scene_name)## [K, N]
    return (scene_name, ious)


print('start constructing initial superpoints')
path_list = []
result = []

for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']:
    path_list += sorted(glob(os.path.join(input_path, area, '*.npy')))

for path in path_list:
    result.append(construct_superpoints(path))

print('end constructing initial superpoints')
all_iou = []
for (scene_name, ious) in result:
    all_iou += ious
print('Mean IoU on target category', sum(all_iou)/len(all_iou))