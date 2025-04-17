import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp
import trimesh
from glob import glob

# # file_list = ['0000_02_target_center.ply', '0000_02_initial_cyl.ply', '0000_02_big_cyl.ply', '0000_02_sml_cyl.ply', '0000_02_forward_cyl.ply', '0000_02_backward_cyl.ply', '0000_02_left_cyl.ply', '0000_02_right_cyl.ply']
# file_list = ['/home/zihui/SSD/ndf/objs/chair_3.off']
#
# for file in file_list:
#     # data = read_ply(file)
#     # points = data[:, 0:3]
#     # colors = data[:, 3:6]
#     mesh = trimesh.load(file)
#     points = trimesh.sample.sample_surface_even(mesh, 10000)[0]
#
#     # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
#     # mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.05, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
#     mesh = pc_segm_to_sphere(points, radius=0.01, resolution=3, with_background=False)### 0.05 radius for ScanNet
#     o3d.visualization.draw_geometries([mesh])
#
#     save_file = file[0:-4] + '3.obj'
#     o3d.io.write_triangle_mesh(save_file, mesh)

# file_list = ['0000_02_target_center.ply', '0000_02_initial_cyl.ply', '0000_02_big_cyl.ply', '0000_02_sml_cyl.ply', '0000_02_forward_cyl.ply', '0000_02_backward_cyl.ply', '0000_02_left_cyl.ply', '0000_02_right_cyl.ply']
# scene_names = sorted(glob(os.path.join('/home/zihui/SSD/ndf/vis_ScanNet', '*.ply')))
scene_id_list = ['0030_00', '0131_02', '0081_02', '0088_00', '0088_01', '0088_03', '0131_00', '0169_00', '0196_00']
all_scene_names = sorted(glob(os.path.join('/home/zihui/SSD/ndf/baseline_scannet', 'Part2object', '*.ply')))
# all_scene_names = sorted(glob(os.path.join('/home/zihui/SSD/ndf/ckpt_scannet_final2/latent_diff/step0.3_envr2_CD0.12_ballinitial/vis', '*.ply')))

out_foler = 'ScanNet/Part2object'
scene_names = []
for scene in all_scene_names:
    for scene_id in scene_id_list:
        if scene_id in scene:
            scene_names.append(scene)

for scene in scene_names:
    data = read_ply(scene)
    print(len(data))
    # data = data[np.random.choice(len(data), len(data)//5, replace=False)]
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.02, resolution=3, with_background=False, default_color=colors)### 0.05/0.04 radius for ScanNet
    # o3d.visualization.draw_geometries([mesh])

    os.makedirs(out_foler, exist_ok=True)
    save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '0.02_3.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)
