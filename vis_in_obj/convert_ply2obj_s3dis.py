import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp
import trimesh
from glob import glob

scene_id_list = ['Area_1/conferenceRoom_1', 'Area_1/office_16', 'Area_3/conferenceRoom_1', 'Area_4/conferenceRoom_1',
                 'Area_4/conferenceRoom_3', 'Area_5/conferenceRoom_1', 'Area_5/conferenceRoom_2', 'Area_6/conferenceRoom_1']
scene_names = []
for scene_id in scene_id_list:
    area, room_name = scene_id.split('/')[0], scene_id.split('/')[1]
    scene_name = os.path.join('/home/zihui/SSD/ndf/baseline_s3dis/', area, 'Part2object', room_name+'.ply')
    # scene_name = os.path.join('/home/zihui/SSD/ndf/vis_ScanNet2S3DIS_Diff/vis', area, room_name+'preds.ply')
    out_foler = 'S3DIS/Part2object'
    scene_names.append(scene_name)

for scene in scene_names:
    data = read_ply(scene)
    print(len(data))
    data = data[np.random.choice(len(data), len(data)//2, replace=False)]
    points = data[:, 0:3]
    points = points - points.mean(0, keepdims=True)
    colors = data[:, 3:6]

    # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    mesh = pc_segm_to_sphere(points, segm=np.arange(len(data)), radius=0.02, resolution=3, with_background=False, default_color=colors)### 0.05 radius for ScanNet
    # o3d.visualization.draw_geometries([mesh])

    os.makedirs(os.path.join(out_foler, scene.split('/')[-3]), exist_ok=True)
    save_file = os.path.join(out_foler, scene.split('/')[-3], scene.split('/')[-1][0:-4] + '0.02_3_div2.obj')
    # os.makedirs(os.path.join(out_foler, scene.split('/')[-2]), exist_ok=True)
    # save_file = os.path.join(out_foler, scene.split('/')[-2], scene.split('/')[-1][0:-4] + '0.02_3_div2.obj')
    o3d.io.write_triangle_mesh(save_file, mesh)
