import open3d as o3d
import numpy as np
import colorsys, random, os, sys
from lib.pc_utils import read_ply
from visual_util import pc_segm_to_sphere
import os.path as osp
import trimesh
from glob import glob


for meth in ['obj1', 'obj2', 'obj3']:
    scene_names = sorted(glob(os.path.join('/home/zihui/Desktop/SDF_vis', meth, '*.off')))
    out_foler = os.path.join('objs', meth)

    for scene in scene_names:
        if 'input' in scene:
            data = read_ply(scene)
            print(len(data))
            # if len(data)>200000:
            #     data = data[np.random.choice(len(data), 200000, replace=False)]
            points = data[:, 0:3]
            points = points - points.mean(0, keepdims=True)

            # mesh = pc_segm_to_sphere(points, segm=labels, radius=0.01, resolution=3, with_background=False, default_color=colors)### 0.02/0.03 radius for ScanNet
            mesh = pc_segm_to_sphere(points, radius=0.01, resolution=10, with_background=False)### 0.05 radius for ScanNet        # o3d.visualization.draw_geometries([mesh])

            os.makedirs(out_foler, exist_ok=True)
            save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '0.01.obj')
            o3d.io.write_triangle_mesh(save_file, mesh)
        else:
            mesh = trimesh.load(scene)  # or "input_file.off"
            # Swap Y and Z coordinates
            # if 'gt' in scene:
                # mesh.vertices[:, [0, 1, 2]] = mesh.vertices[:, [0, 2, 1]]

            os.makedirs(out_foler, exist_ok=True)
            save_file = os.path.join(out_foler, scene.split('/')[-1][0:-4] + '.obj')
            mesh.export(save_file)
