import numpy as np
import colorsys, random, os, sys
from lib.helper_ply import read_ply, write_ply
import os.path as osp
from glob import glob
import colorsys
from typing import List, Tuple
import functools

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

### ScanNet
data_path = './data/safe_scene2/processed/test'
scene_names = sorted(glob(os.path.join(data_path, '*.npz')))

out_path = '/home/zihui/SSD/ndf/baseline_safe2/'

for scene_name in scene_names:
    scene_name = scene_name.split('/')[-1][0:-4]
    points = np.load(os.path.join(data_path, scene_name+'.npz'))['data']
    pc = points[:, :3]
    # pc = pc - pc.mean(0, keepdim=True)
    os.makedirs(out_path, exist_ok=True)

    write_ply(os.path.join(out_path, scene_name+'.ply'), [pc], ['x', 'y', 'z'])

    #### validate ours
    efem_path = '/home/zihui/SSD/MyEFEM/log/new_all/safe_test_r1.0/results_eval/s3dis_format'
    meta_file = os.path.join(efem_path, scene_name+'.txt')
    all_mask = []
    with open(meta_file, "r") as f:
        meta_info = f.readlines()
    for line in meta_info:
        mask_fn, _, score = line.split("\n")[0].split(" ")
        mask_fn = os.path.join('/home/zihui/SSD/MyEFEM/log/new_all/safe_test_r1.0/results_eval/s3dis_format', mask_fn)
        with open(mask_fn, "r") as f:
            data = f.readlines()
        mask = [int(l[0]) for l in data]
        mask = np.asarray(mask, dtype=np.uint8)
        # print(mask.shape, metamask_file)
        all_mask.append(mask[:, None])
    if len(all_mask)>0:
        all_mask = np.concatenate(all_mask, -1)  ## [N, K]
        pred_instance_color = np.vstack(get_evenly_distributed_colors(all_mask.shape[1]))

        predcolor = np.ones_like(pc) * 128
        for mask_id in range(all_mask.shape[1]):
            mask = all_mask[:, mask_id]==1
            predcolor[mask] = pred_instance_color[mask_id]

        os.makedirs(os.path.join(out_path, 'EFEM'), exist_ok=True)
        write_ply(os.path.join(out_path, 'EFEM', scene_name+'.ply'), [pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])