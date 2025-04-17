import numpy as np
import colorsys, random, os, sys
from lib.helper_ply import read_ply, write_ply
import os.path as osp
from glob import glob
import colorsys
from typing import List, Tuple
import functools
import torch

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

### S3dis
data_path = '/home/zihui/SSD/ndf/s3dis'
for area in ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']:

    scene_names = sorted(glob(os.path.join(data_path, area, '*.npy')))

    out_path = '/home/zihui/SSD/ndf/baseline_s3dis/'+area

    unscene3d_path = '/home/zihui/SSD/ndf/Unscene3D/unscene3d_only3d_s3dis_'+area+'_preds_gt.npz'
    unscene3d_mask_dict = dict(np.load(unscene3d_path, allow_pickle=True))

    part2obj_path = '/home/zihui/SSD/ndf/PartObject/part2object_s3dis_'+area+'_preds_gt.npz'
    part2obj_mask_dict = dict(np.load(part2obj_path, allow_pickle=True))

    for scene_name in scene_names:
        scene_name = scene_name.split('/')[-1][0:-4]
        points = np.load(os.path.join(data_path, area, scene_name+'.npy'))
        pc, color, semantic = points[:, :3], points[:, 3:6], torch.from_numpy(points[:, 10:11])
        non_ceiling_mask = torch.logical_and(semantic != 0, semantic != 12).squeeze().numpy()==1
        # os.makedirs(out_path, exist_ok=True)

        write_ply(os.path.join(out_path, scene_name+'.ply'), [pc[non_ceiling_mask], color[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

        # #### EFEM
        # efem_path = '/home/zihui/SSD/MyEFEM/log/new_all/s3dis_area5_cylinder/results_eval/s3dis_format/'
        # meta_file = os.path.join(efem_path, area, scene_name+'.txt')
        # all_mask = []
        # with open(meta_file, "r") as f:
        #     meta_info = f.readlines()
        # for line in meta_info:
        #     mask_fn, _, score = line.split("\n")[0].split(" ")
        #     mask_fn = os.path.join('/home/zihui/SSD/MyEFEM/log/new_all/s3dis_area5_cylinder/results_eval/s3dis_format/', mask_fn)
        #     with open(mask_fn, "r") as f:
        #         data = f.readlines()
        #     mask = [int(l[0]) for l in data]
        #     mask = np.asarray(mask, dtype=np.uint8)
        #     # print(mask.shape, metamask_file)
        #     all_mask.append(mask[:, None])
        # if len(all_mask)>0:
        #     all_mask = np.concatenate(all_mask, -1)  ## [N, K]
        #     pred_instance_color = np.vstack(get_evenly_distributed_colors(all_mask.shape[1]))
        #
        #     predcolor = np.ones_like(pc) * 128
        #     for mask_id in range(all_mask.shape[1]):
        #         mask = all_mask[:, mask_id]==1
        #         predcolor[mask] = pred_instance_color[mask_id]
        #
        #     os.makedirs(os.path.join(out_path, 'EFEM'), exist_ok=True)
        #     write_ply(os.path.join(out_path, 'EFEM', scene_name+'.ply'), [pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


        # ### Unscene3D
        # try:
        #     mask = unscene3d_mask_dict[scene_name]
        #     mask = dict(mask.item())
        #     pred_cls = mask['pred_classes'].squeeze()
        #     pred_mask = mask['pred_masks']
        #     pred_mask = pred_mask[:, np.where(pred_cls==9)[0].squeeze()]
        #     if len(pred_mask.shape)==1:
        #         pred_mask = pred_mask[:, None]
        #     if (pred_cls==9).sum()>0:
        #         pred_instance_color = np.vstack(get_evenly_distributed_colors((pred_cls==9).sum()))
        #
        #         predcolor = np.ones_like(pc) * 128
        #         for mask_id in range((pred_cls==9).sum()):
        #             mask = pred_mask[:, mask_id]==1
        #             predcolor[mask] = pred_instance_color[mask_id]
        #
        #         os.makedirs(os.path.join(out_path, 'Unscene3D'), exist_ok=True)
        #         write_ply(os.path.join(out_path, 'Unscene3D', scene_name+'.ply'), [pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # except:
        #     print('Unscene3D'+scene_name)


        # ### Part2object
        # try:
        #     mask = part2obj_mask_dict[scene_name]
        #     mask = dict(mask.item())
        #     pred_cls = mask['pred_classes'].squeeze()
        #     pred_mask = mask['pred_masks']
        #     pred_mask = pred_mask[:, np.where(pred_cls==9)[0].squeeze()]
        #     if len(pred_mask.shape)==1:
        #         pred_mask = pred_mask[:, None]
        #     if (pred_cls==9).sum()>0:
        #         pred_instance_color = np.vstack(get_evenly_distributed_colors((pred_cls==9).sum()))
        #
        #         predcolor = np.ones_like(pc) * 128
        #         for mask_id in range((pred_cls==9).sum()):
        #             mask = pred_mask[:, mask_id]==1
        #             predcolor[mask] = pred_instance_color[mask_id]
        #
        #         os.makedirs(os.path.join(out_path, 'Part2object'), exist_ok=True)
        #         write_ply(os.path.join(out_path, 'Part2object', scene_name+'.ply'), [pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
        # except:
        #     print('Part2object'+scene_name)
