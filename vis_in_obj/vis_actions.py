import os
import numpy as np
import argparse, random
from lib.helper_ply import read_ply, write_ply

# Fix random seed
np.random.seed(0)
random.seed(0)

data_file = '0000_02.npy'
save_file = '0000_02'
points = np.load(data_file)
pc, color, normals, sp_idx, semantic, instance = points[:, :3], points[:, 3:6], points[:, 6:9], points[:, 9], points[:,10:11], points[:,11:12]
write_ply(save_file+'.ply',[pc, color.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

### estimate a target center in .ply
center = np.array([2.0, 4.5, 0])[None, ...]
pc = pc - center
write_ply(save_file+'_target_center.ply',[pc, color.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

### compute an initail cylinder
r = 1.5
initial_cyl_mask = np.sqrt((pc[:, 0:2]**2).sum(-1))<=r
write_ply(save_file+'_initial_cyl.ply',[pc[initial_cyl_mask], color[initial_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


### bigger cylinder
r_big = 2.5
big_cyl_mask = np.sqrt((pc[:, 0:2]**2).sum(-1))<=r_big
write_ply(save_file+'_big_cyl.ply',[pc[big_cyl_mask], color[big_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

### bigger cylinder
r_sml = 1.0
sml_cyl_mask = np.sqrt((pc[:, 0:2]**2).sum(-1))<=r_sml
write_ply(save_file+'_sml_cyl.ply',[pc[sml_cyl_mask], color[sml_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

### 4 moving cylinder
step = 1.0
pc_forward = pc.copy()
pc_forward[:, 0] += step
forward_cyl_mask = np.sqrt((pc_forward[:, 0:2]**2).sum(-1))<=r
write_ply(save_file+'_forward_cyl.ply',[pc[forward_cyl_mask], color[forward_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
print('forward')

pc_backward = pc.copy()
pc_backward[:, 0] -= step
backward_cyl_mask = np.sqrt((pc_backward[:, 0:2]**2).sum(-1))<=r
write_ply(save_file+'_backward_cyl.ply',[pc[backward_cyl_mask], color[backward_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
print('backward')

pc_left = pc.copy()
pc_left[:, 1] += step
left_cyl_mask = np.sqrt((pc_left[:, 0:2]**2).sum(-1))<=r
write_ply(save_file+'_left_cyl.ply',[pc[left_cyl_mask], color[left_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
print('left')

pc_right = pc.copy()
pc_right[:, 1] -= step
right_cyl_mask = np.sqrt((pc_right[:, 0:2]**2).sum(-1))<=r
write_ply(save_file+'_right_cyl.ply',[pc[right_cyl_mask], color[right_cyl_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
print('right')