import os
import torch
import numpy as np
from depth_render import DepthRender
from pytorch3d.structures import Meshes, join_meshes_as_batch
from lib.helper_ply import write_ply
import trimesh
from glob import glob

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def generate_depth_point_cloud(mesh_path, save_path):
    # Load the mesh
    cls = mesh_path.split('/')[-2]
    name = mesh_path.split('/')[-1][0:-4]
    print('###### Start File', name)
    mesh = trimesh.load(mesh_path)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-loc)
    mesh.apply_scale(1.0 / scale)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]  # Convert to xzy

    # Convert to PyTorch3D mesh
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    pytorch3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    mesh_list = []


    # Generate camera positions on the upper hemisphere
    radius = 2
    render_num = 12
    dist = np.random.uniform(radius, radius, size=render_num)
    # elev = np.linspace(-30, 30, render_num, endpoint=False)### rotation above xy plane
    # azim = np.linspace(80, 360, render_num, endpoint=False) ### rotation in xy plane
    elev = 30*(np.random.rand(render_num)*2-1)### rotation above xy plane
    azim = 180*(np.random.rand(render_num)*2-1) ### rotation in xy plane
    for _ in range(render_num):
        mesh_list.append(pytorch3d_mesh)

    depth_renderer = DepthRender(dist, elev, azim, device)
    depth_list, R_list, t_list, coords_CAM_list, coords_OBJ_list = depth_renderer.render(join_meshes_as_batch(mesh_list))
    os.makedirs(os.path.join(save_path, cls+ '_dep', name), exist_ok=True)
    for idx, point_cloud in enumerate(coords_OBJ_list):
        if len(point_cloud)>5000:
            point_cloud = point_cloud[np.random.choice(len(point_cloud), 5000)]
        np.savez_compressed(os.path.join(save_path, cls+'_dep', name, 'dep_pcl_'+str(idx)+'.npz'), p_w=point_cloud[:, [0, 2, 1]])
        write_ply(os.path.join(save_path, cls+'_dep', name, 'dep_pcl_'+str(idx)+'.ply'), [point_cloud[:, [0, 2, 1]]], ['x', 'y', 'z'])


mesh_dir = '../ONet_data'
save_dir = '../SDF_data'
# processing_cls = ['02691156', '02828884', '03211117', '03636649', '03691459', '04401088', '02933112', '04379243', '04530566', '04090263', '03001627', '04256520']
#################### airplane       bench     display     lamp       loudspeaker telephone    cabinet      table    watercraft    rifle       chair       sofa
processing_cls = ['02691156', '04090263', '04401088', '02933112', '03001627', '04256520']

for cls in processing_cls:
    print('###### Start cls', cls)
    cls_dir = os.path.join(mesh_dir, cls)
    cls_meshes = sorted(glob(os.path.join(cls_dir, '*.off')))
    for mesh_path in cls_meshes:
        generate_depth_point_cloud(mesh_path, save_dir)






