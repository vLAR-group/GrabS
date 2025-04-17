# 2022.7.2 sample the SDF (uni + nss) from DISN water tight mesh, and the aligned with OCC data

# saved_mesh = (ori_mesh.vertices - centroid) / float(m)
# OCC scale: OCC = (ShapenetMesh - loc) / scale

import os
import os.path as osp
import numpy as np
from datetime import datetime
import trimesh
from tqdm import tqdm
from multiprocessing.dummy import Pool

THREADS = 40
ERROR_FN = "SDF_ERROR.txt"

UNI_SAMPLES = 50000
UNI_BOX_SIZE = 0.55
NSS_SAMPLES = 50000
NSS_DELTA = 0.1
PCL_SAMPLES = 100000

GAPS_DIR = "../gaps/bin/x86_64/"


def sample_thread(param):
    # mesh_fn, dst, loc, scale = param
    mesh_fn, dst = param
    print(f"processing {osp.basename(mesh_fn)}")
    if os.path.exists(dst):
        print(f"{dst} exists, skipped")
        return
    os.makedirs(osp.dirname(dst), exist_ok=True)

    # date_time = datetime.now().strftime("%H_%M_%S")
    # tmp_dst = osp.join("/tmp", "zzh_sdf_processing", osp.basename(mesh_fn) + "_" + date_time + "_processing")
    tmp_dst = dst#osp.join("/tmp", "zzh_sdf_processing", osp.basename(mesh_fn) + "_" + date_time + "_processing")
    os.makedirs(tmp_dst, exist_ok=True)

    # normalize the mesh
    # OCC scale: OCC = (ShapenetMesh - loc) / scale
    tmp_mesh_fn = osp.join(tmp_dst, "occ_nrl_mesh.ply")
    ori_mesh = trimesh.load(mesh_fn, process=False)
    ### added by me
    bbox = ori_mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    ###
    ori_mesh.apply_translation(-loc)
    ori_mesh.apply_scale(1.0 / scale)
    ori_mesh.export(tmp_mesh_fn)

    # generate near surface points
    near_surface_fn = os.path.join(tmp_dst, "nearsurface.sdf")
    cmd = f"{GAPS_DIR}msh2pts {tmp_mesh_fn} {near_surface_fn} -near_surface -max_distance {NSS_DELTA} -num_points {NSS_SAMPLES} -v"
    os.system(cmd)
    near_surface_samples = read_pts_file(near_surface_fn)
    near_surface_samples = near_surface_samples[np.random.permutation(near_surface_samples.shape[0])]

    np.savez_compressed(osp.join(tmp_dst, "points_nss.npz"), points=near_surface_samples, scale=scale, loc=loc)
    os.system(f"rm {near_surface_fn}")
    print(f"Near Surface sample Success {near_surface_fn}")

    # generate uniform points
    uniform_fn = os.path.join(tmp_dst, "uniform.sdf")
    cmd = f"{GAPS_DIR}msh2pts {tmp_mesh_fn} {uniform_fn} -uniform_in_bbox -bbox {-UNI_BOX_SIZE} {-UNI_BOX_SIZE} {-UNI_BOX_SIZE} {UNI_BOX_SIZE} {UNI_BOX_SIZE} {UNI_BOX_SIZE} -npoints {UNI_SAMPLES} -v"
    os.system(cmd)
    uniform_samples = read_pts_file(uniform_fn)
    uniform_samples = uniform_samples[np.random.permutation(uniform_samples.shape[0])]
    np.savez_compressed(osp.join(tmp_dst, "points_uni.npz"), points=uniform_samples, scale=scale, loc=loc)

    os.system(f"rm {uniform_fn}")
    print(f"Near Surface sample Success {uniform_fn}")

    # sample surface points
    surface_fn = os.path.join(tmp_dst, "surface.pts")
    cmd = f"{GAPS_DIR}msh2pts {tmp_mesh_fn} {surface_fn} -random_surface_points -npoints {PCL_SAMPLES} -v"
    os.system(cmd)
    print(f"Surface sample Success {surface_fn}")
    surface_samples = read_pts_file(surface_fn)
    surface_samples = surface_samples[np.random.permutation(surface_samples.shape[0])]
    np.savez_compressed(osp.join(tmp_dst, "pointcloud.npz"), points=surface_samples[:, :3], normals=surface_samples[:, 3:], scale=scale, loc=loc)

    os.system(f"rm {surface_fn}")
    os.system(f"rm {tmp_mesh_fn}")
    return


def read_pts_file(path):
    """Reads a .pts or a .sdf point samples file."""
    _, ext = os.path.splitext(path)
    assert ext in [".sdf", ".pts"]
    l = 4 if ext == ".sdf" else 6
    with open(path, "rb") as f:
        points = np.fromfile(f, dtype=np.float32)
    points = np.reshape(points, [-1, l])
    return points


def wrapper(param):
    try:
        sample_thread(param)
    except:
        pass
    mesh_fn, dst = param
    if not osp.exists(dst):
        cmd = f"echo 'Error: {mesh_fn} {dst}' >> {ERROR_FN}"
        os.system(cmd)
        print(cmd)


if __name__ == "__main__":
    # MAIN
    PARAM = {}
    SRC = "ONet_dat/"
    DST = "SDF_data/"

    cates = ['04256520', '03636649', '03001627', '04379243', '02933112']

    print("Preparing params")
    for cate in cates:
        PARAM[cate] = []
        cate_mesh_dir = osp.join(SRC, cate)
        obj_id_list = [f[:-4] for f in os.listdir(cate_mesh_dir) if f.endswith(".off")]
        for obj_id in tqdm(obj_id_list):
            mesh_fn = osp.join(cate_mesh_dir, obj_id + ".off")
            dst = osp.join(DST, cate, obj_id)
            param = (mesh_fn, dst)
            PARAM[cate].append(param)
        print(f"Cate: {cate} has {len(PARAM[cate])} instances")
    for cate, param_list in PARAM.items():
        print(cate)
        with Pool(THREADS) as p:
            p.map(wrapper, param_list)
    print()