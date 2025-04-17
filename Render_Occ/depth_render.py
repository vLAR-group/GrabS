import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)

class DepthRender:
    def __init__(self, dist, elev, azim, device=torch.device("cpu")) -> None:
        self.device = device
        self.R, self.T = look_at_view_transform(dist, elev, azim, device=device)
        self.render_num = dist.shape[0]

        width = 200#192
        height = 200#192
        fov = 20
        cx = width / 2
        cy = height / 2
        fx = cx / np.tan(fov * np.pi / 180 / 2)
        fy = cy / np.tan(fov * np.pi / 180 / 2)

        raster_settings = RasterizationSettings(
            image_size=width,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            # max_faces_per_bin=20000
        )

        u = np.array(list(np.ndindex((height, width)))).reshape(height, width, 2)[:, :, 1]
        v = np.array(list(np.ndindex((height, width)))).reshape(height, width, 2)[:, :, 0]

        self.cameras = FoVPerspectiveCameras(device=device, R=self.R, T=self.T, fov=fov)

        self.xmap = (u - cx) / fx
        self.ymap = (v - cy) / fy

        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)

    def render(self, meshes):
        fragments = self.rasterizer(meshes)
        # print(fragments.zbuf.size())
        depths = fragments.zbuf[:, :, :, 0]

        R_list = []
        t_list = []
        depth_list = []
        coords_CAM_list = []
        coords_OBJ_list = []

        for i in range(self.render_num):
            depth = depths[i].cpu().squeeze().numpy()
            X_ = self.xmap[depth > -1]  # exclude infinity
            Y_ = self.ymap[depth > -1]  # exclude infinity
            depth_ = depth[depth > -1]  # exclude infinity

            X = X_ * depth_
            Y = Y_ * depth_
            Z = depth_

            R = torch.mm(self.R[i], torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).to(self.device))
            R = R.transpose(1, 0).cpu().numpy()
            t = self.T[i].cpu().numpy()

            coords_CAM = np.stack([X, Y, Z]).T  # shape: num_points * 3
            coords_OBJ = (coords_CAM - t) @ R

            depth_list.append(depth)
            R_list.append(R)
            t_list.append(t)
            coords_CAM_list.append(coords_CAM)
            coords_OBJ_list.append(coords_OBJ)

        return depth_list, R_list, t_list, coords_CAM_list, coords_OBJ_list