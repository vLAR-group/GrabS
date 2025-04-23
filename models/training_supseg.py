from __future__ import division
import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
import torch.nn as nn
from lib.helper_ply import read_ply, write_ply
import MinkowskiEngine as ME
from mask3d_models.matcher_tmp import HungarianMatcher
from benchmark.evaluate_semantic_instance import evaluate
import random
import colorsys
from typing import List, Tuple
import functools
import torch.nn.functional as F
from sklearn.decomposition import PCA
import skimage.measure
import plyfile
import logging
from scipy.optimize import linear_sum_assignment
# from PyTorchEMD.emd import earth_mover_distance
# from torch_scatter import scatter_mean, scatter_sum
# from torch.utils.tensorboard import SummaryWriter

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    # random.shuffle(HSV_tuples)
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

class Trainer(object):
    def __init__(self, model, objnet, logger, train_dataset, val_dataset, save_path, lr=1e-4, cfg=None, dynamic_data=None, threshold=0.1, use_label=True):
        device = torch.device("cuda")
        self.model = model.to(device)
        self.objnet = objnet.to(device).eval()
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr= lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.max_dist = threshold
        self.cfg = cfg
        self.use_label = False#use_label
        self.topk_query = model.num_queries
        self.dynamic_data = dynamic_data
        self.matcher = HungarianMatcher()
        self.count = 0
        # self.writer = SummaryWriter(log_dir=self.cfg.save_path)


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss1, loss2 = self.compute_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        return loss.item(), loss1, loss2


    def compute_loss(self, batch):
        coords, feature, target, _, _, _, inverse_map,  pc, sp, sp_full = batch
        # batch_sp = [sp[i].cuda() for i in range(len(sp))]

        pc = pc.cuda()
        in_field = ME.SparseTensor(feature, coords, device=0)
        output = self.model(in_field, raw_coordinates=feature[:, -3:])
        masks = output["pred_masks"]#[(bs), N, 10]
        masks = torch.cat([masks[b].softmax(-1)[inverse_map[b]][None, :] for b in range(len(output["pred_masks"]))], 0)#[bs, 1w, query_num]
        # masks = torch.cat([masks[b].sigmoid()[inverse_map[b]][None, :] for b in range(len(output["pred_masks"]))], 0)#[bs, 1w, query_num]

        matchings = self.matcher(masks, target)

        target_masks = []
        matched_masks = []
        for bs in range(pc.shape[0]):
            target_masks.append(target[bs]['masks'][matchings[bs][1]].t())
            matched_masks.append(masks[bs][:, matchings[bs][0]])#[1w, query_num]
            # target_masks.append(target[bs]['masks'].t())
        # target_masks = torch.cat(target_masks, 0).t().float().cuda()#.sigmoid()#[3w, bs*10]
        matched_masks = torch.cat(matched_masks, -1).float().cuda()#.sigmoid()#[3w, bs*10]
        target_masks = torch.cat(target_masks, -1).float().cuda()#.sigmoid()#[3w, bs*10]

        loss_bce = F.binary_cross_entropy(matched_masks, target_masks)

        matched_masks, target_masks = matched_masks.t(), target_masks.t()
        matched_masks = matched_masks.flatten(1)
        numerator = 2 * (matched_masks * target_masks).sum(-1)
        denominator = matched_masks.sum(-1) + target_masks.sum(-1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).mean()

        loss = loss_bce + loss_dice
        return loss, loss_bce.item(), loss_dice.item()


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            loss_display, loss_display1, loss_display2, interval = 0, 0, 0, 300
            time_curr = time.time()

            for batch_idx, batch in enumerate(train_data_loader):
                iteration = epoch * len(train_data_loader) + batch_idx + 1

                loss, loss1, loss2 = self.train_step(batch)
                loss_display += loss #/ self.train_dataset.num_sample_points
                loss_display1 += loss1 #/ self.train_dataset.num_sample_points
                loss_display2 += loss2 #/ self.train_dataset.num_sample_points

                if (batch_idx+1) % interval ==0:
                    loss_display /= interval
                    loss_display1 /= interval
                    loss_display2 /= interval
                    time_used = time.time() - time_curr
                    self.logger.info(
                        '{} Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, Loss1: {:.5f}, Loss2: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, (batch_idx + 1), len(train_data_loader),
                            100. * (batch_idx + 1) / len(train_data_loader), iteration, loss_display, loss_display1, loss_display2, self.optimizer.param_groups[0]['lr'],
                            time_used, interval))
                    time_curr = time.time()
                    loss_display, loss_display1, loss_display2 = 0, 0, 0

            if epoch % 5 ==0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)
            # self.writer.close()

    def save_checkpoint(self, epoch):
        path = self.save_path + 'checkpoint_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'epoch':epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.save_path+'/*tar')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.save_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.save_path + 'checkpoint_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch

    def generate_mesh(self, decoder, feat, embedding, path=None, N = 64):
        max_batch = 64 ** 3
        voxel_origin = [-1.0, -1.0, -1.0]
        voxel_size = 2.0 / (N)  ### why minus 1?
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

        samples = torch.zeros(N ** 3, 4)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        # samples*= embedding['s'].cpu()

        head, num_samples = 0, N ** 3
        while head < num_samples:
            sample_subset = samples[head: min(head + max_batch, num_samples), 0:3][None, :].cuda()
            samples[head: min(head + max_batch, num_samples), 3] = decoder(sample_subset/2, feat, embedding).squeeze(1).detach().cpu()
            head += max_batch
        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_values.numpy(), level=0.0, spacing=[voxel_size] * 3)
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        mesh_points /= 2

        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        logging.debug("saving mesh to %s" % ('tmp.ply'))
        # ply_data.write('tmp.ply')
        ply_data.write(path)
        logging.debug("converting to ply format and writing to file took s")
        return mesh_points

    def validation(self, vis=True, log=False):
        self.load_checkpoint()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        pred_instance_color = np.vstack(get_evenly_distributed_colors(self.model.num_queries))
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data_loader):
                coords, feature, target, scene_name, _, instance, inverse_map, full_pc, sp, _ = batch
                batch_sp = [sp[i].cuda() for i in range(len(sp))]
                in_field = ME.SparseTensor(feature, coords, device=0)
                output = self.model(in_field, raw_coordinates=feature[:, -3:])
                # output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:])
                masks = output["pred_masks"][0].detach().cpu()
                ######### trans
                masks = masks[inverse_map[0]].softmax(-1)  # .sigmoid()#[3w, bs*10]
                hard_masks = masks == masks.max(-1).values[:, None]  # masks>0.5#[3w, bs*10]
                # masks = masks[inverse_map[0]].sigmoid()  # .sigmoid()#[3w, bs*10]
                # hard_masks = masks >=0.5  # masks>0.5#[3w, bs*10]
                ##
                #cc
                # mask_score, valid_mask_idx = [], []
                # for mask_id in range(self.model.num_queries):
                #     mask = hard_masks[:, mask_id]
                #     if mask.sum() > 10:
                #         pc = full_pc[0][mask]
                #         center = (full_pc[0] * (masks[:, mask_id].unsqueeze(-1))).sum(0,keepdim=True) / masks[:,mask_id].sum()
                #         pc -= center
                #         mu, _, embedding, _, _ = self.objnet.encode(pc[np.random.choice(len(pc), 1024, replace=True)].unsqueeze(0).cuda())
                #         outsdf = self.objnet.decode(pc.unsqueeze(0).cuda(), mu.unsqueeze(0), embedding).abs().cpu()  # [50, 1]
                #         score = (outsdf*masks[:, mask_id][mask]).sum()/masks[:, mask_id][mask].sum()
                #         # mask_score.append(((outsdf*masks[:, mask_id]).sum()/masks[:, mask_id].sum()).item())## rec error as maskscore
                #         mask_score.append(torch.exp(-score).item())## - exp(rec error) as maskscore
                #         # mask_score.append(torch.exp(-((outsdf.mean()))).item())## - exp(rec error) as maskscore
                #         valid_mask_idx.append(mask_id)
                # valid_masks = hard_masks[:, valid_mask_idx]

                if vis:
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis_valid', exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(hard_masks.shape[1]):
                        mask = hard_masks[:, mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis_valid', scene_name[0] + 'preds.ply'), [full_pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis_valid', scene_name[0] + 'gt.ply'), [full_pc, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    # for mask_id in range(self.model.num_queries):
                    #     mask = hard_masks[:, mask_id]
                    #     if mask.sum()>0:
                    #         pc = torch.from_numpy(full_pc[mask])
                    #         center = (torch.from_numpy(full_pc) * (masks[:, mask_id].unsqueeze(-1))).sum(0, keepdim=True) / masks[ :,mask_id].sum()
                    #         pc -= center
                    #         write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + '_'+ str(mask_id) + '.ply'), [pc.numpy(), gtcolor[mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    #         mu, logvar, embedding, _, _ = self.objnet.encode(pc[np.random.choice(len(pc), 1024, replace=True)].unsqueeze(0).cuda())
                    #
                    #         # print('scene', scene_name[0], 'mask', mask_id, 'scale', embedding['s'], 'p_num', mask.sum())
                    #         path = os.path.join(self.cfg.save_path + '/vis/' + scene_name[0] + '_' + str(mask_id) + 'mesh.ply')
                    #         mesh_points = self.generate_mesh(self.objnet.decode, mu.unsqueeze(0), embedding, path)

                    # pca = PCA(n_components=3)
                    # rgb= pca.fit_transform(output['backbone_features'].F.cpu().numpy())
                    # rgb = (rgb - rgb.min())/(rgb.max() - rgb.min())*255
                    # rgb = rgb[inverse_map[0]]
                    # write_ply(os.path.join(self.cfg.save_path + '/vis_valid', scene_name[0] + 'bkb_feat.ply'), [full_pc, rgb.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                # self.preds[scene_name[0]] = {"pred_masks": valid_masks, "pred_scores": (torch.tensor(mask_score)), "pred_classes": torch.zeros(valid_masks.shape[-1])}
                # self.preds[scene_name[0]] = {"pred_masks": hard_masks, "pred_scores": (torch.tensor(mask_score)), "pred_classes": torch.zeros(hard_masks.shape[-1])}
                self.preds[scene_name[0]] = {"pred_masks": hard_masks, "pred_scores": (masks*hard_masks).sum(0), "pred_classes": torch.zeros(masks.shape[-1])}
                self.gt[scene_name[0]] = instance

        evaluate(self.use_label, self.preds, self.gt, self.logger, log)
        # print('SDF', torch.cat(prediction, -1).mean())


    def vis_dynamic(self):
        checkpoints = glob(self.save_path+'/*tar')
        for i in range(len(checkpoints)):
            self.model.load_state_dict(torch.load(checkpoints[i])['model_state_dict'])
            self.count = i
            self.model.eval()
            dynamic_loader = self.dynamic_data.get_loader(shuffle=False)
            pred_instance_color = np.vstack(get_evenly_distributed_colors(self.model.num_queries))
            with torch.no_grad():
                for batch_idx, batch in enumerate(dynamic_loader):
                    coords, feature, target, scene_name, instance, inverse_map, full_pc, sp, _ = batch
                    batch_sp = [sp[i].cuda() for i in range(len(sp))]
                    in_field = ME.SparseTensor(feature, coords, device=0)
                    output = self.model(in_field, raw_coordinates=feature[:, -3:])
                    masks = output["pred_masks"][0].detach().cpu()
                    ######### trans
                    masks = masks[inverse_map[0]].softmax(-1)  # .sigmoid()#[3w, bs*10]
                    hard_masks = masks == masks.max(-1).values[:, None]  # masks>0.5#[3w, bs*10]
                    ###
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis/' + str(self.count), exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(masks.shape[1]):
                        mask = hard_masks[:, mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis/' + str(self.count), scene_name[0] + 'preds.ply'), [full_pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis/' + str(self.count), scene_name[0] + 'gt.ply'), [full_pc, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    for mask_id in range(self.model.num_queries):
                        mask = hard_masks[:, mask_id]
                        if mask.sum()>0:
                            pc = torch.from_numpy(full_pc[mask])
                            center = (torch.from_numpy(full_pc) * (masks[:, mask_id].unsqueeze(-1))).sum(0, keepdim=True) / masks[ :,mask_id].sum()
                            pc -= center
                            write_ply(os.path.join(self.cfg.save_path + '/vis/' + str(self.count), scene_name[0] + '_' + str(mask_id) + '.ply'), [pc.numpy(), gtcolor[mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                            mu, logvar, embedding, _, _ = self.objnet.encode(pc[np.random.choice(len(pc), 1024, replace=True)].unsqueeze(0).cuda())
                            embedding['s'] = embedding['s'] * 0 + 0.82
                            # embedding['z_so3'] = self.channel_equi_vec_normalize(torch.ones_like(embedding['z_so3']))# * 0 + rota_example.cuda()
                            # mu = torch.clamp(mu, min=-1, max=1)
                            # mu = torch.zeros_like(mu)
                            print('scene', scene_name[0], 'mask', mask_id, 'scale', embedding['s'], 'p_num', mask.sum())
                            path = os.path.join(self.cfg.save_path + '/vis/' + str(self.count), scene_name[0] + '_' + str(mask_id) + 'mesh.ply')
                            mesh_points = self.generate_mesh(self.objnet.decode, mu.unsqueeze(0), embedding, path)


    def vis_dynamic_GTloc(self):
        checkpoints = glob(self.save_path+'/*tar')
        for i in [-1]:#range(len(checkpoints)):
            # self.model.load_state_dict(torch.load(checkpoints[i])['model_state_dict'])
            self.count = i
            self.model.eval()
            dynamic_loader = self.dynamic_data.get_loader(shuffle=False)
            prediction = []
            pred_instance_color = np.vstack(get_evenly_distributed_colors(self.model.num_queries))
            with torch.no_grad():
                for batch_idx, batch in enumerate(dynamic_loader):
                    coords, feature, target, scene_name, instance, inverse_map, full_pc, sp, _ = batch

                    in_field = ME.SparseTensor(feature, coords, device=0)
                    output = self.model(in_field, raw_coordinates=feature[:, -3:])

                    masks = output["pred_masks"][0].detach().cpu()
                    sparsed_xyz = output['sparsed_xyz']

                    ####### softmax trans
                    # mask, scene_pc = masks.cuda(), sparsed_xyz[0]
                    # trans = (mask.softmax(0).unsqueeze(-1) * scene_pc.unsqueeze(1).expand(-1, mask.shape[-1], -1)).sum(0).cpu()

                    sem_logits = torch.functional.F.softmax(output["pred_logits"][0], dim=-1).detach().cpu()
                    prediction.append({"pred_logits": sem_logits, "pred_masks": masks})

                    ###
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis/' + str(self.count), exist_ok=True)
                    gtcolor = np.ones_like(full_pc) * 128
                    gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis/' + str(self.count), scene_name[0] + 'gt.ply'), [full_pc, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    gt_mask = target[0]['masks']  # [10, N]
                    center = gt_mask.float() @ torch.from_numpy(full_pc) / gt_mask.sum(-1, keepdim=True)
                    for i in [0, 1, 2]:
                        centered_obj_pc = torch.from_numpy(full_pc) - center[i, :].unsqueeze(0)
                        centered_obj_pc = centered_obj_pc.numpy()
                        cube_mask = (centered_obj_pc[:, 0] >= -0.5) & (centered_obj_pc[:, 0] <= 0.5) & (centered_obj_pc[:, 2] >= -0.5) \
                                & (centered_obj_pc[:, 2] <= 0.5)&(centered_obj_pc[:, 1]>=-0.4)&(centered_obj_pc[:, 1]<=0.5)  # [bs, N, 20]

                        predcolor = np.ones_like(full_pc) * 128
                        predcolor[cube_mask] = pred_instance_color[i]
                        write_ply(os.path.join(self.cfg.save_path + '/vis/'+ str(self.count), scene_name[0] + '_'+ str(i) + 'cube.ply'), [full_pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                        write_ply(os.path.join(self.cfg.save_path + '/vis/'+ str(self.count), scene_name[0] + '_'+ str(i) + 'cubepc.ply'), [centered_obj_pc[cube_mask]], ['x', 'y', 'z'])

                        cube_pc = torch.from_numpy(centered_obj_pc[cube_mask])
                        sampled_centered_pc = cube_pc[np.random.choice(len(cube_pc), 1024, replace=True)].unsqueeze(0)
                        mu, logvar, embedding, _, _ = self.objnet.encode(sampled_centered_pc.cuda())
                        # mu = torch.clamp(mu, min=-1, max=1)
                        path = os.path.join(self.cfg.save_path + '/vis/'+ str(self.count), scene_name[0] + '_'+ str(i) + 'augae_cubemesh.ply')
                        mesh_points = self.generate_mesh(self.objnet.decode, mu.unsqueeze(0), embedding, path)

                        mu, logvar, embedding, _, _ = self.objnet2.encode(sampled_centered_pc.cuda())
                        # mu = torch.clamp(mu, min=-1, max=1)
                        path = os.path.join(self.cfg.save_path + '/vis/'+ str(self.count), scene_name[0] + '_'+ str(i) + 'ae_cubemesh.ply')
                        mesh_points = self.generate_mesh(self.objnet.decode, mu.unsqueeze(0), embedding, path)


    def vis_sdf(self):
        checkpoints = glob(self.save_path+'/*tar')
        for i in [-1]:#range(len(checkpoints)):
            self.model.load_state_dict(torch.load(checkpoints[i])['model_state_dict'])
            self.count = i
            self.model.eval()
            dynamic_loader = self.dynamic_data.get_loader(shuffle=False)
            os.makedirs(self.cfg.save_path + '/vis2', exist_ok=True)
            with torch.no_grad():
                for batch_idx, batch in enumerate(dynamic_loader):
                    coords, feature, target, scene_name, instance, inverse_map, full_pc, sp, _ = batch

                    ###
                    full_pc = full_pc[0].numpy()
                    gt_mask = target[0]['masks']  # [10, N]
                    for i in [0, 1, 2, 3, 4]:
                        cur_mask = gt_mask[i]
                        centered_obj_pc = torch.from_numpy(full_pc)[cur_mask]
                        # center = gt_mask.float() @ torch.from_numpy(full_pc) / gt_mask.sum(-1, keepdim=True)
                        centered_obj_pc-= centered_obj_pc.mean(0, keepdim=True)

                        s_idx = np.random.choice(len(centered_obj_pc), 1024, replace=True)
                        sampled_centered_pc = centered_obj_pc[s_idx].unsqueeze(0)
                        feat = self.objnet.encode(sampled_centered_pc.cuda())[-1].squeeze(0)
                        feat = feat.reshape(-1, feat.shape[-1]).t()

                        pca = PCA(n_components=3)
                        rgb = pca.fit_transform(feat.cpu().numpy())
                        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
                        # rgb = rgb[inverse_map[0]]
                        write_ply(os.path.join(self.cfg.save_path + '/vis2', scene_name[0] +str(i)+ 'sdf_feat.ply'), [centered_obj_pc.squeeze(0)[s_idx].numpy(), rgb.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])


    def emd_approx(self, x, y):
        bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
        assert npts == mpts, "EMD only works if two point clouds are equal size"
        dim = x.shape[-1]
        x = x.reshape(bs, npts, 1, dim)
        y = y.reshape(bs, 1, mpts, dim)
        dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

        emd_lst = []
        dist_np = dist.cpu().detach().numpy()
        for i in range(bs):
            d_i = dist_np[i]
            r_idx, c_idx = linear_sum_assignment(d_i)
            emd_i = dist[i][r_idx, c_idx].mean()
            emd_lst.append(emd_i.unsqueeze(0))
        emd = torch.cat(emd_lst)
        return emd

    def channel_equi_vec_normalize(self, x):
        # B,C,3,...
        assert x.ndim >= 3, "x shape [B,C,3,...]"
        x_dir = F.normalize(x, dim=2)
        x_norm = x.norm(dim=2, keepdim=True)
        x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
        y = x_dir * x_normalized_norm
        return y


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds