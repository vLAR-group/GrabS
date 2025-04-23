from __future__ import division
import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
from lib.helper_ply import read_ply, write_ply
import MinkowskiEngine as ME
from mask3d_models.matcher_tmp import HungarianMatcher
from benchmark.evaluate_semantic_instance_safe import evaluate
import colorsys
from typing import List, Tuple
import functools
import torch.nn.functional as F

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

class Trainer(object):
    def __init__(self, model, logger, train_dataset, val_dataset, save_path, cfg=None, use_label=False):
        self.model = model.cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr= cfg.lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.use_label = use_label
        self.topk_query = model.num_queries
        self.matcher = HungarianMatcher()


    def refresh_info(self):
        ## loss
        self.loss_dict = {'loss': 0, 'seg loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0}
        self.logging_interval = 200
        self.optimize_time = 0
        self.training_iter = 0
        self.elapsed_time = 0

    def train_batch(self, batch, batch_idx, epoch, loader_size):
        ####
        time_cur = time.time()
        coords, feature, normals, target, scene_name, semantic, instance,  full_instance, inverse_map,  unique_map, voxl_pc, pointpc, voxl_sp, pointsp, exist_pseudo = batch
        batch_sp, pc = [voxl_sp[i].cuda() for i in range(len(voxl_sp))], [voxl_pc[i].cuda() for i in range(len(voxl_sp))]
        in_field = ME.SparseTensor(feature, coords, device=0)

        #### pseudo mask, load and save
        valid_bs, gt_mask_list = [], []
        for b in range(len(pc)):
            valid_bs.append(b)
            gt_mask_list.append(target[b]['segment_mask'].t().float().cuda()) ### should change if use nosp
            # gt_mask_list.append(target[b]['masks'].t().float().cuda())

        ### optimize model
        self.model.train()
        #### mask3d_loss
        if self.cfg.use_sp:
            output_train = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:])#, batch_anchor=batch_anchor)
        else:
            output_train = self.model(in_field, raw_coordinates=feature[:, -3:])
        mask_loss, dice_loss, class_loss = self.compute_seg_loss(output_train, output_train["pred_masks"], gt_mask_list, valid_bs, target)
        seg_loss = 2*class_loss + 5*mask_loss + 2*dice_loss
        loss = seg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        ### info
        self.training_iter+=1
        self.loss_dict['loss'] += loss.item()
        self.loss_dict['seg loss'] += seg_loss.item()
        self.loss_dict['mask loss'] += mask_loss.item()
        self.loss_dict['dice loss'] += dice_loss.item()
        self.loss_dict['class loss'] += class_loss.item()
        ### timing
        self.elapsed_time += time.time() - time_cur
        if self.training_iter % self.logging_interval ==0:
            for key, value in self.loss_dict.items():
                self.loss_dict[key] = self.loss_dict[key]/self.logging_interval
            self.logger.info('{} Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.3f}, seg: {:.3f}, mask: {:.3f}, dice: {:.3f}, class: {:.3f}, lr: {:.3e}, Elapsed time: {:.1f}s ({} iters)'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, batch_idx, loader_size, 100. * batch_idx / loader_size, epoch * loader_size + batch_idx,
                    self.loss_dict['loss'], self.loss_dict['seg loss'], self.loss_dict['mask loss'], self.loss_dict['dice loss'], self.loss_dict['class loss'], self.optimizer.param_groups[0]['lr'],
                    self.elapsed_time, self.logging_interval))
            self.refresh_info()

    def compute_seg_loss(self, output, score, pseudo_mask_list, valid_bs, target):
        matchings = self.matcher(output, pseudo_mask_list, valid_bs)
        aux_matching = []
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"]):
                tmp_matching = self.matcher(aux_outputs, pseudo_mask_list, valid_bs)
                aux_matching.append(tmp_matching)

        loss_dice, loss_mask, loss_class = 0, 0, 0
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            matched_slot_num = len(matchings[actual_bs_index][0])
            ### score more sp
            loss_mask += compute_sigmoid_ce_loss(score[actual_bs][:, matchings[actual_bs_index][0]].t(), pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(), matched_slot_num)
            loss_dice += compute_dice_loss(score[actual_bs][:, matchings[actual_bs_index][0]].t(),pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(), matched_slot_num)

        target_classes = torch.full(output["pred_logits"][valid_bs].shape[:-1], self.model.num_classes - 1, dtype=torch.int64, device=output["pred_logits"].device)
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            target_classes[actual_bs_index, matchings[actual_bs_index][0].long()] = 0
        loss_class += F.cross_entropy(output["pred_logits"][valid_bs].transpose(1, 2), target_classes, ignore_index=-1)


        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"]):
                aux_mask = aux_outputs['pred_masks']
                aux_logits = aux_outputs['pred_logits']
                tmp_matching = aux_matching[i]
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    tmp_matched_slot_num = len(tmp_matching[actual_bs_index][0])
                    loss_mask += compute_sigmoid_ce_loss(aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(), pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(), tmp_matched_slot_num)
                    loss_dice += compute_dice_loss(aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(), pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(), tmp_matched_slot_num)

                target_classes = torch.full(aux_logits[valid_bs].shape[:-1], self.model.num_classes - 1, dtype=torch.int64, device=aux_logits.device)
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    target_classes[actual_bs_index, aux_matching[i][actual_bs_index][0].long()] = 0
                loss_class += F.cross_entropy(aux_logits[valid_bs].transpose(1, 2), target_classes, ignore_index=-1)
        return loss_mask, loss_dice, loss_class


    def intersection_over_union(self, mask1, mask2):
        inter_area = (mask1*mask2).sum()
        union_area = mask1.sum() + mask2.sum() - inter_area
        return inter_area / (union_area + 1e-5)

    def get_maxmatch_mask(self, target_mask, cur_mask, env_xyz=None):
        ### if two mask have biggest iou, they are matched; if biggest iou more than one, take the nearest one
        target_mask, cur_mask = target_mask, cur_mask
        ious = []
        for target_mask_id in range(target_mask.shape[-1]):
            iou = self.intersection_over_union(cur_mask, target_mask[:, target_mask_id])
            ious.append(iou.unsqueeze(0))
        if (max(ious)==torch.cat(ious)).sum()>1 and env_xyz is not None:
            cur_mask_center = (cur_mask.unsqueeze(-1)*env_xyz).sum(0, keepdim=True) / cur_mask.sum()
            dist = []
            target_maxiou_mask = target_mask[:, torch.where(max(ious)==torch.cat(ious))[0]]
            for target_mask_id in range(target_maxiou_mask.shape[-1]):
                target_mask_center = (target_maxiou_mask[:, target_mask_id].unsqueeze(-1) * env_xyz).sum(0, keepdim=True) / target_maxiou_mask[:, target_mask_id].sum()
                dist.append((cur_mask_center - target_mask_center).pow(2).sum().sqrt().unsqueeze(0))
            return target_maxiou_mask[:, torch.argmin(torch.cat(dist)).long()], max(ious)
        return target_mask[:, torch.argmax(torch.cat(ious)).long()], max(ious)


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader(shuffle=True)
        start = self.load_checkpoint()
        self.refresh_info()
        for epoch in range(start, epochs):
            for batch_idx, batch in enumerate(train_data_loader):
                self.train_batch(batch, batch_idx+1, epoch, len(train_data_loader))
            if epoch % 10 ==0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)

    def save_checkpoint(self, epoch):
        path = self.save_path + 'checkpoint_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch, 'model_state_dict': self.model.state_dict(),
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
        return epoch

    def validation(self, vis=True, log=False):
        self.load_checkpoint()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        pred_instance_color = np.vstack(get_evenly_distributed_colors(self.model.num_queries))
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, semantic, instance, full_instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, exist_mask = batch
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                in_field = ME.SparseTensor(feature, coords, device=0)
                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:])
                    sp_score = output["pred_masks"]  # [(bs), N, 10]
                    voxel_masks = sp_score[0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:])
                    voxel_score = output["pred_masks"]
                    voxel_masks = voxel_score[0].sigmoid()

                sem_logits = torch.functional.F.softmax(output["pred_logits"][0], dim=-1).detach().cpu()#remove invalid class [100, 14]
                sem_preds = torch.argmax(sem_logits, 1)# +1 ## [100], and need to start from 1, 1-13
                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = (masks>0.5)

                valid_mask_idx, mask_score = [], []

                for mask_id in range(self.model.num_queries):
                    # score = masks[:,mask_id][hard_masks[:, mask_id]].mean()
                    score = masks[:,mask_id][hard_masks[:, mask_id]].mean()#*sem_logits[mask_id, 0]
                    if torch.argmax(output["pred_logits"][0][mask_id])==0:
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())  ## rec error as maskscore
                #
                # valid_masks = hard_masks
                valid_masks = hard_masks[:, valid_mask_idx]
                masks = masks[:, valid_mask_idx]
                hard_masks = hard_masks[:, valid_mask_idx]

            if vis:
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis/'+ scene_name[0].split('/')[0], exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = valid_masks[:, mask_id]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis/', scene_name[0] + 'preds.ply'), [full_pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if len(target[0]['masks'])>0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][:, inverse_map[0]][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + 'vis/', scene_name[0] + 'gt.ply'), [full_pc, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

            self.preds[scene_name[0]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(mask_score)).cpu().numpy(), "pred_classes": 1 * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
            self.gt[scene_name[0]] = full_instance[0]
        evaluate(self.use_label, self.preds, self.gt, self.logger, log, self.save_path)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))


def compute_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks



def compute_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks