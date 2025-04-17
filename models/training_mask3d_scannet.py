from __future__ import division
import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
from lib.helper_ply import read_ply, write_ply
import MinkowskiEngine as ME
from mask3d_models.matcher import HungarianMatcher
from mask3d_models.criterion import SetCriterion
from benchmark.evaluate_semantic_instance import evaluate
import random
import colorsys
from typing import List, Tuple
import functools
from sklearn.decomposition import PCA

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

class Trainer(object):
    def __init__(self, model, logger, train_dataset, val_dataset, save_path, cfg=None, use_label=True):
        device = torch.device("cuda")
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr= cfg.lr)
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.0001, epochs=600, steps_per_epoch=240)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.use_label = use_label
        self.use_sp = cfg.use_sp
        self.topk_query = model.num_queries

        # matcher = HungarianMatcher(cost_class=2, cost_mask=5, cost_dice=2, num_points=-1)
        if self.use_label:
            matcher = HungarianMatcher(cost_class=2, cost_mask=5, cost_dice=2, num_points=-1, ignore_label=-1)
        else:
            matcher = HungarianMatcher(cost_class=0, cost_mask=5, cost_dice=2, num_points=-1, ignore_label=-1)

        weight_dict = {"loss_ce": matcher.cost_class, "loss_mask": matcher.cost_mask, "loss_dice": matcher.cost_dice}
        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        if self.use_label:
            ## For ScanNet
            self.criterion = SetCriterion(num_classes=self.model.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
                                losses=["labels", "masks"], num_points=matcher.num_points, oversample_ratio=3.0,
                                importance_sample_ratio=0.75, class_weights=-1).cuda()
            # self.criterion = SetCriterion(num_classes=self.model.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=1.0,
            #                     losses=["labels", "masks"], num_points=matcher.num_points, oversample_ratio=3.0,
            #                     importance_sample_ratio=0.75, class_weights=-1).cuda()
        else:
            self.criterion = SetCriterion(num_classes=self.model.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
                                losses=["masks"], num_points=matcher.num_points, oversample_ratio=3.0,
                                importance_sample_ratio=0.75, class_weights=-1).cuda()

    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        return loss.item()

    def compute_loss(self, batch):
        coords, feature, normals, target, scene_name, semantic, instance, inverse_map, full_pc, sp, _ = batch
        batch_sp = [sp[i].cuda() for i in range(len(sp))]

        in_field = ME.SparseTensor(feature, coords, device=0)
        if self.use_sp:
            output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:])
        else:
            output = self.model(in_field, raw_coordinates=feature[:, -3:])

        target = [{key: tensor.to(self.device) for key, tensor in dictionary.items()} for dictionary in target]
        if self.use_sp:
            mask_type = "segment_mask"
        else:
            mask_type = "masks"
        losses = self.criterion(output, target, mask_type=mask_type)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return sum(losses.values())

    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            loss_display, interval = 0, 40
            loss_display2, loss_display3, loss_display4 = 0, 0, 0
            time_curr = time.time()

            for batch_idx, batch in enumerate(train_data_loader):
                iteration = epoch * len(train_data_loader) + batch_idx + 1

                loss = self.train_step(batch)
                loss_display += loss #/ self.train_dataset.num_sample_points

                if (batch_idx+1) % interval ==0:
                    loss_display /= interval
                    loss_display2 /= interval
                    loss_display3 /= interval
                    loss_display4 /= interval
                    time_used = time.time() - time_curr
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                            epoch, (batch_idx + 1), len(train_data_loader), 100. * (batch_idx + 1) / len(train_data_loader),
                            iteration, loss_display, self.optimizer.param_groups[0]['lr'], time_used, interval))
                    time_curr = time.time()
                    loss_display = 0
                    loss_display2, loss_display3, loss_display4 = 0, 0, 0

            if epoch % 5 ==0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)


    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(epoch))
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
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data_loader):
                coords, feature, normals, target, scene_name, semantic, instance, inverse_map, full_pc, sp, _ = batch
                # if isinstance(instance, list):
                #     instance = instance[0].squeeze()
                batch_sp = [sp[i].cuda() for i in range(len(sp))]

                in_field = ME.SparseTensor(feature, coords, device=0)
                if self.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:])
                    masks = output["pred_masks"][0].detach().cpu()[sp[0]]
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:])
                    masks = output["pred_masks"][0].detach().cpu()

                torch.cuda.empty_cache()
                torch.cuda.synchronize(torch.device("cuda"))

                sem_logits = torch.functional.F.softmax(output["pred_logits"][0][:, :-1], dim=-1).detach().cpu()#remove invalid class

                scores, masks, classes, heatmap = self.get_mask_and_scores(sem_logits, masks, sem_logits.shape[0], sem_logits.shape[1])
                masks, heatmap = masks[inverse_map[0]], heatmap[inverse_map[0]]

                masks = masks.numpy()
                sort_scores = scores.sort(descending=True)
                sort_scores_index = sort_scores.indices.cpu().numpy()
                sort_scores_values = sort_scores.values.cpu().numpy()
                sort_classes = classes[sort_scores_index]
                sorted_masks = masks[:, sort_scores_index]
                ##
                sort_classes = self.val_dataset.remap_model_output(sort_classes.cpu() + self.val_dataset.label_offset)
                ###
                if vis:
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis', exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(sorted_masks.shape[1]):
                        mask = sorted_masks[:, mask_id]==1
                        predcolor[mask] = pred_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds.ply'), [full_pc, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if len(target[0]['masks'])>0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                        for mask_id in range(len(target[0]['masks'])):
                            gtcolor[target[0]['masks'][mask_id][inverse_map[0]]] = gt_instance_color[mask_id]
                        write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'gt.ply'), [full_pc, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    pca = PCA(n_components=3)
                    rgb= pca.fit_transform(output['backbone_features'].F.cpu().numpy())
                    rgb = (rgb - rgb.min())/(rgb.max() - rgb.min())*255
                    rgb = rgb[inverse_map[0]]
                    write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'bkb_feat.ply'), [full_pc, rgb.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                self.preds[scene_name[0]] = {"pred_masks": sorted_masks, "pred_scores": sort_scores_values, "pred_classes": sort_classes}
                # self.preds[scene_name[0]] = {"pred_masks": sorted_masks, "pred_scores": sort_scores_values, "pred_classes": 5*torch.zeros(masks.shape[-1])}
                gt_file = os.path.join(self.cfg.data_dir, 'instance_gt', self.val_dataset.mode, scene_name[0]+'.txt')
                self.gt[scene_name[0]] = gt_file

        evaluate(self.use_label, self.preds, self.gt, self.logger, log)
        # evaluate(False, self.preds, self.gt, self.logger, log)


    # def get_mask_and_scores(self, mask_cls, mask_pred, num_queries=100, num_classes=18):
    #     labels = (torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1))
    #
    #     if self.use_label:## semantic-instance seg
    #         result_pred_mask = (mask_pred > 0).float()  # 最终每个query对应的pred mask
    #         heatmap = mask_pred.float().sigmoid()  # 表示有多像某个query
    #
    #         mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)  ##驻点平均的instance heatmap
    #         score = mask_cls[:, 0] * mask_scores_per_image
    #         classes = 2*torch.ones_like(score).to(self.device)
    #     else: ## pure instance seg
    #         result_pred_mask = (mask_pred > 0).float()  # 最终每个query对应的pred mask
    #         heatmap = mask_pred.float().sigmoid()  # 表示有多像某个query
    #
    #         mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)  ##驻点平均的instance heatmap
    #         score =  mask_scores_per_image
    #         classes = 2*torch.ones_like(score).to(self.device)
    #
    #     return score, result_pred_mask, classes, heatmap

    def get_mask_and_scores(self, mask_cls, mask_pred, num_queries=100, num_classes=18):
        labels = (torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1))

        if self.use_label:## semantic-instance seg
            if self.topk_query != -1:
                scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(self.topk_query, sorted=True)
            else:
                scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(num_queries, sorted=True)

            labels_per_query = labels[topk_indices]
            topk_indices = topk_indices // num_classes
            mask_pred = mask_pred[:, topk_indices]

            result_pred_mask = (mask_pred > 0).float() #最终每个query对应的pred mask
            heatmap = mask_pred.float().sigmoid()# 表示有多像某个query

            mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)##驻点平均的instance heatmap
            score = scores_per_query * mask_scores_per_image
            classes = labels_per_query
        else: ## pure instance seg
            result_pred_mask = (mask_pred > 0).float()  # 最终每个query对应的pred mask
            heatmap = mask_pred.float().sigmoid()  # 表示有多像某个query

            mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)  ##驻点平均的instance heatmap
            score =  mask_scores_per_image
            classes = torch.zeros_like(score).to(self.device)

        return score, result_pred_mask, classes, heatmap


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