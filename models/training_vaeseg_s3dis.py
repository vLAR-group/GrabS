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
from benchmark.evaluate_semantic_instance_s3dis_chair import evaluate
import random
import colorsys
from typing import List, Tuple
import functools
import pickle
import torch.nn.functional as F
import skimage.measure
import plyfile
import logging
from torch.distributions.categorical import Categorical
from torch_scatter import scatter_mean, scatter_max, scatter_min
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'logprob', 'td_target', 'value', 'advantage'))#, 'sample_rate'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.adv = []

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.adv.append(None)
        self.memory[self.position] = Transition(*args)
        self.adv[self.position] = args[7].squeeze().item()
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def adv_mean_std(self):
        adv = np.array(self.adv)
        return adv.mean(), adv.std()

    def __len__(self):
        return len(self.memory)

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

class Trainer(object):
    def __init__(self, model, objnet, PPO_actor, PPO_critic, logger, train_dataset, val_dataset, val_RL_dataset, save_path, cfg=None, use_norm=True, use_label=False):
        self.model = model.cuda()
        self.objnet = objnet.cuda().eval()
        # self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr= cfg.lr)
        self.optimizer = optim.AdamW(self.model.parameters(), lr= cfg.lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_RL_dataset = val_RL_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.use_label = use_label
        self.use_norm = use_norm
        self.topk_query = model.num_queries
        self.matcher = HungarianMatcher()

        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.max_step = 8
        self.max_eval_step = 8

        self.actor, self.critic = PPO_actor, PPO_critic

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=1e-4, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-4, eps=1e-5)

        self.alpha = 0.2  # €[0, 1]  Scaling factor
        self.nu = 10  # Reward of Trigger
        self.threshold = 0.5
        self.clip_actor_eps = 0.2
        self.gae_lambda = 0.5
        self.gae = True
        self.ent_coeff = 0.1
        self.clip_value = False
        self.clip_value_eps = 0.1
        self.normalize_adv = True

        self.anchor_env_r = 2.0#1.5
        self.moving_step = 0.3
        self.obj_r = 0.6
        self.obj_h = 1.8
        self.scene_h = 4
        ### used to filter sdfmasks, its set by EFEM, can we drop or release it?
        self.min_obj_r = 0.15 ## use EFEM param firstly
        self.max_obj_r = 0.6
        self.max_obj_h = 1.8
        self.self_atten_sample_num = 1024
        self.convergence_sample_num = 1024
        self.obj_center_z = 0.6

        self.batch_iter = 2
        self.sp_pseudo = True
        self.traj_dict_capa = 8*self.cfg.env_num
        self.initial_R = 1.5
        self.min_R = 0.15
        self.max_R = 1.5
        self.R_decay = 0.75

        self.sdf_bs = 100
        self.distance_thr = 0.02
        self.norm_thr = 180
        ## about unsup reward
        self.pc2mesh_thr = 0.7
        self.mesh2pc_thr = 0.6
        self.phy_distance_thr = 0.07

        self.cd_thr = 0.16

        self.mask_min_size = 100
        self.bcyl_min_size = 50

    def refresh_info(self):
        ## loss
        self.loss_dict = {'loss': 0, 'ppo loss': 0 , 'actor loss': 0, 'critic loss': 0, 'ent loss': 0,
                          'seg loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0}
        self.training_iter = 0
        self.logging_interval = len(self.train_dataset.get_loader(shuffle=True))*self.batch_iter
        self.step_reward = 0
        self.traj_length = 0
        self.data_time, self.optimize_time = 0, 0
        self.ious, self.ious50, self.ious25 = 0, 0, 0
        self.num_ious, self.num_ious50, self.num_ious25 = 0, 0, 0

    def init_traj_dict(self):
        self.traj_dict = {}

    def assign_env_info(self, traj_id, cur_bs, cur_env, all_actions, history, curpos, curR, initial_bcyl_center, env_feature, env_xyz, env_norm, cur_GT_bcyl_mask, mask_completeness):
        initial_bcy_mask = self.compute_bcyl([], env_xyz, initial_bcyl_center, self.initial_R)[1]  ### replaced
        if initial_bcy_mask.sum()>self.bcyl_min_size:
            self.traj_dict[str(traj_id)] = {}

            self.traj_dict[str(traj_id)]['bcyl_mask'] = initial_bcy_mask
            self.traj_dict[str(traj_id)]['cur_bs'] = cur_bs ### fixed
            self.traj_dict[str(traj_id)]['cur_env'] = cur_env ### fixed
            self.traj_dict[str(traj_id)]['all_actions'] = all_actions ## accumulated
            self.traj_dict[str(traj_id)]['history'] = history ## accumulated
            self.traj_dict[str(traj_id)]['curpos'] = curpos ### replaced
            self.traj_dict[str(traj_id)]['curR'] = curR ### replaced
            self.traj_dict[str(traj_id)]['initial_bcyl_center'] = initial_bcyl_center ### fixed
            self.traj_dict[str(traj_id)]['env_feature'] = env_feature ### fixed
            self.traj_dict[str(traj_id)]['env_xyz'] = env_xyz ### fixed
            self.traj_dict[str(traj_id)]['env_norm'] = env_norm ### fixed
            self.traj_dict[str(traj_id)]['cur_GT_bcyl_mask'] = cur_GT_bcyl_mask ### fixed
            self.traj_dict[str(traj_id)]['done'] = [False]  ### increased
            self.traj_dict[str(traj_id)]['traj'] = [] ### increased
            self.traj_dict[str(traj_id)]['dist2target'] = self.anchor_env_r ### replaced
            self.traj_dict[str(traj_id)]['target_mask_center'] = None ### replaced
            self.traj_dict[str(traj_id)]['W'] = torch.zeros_like(env_xyz)[:, 0]#.cpu() ### replaced
            self.traj_dict[str(traj_id)]['iou'] = 0
            self.traj_dict[str(traj_id)]['mask_completeness'] = mask_completeness

    def train_batch(self, batch, batch_idx, epoch, loader_size):
        ####
        time_cur = time.time()
        coords, feature, normals, target, scene_name, semantic, instance,  inverse_map,  unique_map, voxl_pc, pointpc, voxl_sp, pointsp, exist_pseudo = batch
        batch_sp, pc = [voxl_sp[i].cuda() for i in range(len(voxl_sp))], [voxl_pc[i].cuda() for i in range(len(voxl_sp))]
        in_field = ME.SparseTensor(feature, coords, device=0)
        bs = len(pc)
        pseudo_batch, obj_score = [[] for _ in range(len(pc))], [[] for _ in range(len(pc))]  ## batch size of []

        ### 1. compute point/sp features and mask, these are only used to compute the TD_target, so can be no gradien
        self.model.eval(), self.actor.eval(), self.critic.eval()
        with torch.no_grad():
            if self.cfg.use_sp:
                output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp, env_num=self.cfg.env_num, is_datacollect=True)
            else:
                output = self.model(in_field, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp, env_num=self.cfg.env_num, is_datacollect=True)
            bkb_feature = output["mask_features"].decomposed_features
            batch_anchor = output['sampled_coords'].detach() #[bs, K, 3]

        #### 2. collect trajectory for current batch
        ###### for each batch, we have B scene, each scene has 50 anchor, leading to 50*B trajectory ########
        self.memory, step_R, step_num, traj_num = ReplayMemory(bs*self.cfg.env_num*self.max_step), 0, 0, 0
        # Here we represent state by the point_idx
        with torch.no_grad():
            state_index, state_pred_mask = [], []
            ### to measure the quality of mask
            for b in range(bs):
                state_index.append([]), state_pred_mask.append([])
                pc2anchor = (pc[b][:, None, :] - batch_anchor[b][None, ...])[:, :, 0:2].norm(p=2, dim=-1) #[x, y]
                for anchor_idx in range(self.cfg.env_num):
                    in_env_idx = torch.where(pc2anchor[:, anchor_idx]<=self.anchor_env_r)[0]
                    state_index[-1].append(in_env_idx)
                    state_pred_mask[-1].append(None)

        ### split traj id into sets
        traj_id_set = []
        for traj_id in range(bs*self.cfg.env_num):
            if traj_id % self.traj_dict_capa == 0:
                traj_id_set.append([])
            traj_id_set[-1].append(traj_id)

        ### random sample 10% traj, otherwise the data collection in RL is too time-consuming
        for l in range(len(traj_id_set)):
            traj_id_set[l] = np.random.choice(traj_id_set[l], len(traj_id_set[l])//10, replace=False).tolist()

        for traj_ids in traj_id_set:
            self.init_traj_dict() ### traj dict is only an temporary storage, it mainly used to record the state feature for many traj
            ### 2.1 init some environment info for current traj_id_set to dict
            for traj_id in traj_ids:
                cur_bs, cur_env = traj_id//self.cfg.env_num, traj_id%self.cfg.env_num
                all_actions, history, bcyl_center = [], torch.zeros((self.max_step, 6 + 1)).cuda(), batch_anchor[cur_bs][cur_env].unsqueeze(0)
                ##
                curpos, initial_bcyl_center = bcyl_center.clone(), bcyl_center.clone()
                curpos[:, -1], initial_bcyl_center[:, -1] = curpos[:, -1]*0, initial_bcyl_center[:, -1]*0
                env_feature = bkb_feature[cur_bs][state_index[cur_bs][cur_env]]
                env_xyz = pc[cur_bs][state_index[cur_bs][cur_env]]
                env_norm = normals[cur_bs][state_index[cur_bs][cur_env]]

                GT_mask = target[cur_bs]['masks'].cuda() ##[K', N]
                GT_env_mask = GT_mask[:, state_index[cur_bs][cur_env]]
                env_GTmask_ratio = GT_env_mask.sum(-1)/GT_mask.sum(-1)
                ###
                GT_env_thr = 0.01# if this env has taeget object, record its mask, GT here is only to check the training process, not influence training
                if max(env_GTmask_ratio)>=GT_env_thr:
                    traj_num +=1
                    GT_idx = torch.where(env_GTmask_ratio>=GT_env_thr)[0]
                    cur_GT_env_mask = GT_env_mask[GT_idx].t()## [N, K]
                    if len(cur_GT_env_mask.shape)==1:
                        cur_GT_env_mask = cur_GT_env_mask.unsqueeze(-1)
                    ### convert GT mask to cylindar
                    cur_GT_bcyl_mask = cur_GT_env_mask#self.to_bcyl(cur_GT_env_mask, env_xyz)
                    ### now we know this is an valud traj_id, so assign to dict
                    self.assign_env_info(traj_id, cur_bs, cur_env, all_actions, history, curpos, self.initial_R, initial_bcyl_center, env_feature, env_xyz, env_norm, cur_GT_bcyl_mask, env_GTmask_ratio[GT_idx])
                else:
                    ### no valid GT in cur area
                    traj_num +=1
                    self.assign_env_info(traj_id, cur_bs, cur_env, all_actions, history, curpos, self.initial_R, initial_bcyl_center, env_feature, env_xyz, env_norm, None, None)


            ### 2.2 making steps simultaneously for the current traj_id_set, using dict
            ### 2.2.1 compute state, state feature, action, logprob, value for these set
            for t in range(self.max_step):
                state_list, not_done_traj = [], []
                cur_envfeat_list, cur_hist_list, cur_centered_pos, cur_centered_envxyz, cur_env_feats, cur_bcyl_mask = [], [], [], [], [], []
                cur_inbcyl_xyz, cur_inbcyl_feats, cur_R = [], [], []
                for traj_id in self.traj_dict.keys():
                    if len(self.traj_dict[traj_id]['done']) > t:
                        if not self.traj_dict[traj_id]['done'][t]:
                            ### state: bcylindar center, history, batch_id, anchor_id
                            state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['curR'], self.traj_dict[traj_id]['history'].unsqueeze(0),
                            self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'], self.traj_dict[traj_id]['initial_bcyl_center'])
                            state_list.append(state)
                            not_done_traj.append(traj_id)
                            ###
                            cur_centered_pos.append((self.traj_dict[traj_id]['curpos'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            ###
                            cur_centered_envxyz.append((self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            cur_env_feats.append(self.traj_dict[traj_id]['env_feature'].unsqueeze(0))
                            #### tiny mask3d
                            cur_bcyl_mask.append(self.traj_dict[traj_id]['bcyl_mask'].unsqueeze(0))
                            inbcyl_xyz = (self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center'])[torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            inbcyl_feats = self.traj_dict[traj_id]['env_feature'][torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            cur_R.append(self.traj_dict[traj_id]['curR'])

                            sample_idx = np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=False) if inbcyl_xyz.shape[0] >= self.self_atten_sample_num \
                                        else np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=True)
                            cur_inbcyl_xyz.append(inbcyl_xyz[sample_idx].unsqueeze(0)), cur_inbcyl_feats.append(inbcyl_feats[sample_idx].unsqueeze(0))
                            cur_hist_list.append(self.traj_dict[traj_id]['history'].unsqueeze(0))

                if len(not_done_traj)==0:
                    break
                else:
                    cur_centered_pos = torch.cat(cur_centered_pos)
                    cur_centered_pos[:, :, -1] *= 0

                    cur_inbcyl_xyz, cur_inbcyl_feats, cur_history = torch.cat(cur_inbcyl_xyz), torch.cat(cur_inbcyl_feats), torch.cat(cur_hist_list)
                    actions, logprobs, values, state_feats = self.select_action(cur_inbcyl_xyz, torch.tensor(cur_R), cur_inbcyl_feats, cur_centered_pos, cur_history)

                    for idx, traj_id in enumerate(not_done_traj): ## record bcyl_mask
                        action = actions[idx].unsqueeze(0)
                        self.traj_dict[traj_id]['all_actions'].append(action)
                        bcyl_center, bcyl_mask, curR = self.compute_bcyl(self.traj_dict[traj_id]['all_actions'], self.traj_dict[traj_id]['env_xyz'], self.traj_dict[traj_id]['initial_bcyl_center'])
                        curpos = bcyl_center
                        self.traj_dict[traj_id]['curpos'] = curpos
                        self.traj_dict[traj_id]['bcyl_mask'] = bcyl_mask
                        self.traj_dict[traj_id]['curR'] = curR


                    sdfmask, inmask_pc, inmask_norm, inmask_prob = self.compute_sdfmask(not_done_traj, batch_sp=batch_sp, state_index=state_index)###only for compute reward, when we take the action, what reward can we have?
                    if inmask_pc is not None:
                        valid_mask, pc2mesh, mesh2pc = self.compute_convergence(sdfmask, inmask_pc, inmask_prob)###only for compute reward, when we take the action, what reward can we have?

                    for idx, traj_id in enumerate(not_done_traj):
                        action, logprob, value = actions[idx].unsqueeze(0), logprobs[idx], values[idx]
                        if inmask_pc is not None:
                            reward, pc2mesh_inrange_ratio, mesh2pc_inrange_ratio = self.compute_reward_CD(idx, valid_mask, pc2mesh, mesh2pc)
                        else:
                            reward = -1

                        ### use GT to check
                        if self.traj_dict[traj_id]['cur_GT_bcyl_mask'] is not None:
                            iou = self.get_maxmatch_mask(self.traj_dict[traj_id]['cur_GT_bcyl_mask'], self.traj_dict[traj_id]['mask_completeness'], sdfmask[idx]).item()
                        else:
                            iou = 0#None

                        if reward == self.nu:
                        # if iou >= self.threshold:
                        #     reward = self.nu
                            if self.cfg.verbose:
                                print('IoU with GT:', iou, 'pc2mesh distance:', pc2mesh_inrange_ratio, 'mesh2pc distance:', mesh2pc_inrange_ratio)
                            self.ious += iou
                            self.num_ious += 1
                            if iou>=0.25:
                                self.ious25 += iou
                                self.num_ious25 += 1
                            if iou>=0.5:
                                self.ious50 += iou
                                self.num_ious50 += 1
                            next_state = None
                            done = True
                        else:
                            next_state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['curR'], self.traj_dict[traj_id]['history'].unsqueeze(0), self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'])
                            done = False
                            ##
                            reward = -1

                        if t == self.max_step-1:
                            done = True
                        self.traj_dict[traj_id]['traj'].append((state_list[idx], action, next_state, reward, logprob, value, done))
                        self.traj_dict[traj_id]['done'].append(done)


                        #### pseudo mask, load and save
                        if reward == self.nu:
                            point_pseudo = torch.zeros_like(pc[self.traj_dict[traj_id]['cur_bs']])[:, 0]
                            point_pseudo[state_index[self.traj_dict[traj_id]['cur_bs']][self.traj_dict[traj_id]['cur_env']]] = sdfmask[idx]#bcyl_mask.float()#tmp
                            # ranking_score = iou
                            # ranking_score = (pc2mesh_inrange_ratio * mesh2pc_inrange_ratio).item()
                            ranking_score = -(pc2mesh_inrange_ratio+mesh2pc_inrange_ratio).item()+10
                            # ranking_score = (pc2mesh_inrange_ratio<=0.07).float().mean(-1).item() * (mesh2pc_inrange_ratio<=0.07).float().mean(-1).item()
                            # ranking_score = torch.rand(1)s.item() + 1e-4
                            if self.cfg.use_sp:
                                pseudo_batch[self.traj_dict[traj_id]['cur_bs']].append(point_pseudo.unsqueeze(-1)), obj_score[self.traj_dict[traj_id]['cur_bs']].append(torch.tensor(ranking_score).unsqueeze(0))
                            else:
                                pseudo_batch[self.traj_dict[traj_id]['cur_bs']].append(point_pseudo.unsqueeze(-1)), obj_score[self.traj_dict[traj_id]['cur_bs']].append(torch.tensor(ranking_score).unsqueeze(0))

                        step_R += reward
                        step_num += 1

            for traj_id in list(self.traj_dict.keys()):
                if self.traj_dict[traj_id]['traj'][-1][3] != self.nu:
                    if random.random() <= 0.9 and len(list(self.traj_dict.keys()))>10:
                        del self.traj_dict[traj_id]
            ### 2.2.1 compute state, state feature, action, logprob, value for these set
            ## generalizaed advantage
            print_reward = True
            if print_reward:
                for traj_id in self.traj_dict.keys():
                    trajectory = self.traj_dict[traj_id]['traj']
                    tmp_reward_list = []
                    for t in range(len(trajectory)):
                        tmp_reward_list.append(trajectory[t][3])

            for traj_id in self.traj_dict.keys():
                trajectory = self.traj_dict[traj_id]['traj']
                if self.gae:
                    lastgae, gae = 0, torch.zeros(len(trajectory))
                    for t in reversed(range(len(trajectory))):
                        if t == len(trajectory) - 1:
                            next_done = trajectory[-1][-1]
                            next_value = 0  # not exist next state trajectory[-1][-2]
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - trajectory[t][-1]
                            nextvalues = trajectory[t + 1][-2]
                        td_delta = trajectory[t][3] + self.GAMMA * nextvalues * nextnonterminal - trajectory[t][-2]
                        gae[t] = lastgae = td_delta + self.GAMMA * self.gae_lambda * nextnonterminal * lastgae

                for step, (state, action, next_state, reward, logprob, value, done) in enumerate(trajectory):
                    # bootstrap value if not done
                    if step < len(trajectory) - 1:
                        next_value = trajectory[step + 1][-2]
                    else:
                        next_value = 0
                    if self.gae:
                        advantage = gae[step]
                        td_target = value + gae[step]
                    else:
                        td_target = reward + self.GAMMA * next_value * (1 - done)
                        td_delta = td_target - value
                        advantage = td_delta
                    self.memory.push(state, action, next_state, reward, logprob, td_target, value, advantage)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        del output

        ### pseudo mask, load and save, only this phase will take 50seconds
        pseudo_mask_list, valid_bs, gt_mask_list = [], [], []  ### sometimes masks in pseudo_mask_list are sp mask
        for b in range(len(pc)):
            if len(pseudo_batch[b]) > 0:
                ### add exist pseudo to current
                cur_pseudo = torch.cat([torch.cat(pseudo_batch[b], dim=-1), exist_pseudo[b][0].cuda()], dim=-1)  # [M , K]
                cur_score = torch.cat([torch.cat(obj_score[b]), exist_pseudo[b][1]], dim=-1).cuda()
            else:
                cur_pseudo = exist_pseudo[b][0].cuda()  # [M , K]
                cur_score = exist_pseudo[b][1].cuda()

            if (cur_pseudo.sum(0) > 0).sum() >0:
                valid_pseudo_idx = torch.where(cur_pseudo.sum(0) > 0)[0]
                cur_pseudo, cur_score = cur_pseudo[:, valid_pseudo_idx], cur_score[valid_pseudo_idx]
                nodup_mask = remove_duplications(cur_pseudo, score=cur_score)
                cur_pseudo, cur_score = cur_pseudo[:, nodup_mask], cur_score[nodup_mask]
                if self.cfg.verbose:
                    print('pseudo mask number:', cur_pseudo.shape[1])
                valid_bs.append(b)

                area_name, room_name = scene_name[b].split('/')[0], scene_name[b].split('/')[1]
                os.makedirs(os.path.join(self.cfg.save_path, 'exist_pseudo', area_name), exist_ok=True)
                exist_pseudo_file = os.path.join(self.cfg.save_path, 'exist_pseudo', area_name, room_name + '.pickle')
                with open(exist_pseudo_file, 'wb') as f:
                    pickle.dump([cur_pseudo.cpu()[inverse_map[b]].bool(), cur_score.cpu()], f)

                if self.cfg.use_sp:
                    sp_pseudo = (scatter_mean(cur_pseudo.float(), batch_sp[b], dim=0) >= 0.5).float()
                    pseudo_mask_list.append(sp_pseudo)
                else:
                    pseudo_mask_list.append(cur_pseudo.float())

        ### timing
        self.data_time += time.time() - time_cur
        time_cur = time.time()
        ### 3. optimize model
        self.model.train(), self.actor.train(), self.critic.train()
        for iter in range(self.batch_iter):
            self.optimizer_actor.zero_grad(), self.optimizer_critic.zero_grad()
            self.optimizer.zero_grad()
            #### mask3d_loss
            if self.cfg.use_sp:
                output_train = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp, anchor=batch_anchor)
            else:
                output_train = self.model(in_field, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp)

            ### ppo_loss
            ppo_loss, pg_loss, critic_loss, entropy_loss = self.compute_rl_loss(pc, output_train, state_index)

            if len(valid_bs) > 0:  ### means in some scene, we found object in RL collecting data process
                mask_loss, dice_loss, class_loss = self.compute_seg_loss(output_train, output_train["pred_masks"], pseudo_mask_list, valid_bs)
                seg_loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss

            else:
                mask_loss = dice_loss = class_loss = torch.tensor(0)
                seg_loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss

            loss = ppo_loss + seg_loss
            # loss = seg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5), nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            # nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer_actor.step(), self.optimizer_critic.step()
            self.optimizer.step()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

            ### info
            self.loss_dict['loss'] += loss.item()
            self.loss_dict['ppo loss'] += ppo_loss.item()
            self.loss_dict['actor loss'] += pg_loss.item()
            self.loss_dict['critic loss'] += critic_loss.item()
            self.loss_dict['ent loss'] += entropy_loss.item()
            self.loss_dict['seg loss'] += seg_loss.item()
            self.loss_dict['mask loss'] += mask_loss.item()
            self.loss_dict['dice loss'] += dice_loss.item()
            self.loss_dict['class loss'] += class_loss.item()

            self.training_iter += 1
            self.step_reward += step_R / step_num
            self.traj_length += step_num / traj_num
            ### timing
            self.optimize_time += time.time() - time_cur
            if self.training_iter % self.logging_interval == 0:
                for key, value in self.loss_dict.items():
                    self.loss_dict[key] = self.loss_dict[key] / self.logging_interval
                self.step_reward /= self.logging_interval
                self.traj_length /= self.logging_interval
                self.logger.info(
                    '{} Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.3f}, ppo: {:.3f}, actor: {:.3f}, critic: {:.3f}, ent: {:.3f}, StepR: {:.2f}, seg: {:.3f}, mask: {:.3f}, '
                    'dice: {:.3f}, class: {:.3f}, lr: {:.3e}, Traj: {:.2f}, data time: {:.1f}s, optimize time: {:.1f}s, Elapsed time: {:.1f}s ({} iters)'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, batch_idx, loader_size,
                        100. * batch_idx / loader_size, epoch * loader_size + batch_idx,
                        self.loss_dict['loss'], self.loss_dict['ppo loss'], self.loss_dict['actor loss'],
                        self.loss_dict['critic loss'], self.loss_dict['ent loss'], self.step_reward,
                        self.loss_dict['seg loss'], self.loss_dict['mask loss'], self.loss_dict['dice loss'],
                        self.loss_dict['class loss'], self.optimizer.param_groups[0]['lr'],
                        self.traj_length, self.data_time, self.optimize_time, self.data_time + self.optimize_time,
                        self.logging_interval))
                self.logger.info(
                    '50iou percent: {:.3f}, 25iou percent: {:.3f}, AVG iou: {:.3f}, AVG 50iou: {:.3f}, AVG 25iou: {:.3f})'.format(
                        self.num_ious50 / (self.num_ious + 1e-5), self.num_ious25 / (self.num_ious + 1e-5),
                        self.ious / (self.num_ious + 1e-5), self.ious50 / (self.num_ious50 + 1e-5),
                        self.ious25 / (self.num_ious25 + 1e-5)))
                self.refresh_info()
            time_cur = time.time()
        del batch_anchor


    def compute_seg_loss(self, output, score, pseudo_mask_list, valid_bs):
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
            target_classes[actual_bs_index, matchings[actual_bs_index][0].long()] = 0### 0 means chair/forground
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


    def compute_rl_loss(self, pc, output, state_index):
        bkb_feature = output["mask_features"].decomposed_features
        #######
        state_info = []
        for b in range(len(pc)):
            state_info.append([])
            for anchor_idx in range(self.cfg.env_num):
                state_info[-1].append(None)

        for traj_id in self.traj_dict.keys():
            env_idx = state_index[self.traj_dict[str(traj_id)]['cur_bs']][self.traj_dict[str(traj_id)]['cur_env']]
            tmp_anchor, env_xyz, env_feature = self.traj_dict[traj_id]['initial_bcyl_center'], pc[self.traj_dict[str(traj_id)]['cur_bs']][env_idx], bkb_feature[self.traj_dict[str(traj_id)]['cur_bs']][env_idx]
            state_info[self.traj_dict[str(traj_id)]['cur_bs']][self.traj_dict[str(traj_id)]['cur_env']] = (env_feature, env_xyz, tmp_anchor)

        if len(self.memory) < self.BATCH_SIZE:
            transitions = self.memory.sample(len(self.memory))
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        action_batch = torch.cat(batch.action).view(-1, 2).long().cuda()
        logprob_batch = torch.FloatTensor(batch.logprob).view(-1, 1).float().cuda()
        value_batch = torch.FloatTensor(batch.value).view(-1, 1).float().cuda()
        td_target_batch = torch.FloatTensor(batch.td_target).view(-1, 1).float().cuda()
        advantage_batch = torch.FloatTensor(batch.advantage).view(-1, 1).float().cuda()
        if self.normalize_adv:
            mu, sigma = self.memory.adv_mean_std()
            mu, sigma = torch.tensor(mu).cuda(), torch.tensor(sigma).cuda()
            advantage_batch = (advantage_batch - mu) / (sigma + 1e-8)

        curpos = torch.cat([batch.state[i][0] for i in range(len(batch.state))]).cuda() ##[bs, 3]
        curR = [batch.state[i][1] for i in range(len(batch.state))]
        history = torch.cat([batch.state[i][2] for i in range(len(batch.state))]).cuda() ##[bs, C]
        bs_idx = torch.tensor([batch.state[i][3] for i in range(len(batch.state))]).cuda().long() ##[bs]
        env_idx = torch.tensor([batch.state[i][4] for i in range(len(batch.state))]).cuda().long() ##[bs]
        initial_bcyl_center = torch.cat([batch.state[i][5] for i in range(len(batch.state))]).cuda() ##[bs, 3]

        cur_envfeat_list, cur_centered_pos, cur_centered_envxyz = [], [], []
        cur_inbcyl_xyz, cur_inbcyl_feats = [], []
        for idx in range(len(bs_idx)):
            cur_centered_pos.append((curpos[idx].unsqueeze(0) - initial_bcyl_center[idx].unsqueeze(0)).unsqueeze(0))
            env_feature, env_xyz, anchor_use2checke = state_info[bs_idx[idx]][env_idx[idx]]
            cur_envfeat_list.append(env_feature), cur_centered_envxyz.append(env_xyz - initial_bcyl_center[idx])

            _, bcyl_mask, _ = self.compute_bcyl([], env_xyz, curpos[idx].unsqueeze(0), r=curR[idx])
            inbcyl_xyz, inbcyl_feats = (env_xyz - initial_bcyl_center[idx].unsqueeze(0))[torch.where(bcyl_mask)[0]], env_feature[torch.where(bcyl_mask)[0]]

            sample_idx = np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=False) if inbcyl_xyz.shape[0] >= self.self_atten_sample_num \
                else np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=True)

            inbcyl_xyz, inbcyl_feats = inbcyl_xyz[sample_idx], inbcyl_feats[sample_idx]
            cur_inbcyl_xyz.append(inbcyl_xyz.unsqueeze(0)), cur_inbcyl_feats.append(inbcyl_feats.unsqueeze(0))

        cur_centered_pos = torch.cat(cur_centered_pos)
        cur_centered_pos[:, :, -1] *= 0

        # history_embedding = self.actor.foward_hist(history)
        logits_moving, logits_scale, hidden, state_feats = self.actor(torch.cat(cur_inbcyl_xyz), torch.tensor(curR), torch.cat(cur_inbcyl_feats), cur_centered_pos, history=None)

        curr_moving_probs, curr_scale_probs = F.softmax(logits_moving, dim=-1), F.softmax(logits_scale, dim=-1)
        curr_value = self.critic(hidden)
        logratio = curr_moving_probs.log().gather(1, action_batch[:, 0].unsqueeze(-1)) + curr_scale_probs.log().gather(1, action_batch[:, 1].unsqueeze(-1)) - logprob_batch
        ratio = logratio.exp()

        # Policy loss
        pg_loss1 = advantage_batch * ratio
        pg_loss2 = advantage_batch * torch.clamp(ratio, 1 - self.clip_actor_eps, 1 + self.clip_actor_eps)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

        # Critic loss
        if self.clip_value:
            v_loss_unclipped = (curr_value - td_target_batch) ** 2
            v_clipped = value_batch + torch.clamp(curr_value - value_batch, -self.clip_value_eps, self.clip_value_eps)
            v_loss_clipped = (v_clipped - td_target_batch).pow(2)
            critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            critic_loss = 0.5 * (curr_value - td_target_batch.detach()).pow(2).mean()
        ####
        curr_probs = (curr_moving_probs[:, None, :] * curr_scale_probs[:, :, None]).view(curr_moving_probs.shape[0], -1)
        entropy_loss = - (curr_probs * curr_probs.log()).sum(-1).mean()

        ppo_loss = pg_loss + 1 * critic_loss - self.ent_coeff * entropy_loss
        return ppo_loss, pg_loss, critic_loss, entropy_loss


    def compute_sdfmask(self, traj_id_list, iters=1, max_iters=1, batch_sp=None, state_index=None):
        ### input points colllection
        final_mask = {}
        valid_traj_id = []
        ## some cylindar are near empty
        for traj_id in traj_id_list:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            pre_center = self.traj_dict[traj_id]['curpos']  # .cpu()#inbcyl_pc.mean(0, keepdim=True)

            # pre_bcyl_mask = self.compute_cur_bcyl(env_xyz, pre_center, r=0.35)[1]  ## query point mask
            # pre_bcyl_mask = self.compute_cur_bcyl(env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]  ## query point mask
            # pre_bcyl_mask2 = self.compute_cur_bcyl(env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]  ## query point mask

            # newnormalize
            pre_center[:, -1] = self.obj_center_z
            pre_bcyl_mask = self.compute_cur_ball(env_xyz, pre_center, r=0.6)[1]  ## query point mask
            pre_bcyl_mask2 = self.compute_cur_ball(env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]  ## query point mask

            if pre_bcyl_mask2.sum()>self.mask_min_size and pre_bcyl_mask.sum()>self.bcyl_min_size:
                tmp_pc = env_xyz[torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                mask_size = (tmp_pc - tmp_pc.mean(0, keepdim=True))[:, 0:2].norm(p=2, dim=-1).max()
                if mask_size >=self.min_obj_r and mask_size<=self.max_obj_r:# and mask_height<=self.max_obj_h:
                    valid_traj_id.append(traj_id)
            final_mask[traj_id] = torch.zeros_like(env_xyz)[:, 0]
        if len(valid_traj_id)==0:
            return list(final_mask.values()), None, None, None

        for iter in range(iters+1):
            ### check validness at first, refresh final mask for valid_traj_id every iter, as non-valid never update, so no need to refresh
            for traj_id in valid_traj_id:
                if iter>0:
                    W = self.traj_dict[traj_id]['W']
                    if (W>0).sum()<self.mask_min_size:
                        valid_traj_id.remove(traj_id)
                final_mask[traj_id] = torch.zeros_like(self.traj_dict[traj_id]['env_xyz'])[:, 0]
            if len(valid_traj_id)==0:
                return list(final_mask.values()), None, None, None
            else:
                input_pc, query_pc, query_gt_norm = [], [], []
                tmp_record = []
                valid_traj_id_tmp = valid_traj_id.copy()
                for traj_id in valid_traj_id_tmp:
                    env_xyz = self.traj_dict[traj_id]['env_xyz']#.cpu()
                    env_norm = self.traj_dict[traj_id]['env_norm']
                    if iter==0:
                        center = self.traj_dict[traj_id]['curpos']#.cpu()#inbcyl_pc.mean(0, keepdim=True)
                        # bcyl_mask_tmp = self.compute_cur_bcyl(env_xyz, center, r=0.35)[1] ## input point mask
                        # bcyl_mask_tmp = self.compute_cur_bcyl(env_xyz, center, r=self.traj_dict[traj_id]['curR'])[1] ## input point mask
                        # bcyl_mask = self.compute_cur_bcyl(env_xyz, center, r=self.traj_dict[traj_id]['curR'])[1]## query point mask

                        ## newnormalize
                        center[:, -1] = self.obj_center_z
                        bcyl_mask_tmp = self.compute_cur_ball(env_xyz, center, r=0.6)[1] ## input point mask
                        bcyl_mask = self.compute_cur_ball(env_xyz, center, r=self.traj_dict[traj_id]['curR'])[1]## query point mask

                        self.traj_dict[traj_id]['W'][torch.where(bcyl_mask_tmp)[0]] = 1.0
                        # self.traj_dict[traj_id]['W'][torch.where(torch.logical_and(bcyl_mask_tmp, bcyl_mask))[0]] = 1.0

                        assert (self.traj_dict[traj_id]['W'].cpu().sum() >0), 'tmd' + str(self.traj_dict[traj_id]['W'].cpu().sum().item())
                        idx = torch.multinomial(self.traj_dict[traj_id]['W'].cpu(), num_samples=1024, replacement=True)
                        inbcyl_pc = env_xyz[idx].clone()

                        # center = inbcyl_pc.mean(0, keepdim=True)
                        bbox_min, bbox_max = (inbcyl_pc-center).min(0).values, (inbcyl_pc-center).max(0).values
                        scale = (bbox_max - bbox_min).max() + 1e-6
                        inbcyl_pc = (inbcyl_pc - center)/scale
                        ###
                        input_pc.append((inbcyl_pc)[None, ...])
                        ''' !!!!! Notice that, now let's temporarily use the fixed raduis=0.6 cylindar, and also input points as query'''
                        ''' This also will affect the final return mask '''
                        query_pc.append((env_xyz[torch.where(bcyl_mask)[0]]-center)[None, ...]/scale)
                        query_gt_norm.append(env_norm[torch.where(bcyl_mask)[0]])
                        tmp_record.append(torch.where(bcyl_mask)[0])
                    else:
                        idx = torch.multinomial(self.traj_dict[traj_id]['W'], num_samples=1024, replacement=True)
                        inmask_pc = env_xyz[idx]
                        center = inmask_pc.mean(0, keepdim=True)
                        bbox_min, bbox_max = (inmask_pc-center).min(0).values, (inmask_pc-center).max(0).values
                        scale = (bbox_max - bbox_min).max() + 1e-6
                        inmask_pc = (inmask_pc - center)/scale
                        ###
                        # bcyl_mask = self.compute_cur_bcyl(env_xyz, center, r=self.traj_dict[traj_id]['curR'])[1]
                        # newnormalize
                        bcyl_mask = self.compute_cur_ball(env_xyz, center, r=self.traj_dict[traj_id]['curR'])[1]

                        input_pc.append((inmask_pc)[None, ...])
                        query_pc.append((env_xyz[torch.where(bcyl_mask)[0]]-center)[None, ...]/scale)
                        query_gt_norm.append(env_norm[torch.where(bcyl_mask)[0]])
                        tmp_record.append(torch.where(bcyl_mask)[0])
                input_pc = torch.cat(input_pc).cuda()

            query_sdf, query_normdiff = [], []
            with torch.no_grad():
                head = 0
                embedding_0, embedding_1 = [], []
                while head < len(valid_traj_id_tmp):
                    embedd = self.objnet.encode(input_pc[head:min(head + self.sdf_bs, len(valid_traj_id_tmp))])
                    embedding_0.append(embedd[0]), embedding_1.append(embedd[1])
                    head += self.sdf_bs
                embedding_0, embedding_1 = torch.cat(embedding_0), torch.cat(embedding_1)
            for id, traj_id in enumerate(valid_traj_id_tmp):
                query_points = query_pc[id]#.cuda()
                tmp_embedding = (embedding_0[id].unsqueeze(0), embedding_1[id].unsqueeze(0))
                if self.use_norm:
                    query_points.requires_grad = self.use_norm
                    cur_onsurf_sdf = self.objnet.decode(query_points, *tmp_embedding).abs()  # [bs*5, N]

                    J = cur_onsurf_sdf.sum()
                    J.backward()
                    cur_onsurf_sdf = cur_onsurf_sdf.float()
                    norm_pred = (query_points.grad).detach().float().squeeze(0)
                    with torch.no_grad():
                        norm_pred = F.normalize(norm_pred, dim=-1)
                        scene_norm = F.normalize(query_gt_norm[id].cuda(), dim=-1)
                        norm_inner = (scene_norm * norm_pred).sum(-1).abs()  ## normal cosine error
                        norm_degdiff = torch.acos(torch.clamp(norm_inner, -1.0, 1.0)) / np.pi * 180#[bs, 1w]
                    query_sdf.append(cur_onsurf_sdf.abs().squeeze(0).detach())
                    query_normdiff.append(norm_degdiff.abs().squeeze(0).detach())
                else:
                    with torch.no_grad():
                        cur_onsurf_sdf = self.objnet.decode(query_points, *tmp_embedding).abs()  # [bs*5, N]
                        cur_onsurf_sdf = cur_onsurf_sdf.float()
                        query_sdf.append(cur_onsurf_sdf.abs().squeeze(0).detach())
                sdfmask = query_sdf[-1]<=self.distance_thr
                # exponent = (np.clip(iter, 0, max_iters) / float(max_iters) * (40 - 10) + 10) * query_sdf[-1]
                exponent = (np.clip(iter, 0, max_iters) / float(max_iters) * (20 - 5) + 5) * query_sdf[-1]
                exponent[query_sdf[-1]>(np.clip(iter, 0, max_iters) / float(max_iters) * (0.15 - 0.2) + 0.2)] = 100###or thr 0.2
                # exponent = (np.clip(iter, 0, max_iters) / float(max_iters) * (200 - 5) + 5) * query_sdf[-1]
                # exponent[query_sdf[-1]>(np.clip(iter, 0, max_iters) / float(max_iters) * (0.1 - 0.15) + 0.15)] = 100###or thr 0.2
                if self.use_norm:
                    sdfmask = torch.logical_and(sdfmask, query_normdiff[-1]<=self.norm_thr) ## the mask for query points, still need to be convert to env_xyz
                    exponent_norm = (np.clip(iter, 0, max_iters) / float(max_iters) * (0.003 - 0.002) + 0.002) * query_normdiff[-1]
                    exponent_norm[query_normdiff[-1]>(np.clip(iter, 0, max_iters) / float(max_iters) * (70 - 120) + 120)] = 100###or thr 120
                    exponent += exponent_norm
                prob = torch.exp(-exponent)#.cpu()
                prob[query_sdf[-1]>(np.clip(iter, 0, max_iters) / float(max_iters) * (0.15 - 0.2) + 0.2)] = 0

                if self.use_norm:
                    prob[query_normdiff[-1]>(np.clip(iter, 0, max_iters) / float(max_iters) * (70 - 120) + 120)] = 0
                prob[torch.isinf(prob)] = 0.0
                prob[torch.isnan(prob)] = 0.0
                self.traj_dict[traj_id]['W'] = torch.zeros_like(self.traj_dict[traj_id]['W'])
                self.traj_dict[traj_id]['W'][tmp_record[id]] = prob
                self.traj_dict[traj_id]['W'][torch.where(self.traj_dict[traj_id]['W']>=1)[0]] = 1
                self.traj_dict[traj_id]['W'][torch.where(self.traj_dict[traj_id]['W']<=0)[0]] = 0
                if (self.traj_dict[traj_id]['W']>0.1).sum() <self.mask_min_size:#== 0:
                    valid_traj_id.remove(traj_id)
                elif self.traj_dict[traj_id]['env_xyz'][torch.where(self.traj_dict[traj_id]['W'] != 0)].max(0).values[-1] > self.max_obj_h:
                    valid_traj_id.remove(traj_id)
                ''' !!!!! Notice that, now let's temporarily use the fixed raduis=0.6 cylindar, and also input points as query'''
                ''' This also will affect the final return mask '''
                final_mask[traj_id] = torch.zeros_like(final_mask[traj_id])
                final_mask[traj_id][tmp_record[id]] = sdfmask.float()

                if batch_sp is not None: ### if mysp, it will have -1
                    tmp_batch_sp = batch_sp[self.traj_dict[traj_id]['cur_bs']]
                    tmp_state_index = state_index[self.traj_dict[traj_id]['cur_bs']][self.traj_dict[traj_id]['cur_env']]
                    env_batch_sp = tmp_batch_sp[tmp_state_index]
                    ### maybe -1 in env_sp
                    valid_env_batch_sp_idx = torch.where(env_batch_sp!=-1)[0].long()
                    sp_pseudo = (scatter_mean(final_mask[traj_id].float()[valid_env_batch_sp_idx], env_batch_sp[valid_env_batch_sp_idx], dim=0) >= 0.5).float()
                    final_mask[traj_id][valid_env_batch_sp_idx] = sp_pseudo[env_batch_sp[valid_env_batch_sp_idx]]
                    ###

        final_inmask_prob, final_inmask_pc, final_inmask_norm = [], [], []
        for traj_id in final_mask.keys():
            final_inmask_prob.append(self.traj_dict[traj_id]['W'][torch.where(final_mask[traj_id])[0]])
            final_inmask_pc.append(self.traj_dict[traj_id]['env_xyz'][torch.where(final_mask[traj_id])[0]])
            final_inmask_norm.append(self.traj_dict[traj_id]['env_norm'][torch.where(final_mask[traj_id])[0]])
        return list(final_mask.values()), final_inmask_pc, final_inmask_norm, final_inmask_prob


    def compute_convergence(self, mask_list, pc_list, prob_list):
        ### some mask are all zeros
        valid_mask = [(mask.sum()>=self.mask_min_size).item() for mask in mask_list]
        encoder_input_pc, query_pc = [], []
        for item_idx, (pc, prob, validness) in enumerate(zip(pc_list, prob_list, valid_mask.copy())):
            if validness:
                idx = torch.multinomial(prob, num_samples=1024, replacement=True)
                sampled_pc = pc[idx]
                center = sampled_pc.mean(0, keepdim=True)
                mask_size = (pc - center)[:, 0:2].norm(p=2, dim=-1).max()
                mask_height = pc[:, -1].max()#(pc - center)[:, -1].max() - (pc - center)[:, -1].min()
                if mask_size >=self.min_obj_r and mask_size<=self.max_obj_r and mask_height<=self.max_obj_h:
                ##
                    bbox_min, bbox_max = (sampled_pc - center).min(0).values, (sampled_pc - center).max(0).values
                    scale = (bbox_max - bbox_min).max() + 1e-6
                    encoder_input_pc.append((sampled_pc - center).unsqueeze(0)/scale)
                    sample_idx = np.random.choice(pc.shape[0], self.convergence_sample_num, replace=False) if pc.shape[0] >= self.convergence_sample_num \
                        else np.random.choice(pc.shape[0], self.convergence_sample_num, replace=True)
                    query_pc.append((pc - center)[sample_idx].unsqueeze(0)/scale)
                else:
                    valid_mask[item_idx] = False

        pc2mesh_list, mesh2pc_list, sec_validness = [], [], []
        if np.array(valid_mask).sum()>0:
            encoder_input_pc = torch.cat(encoder_input_pc)
            query_pc = torch.cat(query_pc)
            ###
            head = 0
            with torch.no_grad():
                while head < encoder_input_pc.shape[0]:
                    embedd = self.objnet.encode(encoder_input_pc[head:min(head + self.sdf_bs, encoder_input_pc.shape[0])])

                    canonical_query_pc = query_pc[head:min(head + self.sdf_bs, encoder_input_pc.shape[0])]# / embedd['s'][:, None, None]
                    ### mesh2pc, 1.put pc into canoical space, 2.extract mesh points on canoical space, 3. distance computing
                    mesh_pts, validness = self.extract_shape_pts(self.objnet.decode, embedd, sample_pts_num=query_pc.shape[1], N=32)

                    batch_pc2mesh = 1000 * torch.ones_like(canonical_query_pc)[:, :, 1]
                    batch_mesh2pc = 1000 * torch.ones_like(canonical_query_pc)[:, :, 1]
                    if len(mesh_pts)>0:
                        valid_canonical_query_pc = canonical_query_pc[np.where(validness)[0]]

                        pc2mesh = (mesh_pts[:, :, None, :] - valid_canonical_query_pc[:, None, :, :]).norm(p=2,dim=-1).min(1).values
                        mesh2pc = (mesh_pts[:, :, None, :] - valid_canonical_query_pc[:, None, :, :]).norm(p=2, dim=-1).min(2).values

                        batch_pc2mesh[np.where(validness)[0]] = pc2mesh
                        batch_mesh2pc[np.where(validness)[0]] = mesh2pc

                    pc2mesh_list.append(batch_pc2mesh)  ###shape: [bs,1024]
                    mesh2pc_list.append(batch_mesh2pc)
                    head += self.sdf_bs

            ## use numpy to update list
            valid_mask_array = np.array(valid_mask)
            valid_mask = valid_mask_array.tolist()
            if len(pc2mesh_list) >0:
                return valid_mask, torch.cat(pc2mesh_list), torch.cat(mesh2pc_list)
            else:
                return valid_mask, [], []
        else:
            return valid_mask, [], []


    def select_action(self, xyz, R, feats, curpos, history):
        with torch.no_grad():
            # history_embedding = self.actor.foward_hist(history)
            logits_moving, logits_scale, hidden, state_feats = self.actor(xyz, R, feats, curpos,  history=None)
            prob_moving, prob_scale = F.softmax(logits_moving, dim=-1).detach(), F.softmax(logits_scale, dim=-1).detach()
            value = self.critic(hidden).detach()
            moving_action_dist, scale_action_dist = Categorical(probs=prob_moving.cpu()), Categorical(probs=prob_scale.cpu())
            moving_action, scale_action = moving_action_dist.sample(), scale_action_dist.sample()
            return torch.cat((moving_action.unsqueeze(-1), scale_action.unsqueeze(-1)), dim=-1), moving_action_dist.log_prob(moving_action) + scale_action_dist.log_prob(scale_action), value, state_feats

    def select_best_action(self, xyz, R, feats, curpos, history):
        with torch.no_grad():
            # history_embedding = self.actor.foward_hist(history)
            logits_moving, logits_scale, hidden, state_feats = self.actor(xyz, R, feats, curpos,  history=None)
            prob_moving, prob_scale = F.softmax(logits_moving, dim=-1).detach(), F.softmax(logits_scale, dim=-1).detach()
            moving_action, scale_action = torch.max(prob_moving, 1).indices, torch.max(prob_scale, 1).indices
            return torch.cat((moving_action.unsqueeze(-1), scale_action.unsqueeze(-1)), dim=-1), state_feats

    def compute_cur_bcyl(self, env_xyz, initial_bcyl_center, r=None):
        if r is None:
            r = self.initial_R
        bcyl_center = initial_bcyl_center.clone()
        bcyl_mask = torch.logical_and((env_xyz - bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
        return bcyl_center, bcyl_mask, r

    def compute_cur_ball(self, env_xyz, initial_bcyl_center, r=None):
        if r is None:
            r = self.initial_R
        bcyl_center = initial_bcyl_center.clone()
        bcyl_mask = torch.logical_and((env_xyz - bcyl_center).norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
        return bcyl_center, bcyl_mask, r

    def compute_bcyl(self, all_actions, env_xyz, initial_bcyl_center, r=None):
        if r is None:
            r = self.initial_R
        ### action 0 is stop, 1,2,3,4 are x-foward/backward, y-forward/backward
        # a = initial_bcyl_center.clone()
        bcyl_center = initial_bcyl_center.clone()
        for action in all_actions:
            moving_action, scale_action = action[:, 0], action[:, 1]
            #### moving
            if moving_action == 1:
                tmp_bcyl_center = bcyl_center.clone()
                tmp_bcyl_center[:, 0] = bcyl_center[:, 0] + self.moving_step
                if (tmp_bcyl_center - initial_bcyl_center)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    ### some floating value problem happen, check if close
                    if torch.isclose(torch.tensor(self.anchor_env_r).to(tmp_bcyl_center.device)**2, (tmp_bcyl_center-initial_bcyl_center)[:, 1]**2):
                        tmp_bcyl_center[:, 0] = initial_bcyl_center[:, 0]
                    else:
                        tmp_bcyl_center[:, 0] = initial_bcyl_center[:, 0] + torch.sqrt(self.anchor_env_r**2 - (tmp_bcyl_center-initial_bcyl_center)[:, 1]**2)
                tmp_bcyl_mask = torch.logical_and((env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
                if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                    bcyl_center = tmp_bcyl_center

            elif moving_action==2:
                tmp_bcyl_center = bcyl_center.clone()
                tmp_bcyl_center[:, 0] = bcyl_center[:, 0] - self.moving_step
                if (tmp_bcyl_center - initial_bcyl_center)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    if torch.isclose(torch.tensor(self.anchor_env_r).to(tmp_bcyl_center.device)**2, (tmp_bcyl_center-initial_bcyl_center)[:, 1]**2):
                        tmp_bcyl_center[:, 0] = initial_bcyl_center[:, 0]
                    else:
                        tmp_bcyl_center[:, 0] = initial_bcyl_center[:, 0] -torch.sqrt(self.anchor_env_r**2 - (tmp_bcyl_center-initial_bcyl_center)[:, 1]**2)
                tmp_bcyl_mask = torch.logical_and((env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
                if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                    bcyl_center = tmp_bcyl_center

            elif moving_action==3:
                tmp_bcyl_center = bcyl_center.clone()
                tmp_bcyl_center[:, 1] = bcyl_center[:, 1] + self.moving_step
                if (tmp_bcyl_center - initial_bcyl_center)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    if torch.isclose(torch.tensor(self.anchor_env_r).to(tmp_bcyl_center.device)**2, (tmp_bcyl_center-initial_bcyl_center)[:, 0]**2):
                        tmp_bcyl_center[:, 1] = initial_bcyl_center[:, 1]
                    else:
                        tmp_bcyl_center[:, 1] = initial_bcyl_center[:, 1] + torch.sqrt(self.anchor_env_r**2 - (tmp_bcyl_center-initial_bcyl_center)[:, 0]**2)
                tmp_bcyl_mask = torch.logical_and((env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
                if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                    bcyl_center = tmp_bcyl_center

            elif moving_action==4:
                tmp_bcyl_center = bcyl_center.clone()
                tmp_bcyl_center[:, 1] = tmp_bcyl_center[:, 1] -self.moving_step
                if (tmp_bcyl_center - initial_bcyl_center)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    if torch.isclose(torch.tensor(self.anchor_env_r).to(tmp_bcyl_center.device)**2, (tmp_bcyl_center-initial_bcyl_center)[:, 0]**2):
                        tmp_bcyl_center[:, 1] = initial_bcyl_center[:, 1]
                    else:
                        tmp_bcyl_center[:, 1] = initial_bcyl_center[:, 1] -torch.sqrt(self.anchor_env_r**2 - (tmp_bcyl_center-initial_bcyl_center)[:, 0]**2)
                tmp_bcyl_mask = torch.logical_and((env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
                if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                    bcyl_center = tmp_bcyl_center

            #### scaling
            if scale_action == 1:
                if r*self.R_decay>=self.min_R:
                    tmp_r = r*self.R_decay
                    tmp_bcyl_mask = torch.logical_and((env_xyz - bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=tmp_r, env_xyz[:, -1]<=self.obj_h)
                    if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                        r = tmp_r

            elif scale_action == 2:
                if r/self.R_decay<=self.max_R:
                    tmp_r = r/self.R_decay
                    tmp_bcyl_mask = torch.logical_and((env_xyz - bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=tmp_r, env_xyz[:, -1]<=self.obj_h)
                    if tmp_bcyl_mask.sum()>self.bcyl_min_size:
                        r = tmp_r

        bcyl_mask = torch.logical_and((env_xyz - bcyl_center)[:, 0:2].norm(p=2, dim=-1)<=r, env_xyz[:, -1]<=self.obj_h)
        return bcyl_center, bcyl_mask, r

    def compute_reward(self, idx, valid_mask, pc2meshs, mesh2pcs):
        if valid_mask[idx]:
            true_list = [i for i, x in enumerate(valid_mask) if x]
            idx_in_true_list = true_list.index(idx)
            pc2mesh = pc2meshs[idx_in_true_list]
            mesh2pc = mesh2pcs[idx_in_true_list]
            if (pc2mesh<self.phy_distance_thr).float().mean(-1) >=self.pc2mesh_thr and (mesh2pc<self.phy_distance_thr).float().mean(-1) >=self.mesh2pc_thr:
                return self.nu, (pc2mesh<self.phy_distance_thr).float().mean(-1), (mesh2pc<self.phy_distance_thr).float().mean(-1)
            else:
                return -1, (pc2mesh<self.phy_distance_thr).float().mean(-1), (mesh2pc<self.phy_distance_thr).float().mean(-1)
        else:
            return -1, None, None


    def compute_reward_CD(self, idx, valid_mask, pc2meshs, mesh2pcs):
        if valid_mask[idx]:
            true_list = [i for i, x in enumerate(valid_mask) if x]
            idx_in_true_list = true_list.index(idx)
            pc2mesh = pc2meshs[idx_in_true_list]
            mesh2pc = mesh2pcs[idx_in_true_list]
            if (pc2mesh.mean()+mesh2pc.mean())<=self.cd_thr:
            # if pc2mesh.mean()<=0.05 and mesh2pc.mean()<=0.09:
                return self.nu, pc2mesh.mean(), mesh2pc.mean()
            else:
                return -1, pc2mesh.mean(), mesh2pc.mean()
        else:
            return -1, None, None

    def intersection_over_union(self, mask1, mask2):
        inter_area = (mask1*mask2).sum()
        union_area = mask1.sum() + mask2.sum() - inter_area
        return inter_area / (union_area + 1e-5)


    def get_maxmatch_mask(self, target_mask, target_mask_completeness, cur_mask):
        ious = []
        for target_mask_id in range(target_mask.shape[-1]):
            inter_area = (cur_mask * target_mask[:, target_mask_id]).sum()
            union_area = cur_mask.sum() + target_mask[:, target_mask_id].sum()/(target_mask_completeness[target_mask_id]+1e-5) - inter_area
            iou = inter_area / (union_area + 1e-5)
            completeness = target_mask_completeness[target_mask_id]
            ious.append(iou*completeness)
        return max(ious)


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader(shuffle=True)
        start = self.load_checkpoint()
        self.refresh_info()
        for epoch in range(start, epochs):
            for batch_idx, batch in enumerate(train_data_loader):
                self.train_batch(batch, batch_idx+1, epoch, len(train_data_loader))
            if epoch % 25 ==0:
                self.save_checkpoint(epoch)
                self.validation_RL(vis=False, log=True)
                self.validation(vis=False, log=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(epoch))
        if not os.path.exists(path):
            torch.save({'epoch':epoch,
                'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(), 'opt_actor_state_dict': self.optimizer_actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(), 'opt_critic_state_dict': self.optimizer_critic.state_dict()
                }, path)

    def load_checkpoint(self, ckpt_path=None):
        if ckpt_path is not None:
            path = ckpt_path
        else:
            checkpoints = glob(self.save_path+'/*tar')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.save_path))
                return 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(checkpoints[-1]))
            # path = '/home/zihui/SSD/GOPS/ckpt_scannet_final_latest/scaleVAE91_wpos/step0.3_envr2_CD0.12_ballinitial/checkpoint_440.tar'
            # path = '/home/zihui/SSD/GOPS/ckpt_scannet_final_latest/latent_diff/step0.3_envr2_CD0.12_ballinitial/checkpoint_450.tar'
            # path = '/home/zihui/SSD/GOPS/ckpt_scannet_final_latest/scaleVAE91_wpos/step0.3_envr2_CD0.16_ballinitial/checkpoint_260.tar'

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['opt_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def validation_pseudo(self, vis=True, log=False):
        self.refresh_info()
        self.preds, self.gt = {}, {}
        train_data_loader = self.train_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(train_data_loader):
            coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pcs, voxl_sp, pointsp, exist_pseudo = batch
            ### pseudo mask, load and save, only this phase will take 50seconds
            for b in range(len(exist_pseudo)):
                ### add exist pseudo to current
                cur_pseudo = exist_pseudo[b][0]
                cur_score = exist_pseudo[b][1]
                cur_pseudo = torch.tensor(cur_pseudo)

                valid_masks = cur_pseudo[inverse_map[b]].detach()
                mask_score = cur_score

                if vis:
                    with torch.no_grad():
                        full_pc = full_pcs[b].numpy()
                        non_ceiling_mask = (torch.logical_and(semantic[b] != 0, semantic[b] != 12))[inverse_map[b]]
                        os.makedirs(self.cfg.save_path + '/vis_pseudo/' + scene_name[b], exist_ok=True)
                        predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                        pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1] + 1))
                        for mask_id in range(valid_masks.shape[1]):
                            predcolor2 = np.ones_like(full_pc) * 128
                            mask = valid_masks[:, mask_id]
                            predcolor2[mask] = pred_instance_color[mask_id]
                            predcolor[mask] = pred_instance_color[mask_id]
                            # write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds_'+str(mask_id)+'.ply'), [full_pc, predcolor2.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                        write_ply(os.path.join(self.cfg.save_path + '/vis_pseudo', scene_name[b] + 'preds.ply'), [full_pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                        if target[b]['masks'].sum() > 0:
                            gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[b]['masks'])))
                            for mask_id in range(len(target[b]['masks'])):
                                gtcolor[target[b]['masks'][:, inverse_map[b]][mask_id]] = gt_instance_color[mask_id]
                        write_ply(os.path.join(self.cfg.save_path + 'vis_pseudo', scene_name[b] + 'gt.ply'), [full_pc[non_ceiling_mask], gtcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                        for mask_id in range(valid_masks.shape[1]):  # self.model.num_queries-1):
                            cur_mask = cur_pseudo[:, mask_id]
                            if cur_mask.sum()>0:
                                # print(cur_mask.sum(), scene_name[b], mask_id)
                                pc = voxl_pc[b]
                                pc = pc[torch.where(cur_mask)[0]]
                                pc -= pc.mean(0)
                                bbox_min, bbox_max = pc.min(0).values, pc.max(0).values
                                scale = (bbox_max - bbox_min).max() + 1e-6
                                pc = pc / scale
                                embedding = self.objnet.encode(pc[np.random.choice(len(pc), 1024, replace=True)].unsqueeze(0).cuda())
                                path = os.path.join(self.cfg.save_path + 'vis_pseudo',  scene_name[b], str(mask_id) + 'maskmesh.ply')
                                write_ply(os.path.join(self.cfg.save_path + '/vis_pseudo', scene_name[b], str(mask_id) + 'maskpc.ply'), [pc.numpy()], ['x', 'y', 'z'])
                                # mesh_points = self.generate_mesh(self.objnet.decode, mu.unsqueeze(0), embedding, path, scale_gt=(bound_max - bound_min).max().item())
                                mesh_pts = self.generate_mesh(self.objnet.decode, embedding, path=path).cpu()
                                mesh2pc = (mesh_pts[:, None, :] - pc[None, :, :]).norm(p=2,dim=-1).min(1).values
                                pc2mesh = (mesh_pts[:, None, :] - pc[None, :, :]).norm(p=2, dim=-1).min(0).values
                                print(scene_name[b], mask_id, (pc2mesh<self.phy_distance_thr).float().mean(-1), (mesh2pc<self.phy_distance_thr).float().mean(-1))

                self.preds[scene_name[b]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(mask_score)).cpu().numpy(), "pred_classes": (8+1) * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
                gt_file = os.path.join(self.cfg.data_dir, 'instance_gt', scene_name[b] + '.txt')
                self.gt[scene_name[b]] = gt_file
        evaluate(self.use_label, self.preds, self.gt, self.logger, log, self.save_path)

    def validation(self, vis=True, log=False, ckpt_path=None):
        self.load_checkpoint(ckpt_path)
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, exist_pseudo = batch
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                in_field = ME.SparseTensor(feature, coords, device=0)
                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp)
                    sp_score = output["pred_masks"]  # [(bs), N, 10]
                    voxel_masks = sp_score[0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp)
                    voxel_score = output["pred_masks"]
                    voxel_masks = voxel_score[0].sigmoid()

                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = (masks>0.5)

                valid_mask_idx, mask_score = [], []

                for mask_id in range(self.model.num_queries):
                    score = masks[:,mask_id][hard_masks[:, mask_id]].mean()
                    if torch.argmax(output["pred_logits"][0][mask_id])==0:# and (hard_masks[:, mask_id]==1).sum()>50:
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())  ## rec error as maskscore
                #
                valid_masks = hard_masks[:, valid_mask_idx]
                if len(valid_mask_idx)>0:
                    pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))
            if vis:
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    non_ceiling_mask = (torch.logical_and(semantic[0]!=0, semantic[0]!=12))[inverse_map[0]]
                    area_name, room_name = scene_name[0].split('/')[0], scene_name[0].split('/')[1]
                    os.makedirs(self.cfg.save_path + '/vis/'+ area_name, exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = valid_masks[:, mask_id]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                        # write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds_'+str(mask_id)+'.ply'), [full_pc, predcolor2.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    write_ply(os.path.join(self.cfg.save_path + '/vis/', area_name, room_name + 'preds.ply'), [full_pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if target[0]['masks'].sum()>0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                        for mask_id in range(len(target[0]['masks'])):
                            gtcolor[target[0]['masks'][:, inverse_map[0]][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + '/vis/', area_name, room_name + 'gt.ply'), [full_pc[non_ceiling_mask], gtcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

            self.preds[scene_name[0]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(mask_score)).cpu().numpy(), "pred_classes": (8+1) * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
            gt_file = os.path.join(self.cfg.data_dir, 'instance_gt', scene_name[0] + '.txt')
            self.gt[scene_name[0]] = gt_file
        evaluate(self.use_label, self.preds, self.gt, self.logger, log, self.save_path)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

    def validation_RL(self, vis=True, log=False):
        self.load_checkpoint()
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.model.eval(), self.actor.eval(), self.critic.eval()
        val_data_loader = self.val_RL_dataset.get_loader(shuffle=False)
        pred_instance_color = np.vstack(get_evenly_distributed_colors(self.cfg.env_num))
        for batch_idx, batch in enumerate(val_data_loader):
            coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, exist_pseudo = batch
            batch_sp, pc = [voxl_sp[i].cuda() for i in range(len(voxl_sp))], [voxl_pc[i].cuda() for i in range(len(voxl_sp))]
            in_field = ME.SparseTensor(feature, coords, device=0)
            with torch.no_grad():
                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp)
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:], train_on_segments=self.cfg.use_sp)
                bkb_feature = output["mask_features"].decomposed_features
                batch_anchor = output['sampled_coords'].detach()  # [bs, K, 3]

                valid_masks, mask_score, traj_masks, traj_score = [], [], {}, {}

            step_R, step_num, traj_num = 0, 0, 0
            # Here we represent state by the point_idx
            state_index = []
            pc2anchor = (pc[0][:, None, :] - batch_anchor[0][None, ...])[:, :, 0:2].norm(p=2, dim=-1)  # [x, y]
            for anchor_idx in range(self.cfg.env_num):
                in_env_idx = torch.where(pc2anchor[:, anchor_idx] <= self.anchor_env_r)[0]
                state_index.append(in_env_idx)

            self.init_traj_dict()
            for traj_id in range(self.cfg.env_num):
                traj_masks[str(traj_id)], traj_score[str(traj_id)] = [], []
                cur_env = traj_id
                # all_actions, history, bcyl_center = [], torch.zeros((self.max_eval_step, 1*128+1)).cuda(), batch_anchor[0][cur_env].unsqueeze(0)
                all_actions, history, bcyl_center = [], torch.zeros((self.max_eval_step, 6+1)).cuda(), batch_anchor[0][cur_env].unsqueeze(0)
                ##
                curpos, initial_bcyl_center = bcyl_center.clone(), bcyl_center.clone()
                curpos[:, -1], initial_bcyl_center[:, -1] = curpos[:, -1] * 0, initial_bcyl_center[:, -1] * 0
                env_feature = bkb_feature[0][state_index[cur_env]]
                env_xyz = pc[0][state_index[cur_env]]
                env_norm = normals[0][state_index[cur_env]]

                GT_mask = target[0]['masks'].cuda()  ##[K', N]
                GT_env_mask = GT_mask[:, state_index[cur_env]]
                env_GTmask_ratio = GT_env_mask.sum(-1) / GT_mask.sum(-1)
                ###
                if len(target[0]['masks'])>0:
                    GT_mask = target[0]['masks'].cuda()  ##[K', N]
                    GT_env_mask = GT_mask[:, state_index[cur_env]]
                    env_ratio = GT_env_mask.sum(-1) / GT_mask.sum(-1)
                    ###
                    GT_env_thr = 1
                    if max(env_ratio) >= GT_env_thr:
                        traj_num += 1
                        GT_idx = torch.where(env_GTmask_ratio >= GT_env_thr)[0]
                        cur_GT_env_mask = GT_env_mask[GT_idx].t()  ## [N, K]
                        if len(cur_GT_env_mask.shape) == 1:
                            cur_GT_env_mask = cur_GT_env_mask.unsqueeze(-1)
                        ### convert GT mask to cylindar
                        cur_GT_bcyl_mask = cur_GT_env_mask#self.to_bcyl(cur_GT_env_mask, env_xyz)
                        ### now we know this is an valud traj_id, so assign to dict
                        self.assign_env_info(traj_id, 0, cur_env, all_actions, history, curpos, self.initial_R, initial_bcyl_center, env_feature, env_xyz, env_norm, cur_GT_bcyl_mask, env_GTmask_ratio[GT_idx])
                    else:
                        ### no valid GT in cur area
                        ## traj_num +=1
                        self.assign_env_info(traj_id, 0, cur_env, all_actions, history, curpos, self.initial_R, initial_bcyl_center, env_feature, env_xyz, env_norm, None, None)

            ### 2.2 making steps simultaneously for the current traj_id_set, using dict
            ### 2.2.1 compute state, state feature, action, logprob, value for these set
            for t in range(self.max_eval_step):
                state_list, not_done_traj = [], []
                cur_envfeat_list, cur_hist_list, cur_centered_pos, cur_centered_envxyz, cur_env_feats, cur_bcyl_mask = [], [], [], [], [], []
                cur_inbcyl_xyz, cur_inbcyl_feats, cur_R = [], [], []
                for traj_id in self.traj_dict.keys():
                    if len(self.traj_dict[traj_id]['done']) > t:
                        if not self.traj_dict[traj_id]['done'][t]:
                            ### state: bcylindar center, history, batch_id, anchor_id
                            state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['curR'], self.traj_dict[traj_id]['history'].unsqueeze(0),
                            self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'], self.traj_dict[traj_id]['initial_bcyl_center'])
                            state_list.append(state)
                            not_done_traj.append(traj_id)
                            ###
                            cur_centered_pos.append((self.traj_dict[traj_id]['curpos'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            ###
                            cur_centered_envxyz.append((self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            cur_env_feats.append(self.traj_dict[traj_id]['env_feature'].unsqueeze(0))
                            #### tiny mask3d
                            cur_bcyl_mask.append(self.traj_dict[traj_id]['bcyl_mask'].unsqueeze(0))
                            inbcyl_xyz = (self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center'])[torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            inbcyl_feats = self.traj_dict[traj_id]['env_feature'][torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            cur_R.append(self.traj_dict[traj_id]['curR'])

                            sample_idx = np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=False) if inbcyl_xyz.shape[0] >= self.self_atten_sample_num \
                                        else np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=True)
                            cur_inbcyl_xyz.append(inbcyl_xyz[sample_idx].unsqueeze(0)), cur_inbcyl_feats.append(inbcyl_feats[sample_idx].unsqueeze(0))
                            cur_hist_list.append(self.traj_dict[traj_id]['history'].unsqueeze(0))

                if len(not_done_traj) == 0:
                    break
                else:
                    cur_centered_pos = torch.cat(cur_centered_pos)
                    cur_centered_pos[:, :, -1] *= 0

                    cur_inbcyl_xyz, cur_inbcyl_feats, cur_history = torch.cat(cur_inbcyl_xyz), torch.cat(cur_inbcyl_feats), torch.cat(cur_hist_list)
                    actions, state_feats = self.select_best_action(cur_inbcyl_xyz, torch.tensor(cur_R), cur_inbcyl_feats, cur_centered_pos, cur_history)

                    for idx, traj_id in enumerate(not_done_traj): ## record bcyl_mask
                        action = actions[idx].unsqueeze(0)
                        self.traj_dict[traj_id]['all_actions'].append(action)
                        bcyl_center, bcyl_mask, curR = self.compute_bcyl(self.traj_dict[traj_id]['all_actions'], self.traj_dict[traj_id]['env_xyz'], self.traj_dict[traj_id]['initial_bcyl_center'])
                        curpos = bcyl_center
                        self.traj_dict[traj_id]['history'][t][action] = 1
                        self.traj_dict[traj_id]['history'][t][-1] = 1
                        self.traj_dict[traj_id]['curpos'] = curpos
                        self.traj_dict[traj_id]['bcyl_mask'] = bcyl_mask
                        self.traj_dict[traj_id]['curR'] = curR

                    sdfmask, inmask_pc, inmask_norm, inmask_prob = self.compute_sdfmask(not_done_traj, batch_sp=batch_sp, state_index=[state_index])###only for compute reward, when we take the action, what reward can we have?
                    if inmask_pc is not None:
                        valid_mask, pc2mesh, mesh2pc = self.compute_convergence(sdfmask, inmask_pc, inmask_prob)
                    for idx, traj_id in enumerate(not_done_traj):
                        if inmask_pc is not None:
                            reward, pc2mesh_inrange_ratio, mesh2pc_inrange_ratio = self.compute_reward_CD(idx, valid_mask, pc2mesh, mesh2pc)
                        else:
                            reward = -1
                        if self.traj_dict[traj_id]['cur_GT_bcyl_mask'] is not None:
                            iou = self.get_maxmatch_mask(self.traj_dict[traj_id]['cur_GT_bcyl_mask'], self.traj_dict[traj_id]['mask_completeness'], sdfmask[idx])
                        else:
                            iou = 0#None

                        if reward == self.nu:
                        # if iou >= self.threshold:
                            print(traj_id, iou, pc2mesh_inrange_ratio, mesh2pc_inrange_ratio)
                            next_state = None
                            done = True
                        else:
                            next_state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['history'], self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'])
                            done = False
                        if t == self.max_step-1:
                            done = True

                        self.traj_dict[traj_id]['done'].append(done)

                        outmask = torch.zeros_like(pc[0])[:, 0].float()
                        outmask[state_index[self.traj_dict[traj_id]['cur_env']]] = sdfmask[idx]

                        if reward == self.nu:
                        # if iou >= self.threshold:
                        #     ranking_score = -kl+10
                            ranking_score = iou#pc2mesh_inrange_ratio * torch.clamp(mesh2pc_inrange_ratio, min=0, max=1)
                            traj_masks[traj_id].append(outmask[inverse_map[0]].unsqueeze(-1).detach().cpu()), traj_score[traj_id].append(ranking_score)

            for traj_id in traj_masks.keys():
                if len(traj_masks[traj_id])>0:
                    valid_masks.append(traj_masks[traj_id][-1]), mask_score.append(traj_score[traj_id][-1])
            if len(valid_masks)>0:
                valid_masks = torch.cat(valid_masks, dim=-1).detach().cpu()
                nodup = remove_duplications(valid_masks, torch.tensor(mask_score))
                valid_masks = valid_masks[:, nodup]
                mask_score = torch.tensor(mask_score)[nodup.long()]
            else:
                valid_masks, mask_score = torch.zeros_like(full_pc[0])[:, 0].unsqueeze(-1), torch.tensor([0])
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

            path = '/vis_RL_1iters_unsup2/'
            if vis:
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    non_ceiling_mask = (torch.logical_and(semantic[0]!=0, semantic[0]!=12))[inverse_map[0]]
                    os.makedirs(self.cfg.save_path + path + '/'+scene_name[0], exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = torch.where(valid_masks[:, mask_id])[0]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                        # write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds_'+str(mask_id)+'.ply'), [full_pc, predcolor2.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    write_ply(os.path.join(self.cfg.save_path + path +scene_name[0] + '/preds.ply'), [full_pc[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if len(target[0]['masks']) > 0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][:, inverse_map[0]][mask_id]] = gt_instance_color[mask_id]
                    write_ply(os.path.join(self.cfg.save_path + path + scene_name[0] + '/gt.ply'),[full_pc[non_ceiling_mask], gtcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    for traj_id in self.traj_dict.keys():
                        env_color = np.zeros_like(pc[0].cpu().numpy()) * 128
                        env_color[state_index[int(traj_id)].cpu().long()] = pred_instance_color[int(traj_id)]
                        env_color = env_color[inverse_map[0]]
                        write_ply(os.path.join(self.cfg.save_path + path + scene_name[0] + '/env' + traj_id + '.ply'),
                                  [full_pc[non_ceiling_mask], env_color[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    for traj_id_idx, traj_id in enumerate(traj_masks.keys()):
                        if len(traj_masks[traj_id]) > 0:
                            for t in range(len(traj_masks[traj_id])):
                                traj_color = np.zeros_like(full_pc) * 128
                                traj_color[torch.where(traj_masks[traj_id][t].squeeze())[0]] = pred_instance_color[int(traj_id)]
                                write_ply(os.path.join(self.cfg.save_path + path + scene_name[0] + '/traj' + traj_id + '_' + str(t) + '.ply'), [full_pc, traj_color.astype(np.uint8)],
                                          ['x', 'y', 'z', 'red', 'green', 'blue'])
                                print('bcyl iou:', scene_name[0], traj_id, t, traj_score[traj_id][t], 'action:', self.traj_dict[traj_id]['all_actions'][t])

            self.preds[scene_name[0]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(mask_score)).cpu().numpy(), "pred_classes": (8+1) * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
            gt_file = os.path.join(self.cfg.data_dir, 'instance_gt', scene_name[0] + '.txt')
            self.gt[scene_name[0]] = gt_file
        evaluate(self.use_label, self.preds, self.gt, self.logger, log, self.save_path)

    def extract_shape_pts(self, decoder, embedding, N=32, sample_pts_num=10000):
        bs = embedding[0].shape[0]
        validness = []
        max_batch = self.sdf_bs
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
        samples = samples.unsqueeze(0).repeat(bs, 1, 1)

        head, num_samples = 0, N ** 3

        with torch.no_grad():
            while head < bs:
                tmp_embedding_0 = embedding[0][head: min(head + max_batch, num_samples)]
                tmp_embedding_1 = embedding[1][head: min(head + max_batch, num_samples)]
                sample_subset = samples[head: min(head + max_batch, num_samples), :, 0:3].cuda()
                # print('###################', sample_subset.shape)
                samples[head: min(head + max_batch, num_samples), :, 3] = decoder(sample_subset/2, tmp_embedding_0, tmp_embedding_1).squeeze(1).detach().cpu().float()
                head += max_batch
        sdf_values = samples[:, :, 3]

        onsurf_points_list = []
        for b in range(bs):
            sdf_value = sdf_values[b].reshape(N, N, N)
            try:
                verts, faces, normals, values = skimage.measure.marching_cubes(sdf_value.numpy(), level=0.0, spacing=[voxel_size] * 3)
                mesh_points = np.zeros_like(verts)
                mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
                mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
                mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
                mesh_points /= 2

                onsurf_points = torch.from_numpy(sample_points_from_mesh(mesh_points, faces, sample_pts_num)).float().cuda()
                if onsurf_points.shape[0]!=sample_pts_num:
                    onsurf_points = onsurf_points[np.random.choice(onsurf_points.shape[0], sample_pts_num, replace=True)]
                onsurf_points_list.append(onsurf_points.unsqueeze(0))
                validness.append(True)
            except:
                validness.append(False)
                print('cannot recovery')
        try:
            if len(onsurf_points_list)>0:
                return torch.cat(onsurf_points_list), validness
            else:
                return [], validness
        except:
            print(1)


    def generate_mesh(self, decoder, embedding, N=32, path=None):
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
        samples = samples.unsqueeze(0)

        with torch.no_grad():
            sample_subset = samples[:, :, 0:3].cuda()
            samples[:, :, 3] = decoder(sample_subset / 2, *embedding).squeeze(1).detach().cpu().float()
        sdf_values = samples[:, :, 3]

        sdf_value = sdf_values.reshape(N, N, N)
        verts, faces, normals, values = skimage.measure.marching_cubes(sdf_value.numpy(), level=0.0,spacing=[voxel_size] * 3)
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
        ply_data.write(path)
        return torch.from_numpy(sample_points_from_mesh(mesh_points, faces, self.convergence_sample_num)).float().cuda()


def sample_points_from_mesh(vertices, faces, num_samples):
    def compute_area(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_product = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(cross_product, axis=1) * 0.5
        return area

    def sample_faces(faces, areas, num_samples):
        cumulative_areas = np.cumsum(areas)
        cumulative_areas /= cumulative_areas[-1]
        samples = np.random.rand(num_samples)
        face_indices = np.searchsorted(cumulative_areas, samples)
        return face_indices

    def sample_points(vertices, faces, face_indices):
        v0 = vertices[faces[face_indices, 0]]
        v1 = vertices[faces[face_indices, 1]]
        v2 = vertices[faces[face_indices, 2]]

        u = np.random.rand(len(face_indices), 1)
        v = np.random.rand(len(face_indices), 1)
        is_above = (u + v) > 1
        u[is_above] = 1 - u[is_above]
        v[is_above] = 1 - v[is_above]

        sampled_points = (1 - u - v) * v0 + u * v1 + v * v2
        return sampled_points

    areas = compute_area(vertices, faces)
    face_indices = sample_faces(faces, areas, num_samples)
    sampled_points = sample_points(vertices, faces, face_indices)

    return sampled_points


def remove_duplications(masks, score, iou_th=0.5, inclusion_flag=True, inclusion_th=0.8):
    ## masks: [N, K]
    ## scores: [K]
    N = masks.shape[-1]
    active_mask = torch.ones(N).to(masks.device)
    for i in range(N):
        if active_mask[i] == 0:
            continue  # if removed already
        # find duplication
        '''here B can be replaced by ppt.trj[w]???'''
        B = masks * active_mask[None, :]
        '''D is the raw sdf, get its valid proposal, and compute iou for each proposal with all others'''
        inter = torch.logical_and(B, B[:, i : i + 1])
        union = torch.logical_or(B, B[:, i : i + 1])
        iou = inter.sum(0).float() / (union.sum(0).float() + 1e-6)
        duplication_mask = iou >= iou_th
        '''for each proposal, identify all duplications, only retain the highest score one, may delete cur proposal itself'''
        # merge
        if duplication_mask.sum() > 1:
            _score = score.clone()
            _score[~duplication_mask] = 0.0
            merge_to_i = _score.argmax()
            active_mask[duplication_mask] = 0.0
            active_mask[merge_to_i] = 1.0

    if inclusion_flag:
        for i in range(N):
            if active_mask[i] == 0:
                continue  # if removed already
            # find duplication
            B = masks * active_mask[None, :]
            inter = torch.logical_and(B, B[:, i : i + 1])
            ratio = inter.sum(0).float() / (B[:, i : i + 1].sum().float() + 1e-6)
            ratio[i] = 0.0
            inclusion_ratio = ratio.max()
            if inclusion_ratio > inclusion_th:  # reject
                active_mask[i] = 0.0

    return torch.where(active_mask==1)[0]


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
    return loss.mean(1).sum() / num_masks