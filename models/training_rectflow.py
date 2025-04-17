import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, cond_net, diffuse_net, VAE, logger, train_dataset, val_dataset, save_path, cfg):
        device = torch.device("cuda")
        self.cond_net = cond_net.to(device)
        self.diffuse_net = diffuse_net.to(device)
        self.VAE = VAE.to(device)

        self.device = device
        self.optimizer = optim.Adam([{'params': self.cond_net.parameters()}, {'params': self.diffuse_net.parameters()}], lr=cfg.lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.val_min = None
        self.cfg = cfg

    def train_step(self,batch):
        self.cond_net.train(), self.diffuse_net.train(), self.VAE.eval()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, batch):
        device = self.device

        inputs = batch.get('on_surface_points').to(device)
        z_clean, mu, logvar, pred_R_inv = self.VAE.encode(inputs, vae_training=True)
        # z_clean = self.VAE.reparameterize(mu, logvar)
        cond_feat = self.cond_net(torch.bmm(inputs, pred_R_inv).detach())
        cond_feat[np.where(np.random.rand(cond_feat.shape[0]) < self.cfg.uncond_prob)] = 0
        ### step 1, prepare x_0, x_1, t
        x_1, x_0 = z_clean, torch.randn_like(z_clean)

        t = torch.rand(z_clean.shape[0]).to(device) ## uniform distribution in [0, 1]

        ### step 2, create x_t
        x_t = t[:,None] * x_1 + (1 - t[:,None]) * x_0 # [B, 1]

        ### step 3, predicted v and gt v, t:[0, 1] --> [0, 1000]
        v_pred = self.diffuse_net(x_t, cond_feat, t*1000)
        loss = F.mse_loss(x_1-x_0, v_pred)

        return loss.mean()


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            loss_display, interval = 0, 37
            time_curr = time.time()

            for batch_idx, batch in enumerate(train_data_loader):
                iteration = epoch * len(train_data_loader) + batch_idx + 1

                loss = self.train_step(batch)
                loss_display += loss

                if (batch_idx+1) % interval ==0:
                    loss_display /= interval
                    time_used = time.time() - time_curr
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                            epoch, (batch_idx + 1), len(train_data_loader), 100. * (batch_idx + 1) / len(train_data_loader),
                            iteration, loss_display, self.optimizer.param_groups[0]['lr'], time_used, interval))
                    time_curr = time.time()
                    loss_display = 0

            if epoch % 10 ==0:
                self.save_checkpoint(epoch)
                val_loss = self.compute_val_loss()
                if self.val_min is None:
                    self.val_min = val_loss
                if val_loss < self.val_min:
                    self.val_min = val_loss
                self.logger.info('Epoch: {}, val_loss={}, val_min_loss={}'.format(epoch, val_loss, self.val_min))

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path,  'checkpoint_{}.tar'.format(epoch))
        if not os.path.exists(path):
            torch.save({'epoch':epoch, 'cond_net_state_dict': self.cond_net.state_dict(),
                        'diffuse_net_state_dict': self.diffuse_net.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.save_path+'/*tar')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.save_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.cond_net.load_state_dict(checkpoint['cond_net_state_dict'])
        self.diffuse_net.load_state_dict(checkpoint['diffuse_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):
        self.cond_net.eval(), self.diffuse_net.eval(), self.VAE.eval()

        sum_val_loss = 0
        num_batches = 100
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    val_batch = self.val_data_iterator.next()
                except:
                    self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                    val_batch = self.val_data_iterator.next()
                ##################################3
                inputs = val_batch.get('on_surface_points').cuda()
                p = val_batch.get('off_surface_points').cuda()
                df_gt = val_batch.get('df').cuda()
                ## condition
                # cond_feat = self.cond_net(inputs)
                ## condition
                pred_R_inv = self.VAE.encode(inputs)[-1]
                cond_feat = self.cond_net(torch.bmm(inputs, pred_R_inv).detach())

                pred_clean_z = self.generation(cond_feat)

                ###
                # print((pred_clean_z-mu).norm(p=2, dim=-1).mean())
                df_pred = self.VAE.decode(p, pred_clean_z, pred_R_inv)
                sdf_error = abs(df_pred - df_gt)
                sdf_near_mask = (sdf_error < 0.1).float().detach()
                sdf_w_error = sdf_error * sdf_near_mask + sdf_error * (1.0 - sdf_near_mask) * 0.5
                loss_sdf_uni, loss_sdf_nss = sdf_w_error[:,int(5000 * 0.9):], sdf_w_error[:, :int(5000 * 0.9)]
                loss_sdf = loss_sdf_uni.mean() * 0.5 + loss_sdf_nss.mean() * 0.5
                sum_val_loss += loss_sdf.item()
        return sum_val_loss / num_batches #/ self.train_dataset.num_sample_points


    @torch.no_grad()
    def generation(self, cond):
        with torch.no_grad():
            dt = 1/self.cfg.sample_steps
            ## initially, x_t is noise, equals to x_0
            # x_t = torch.randn(cond.shape[0], 256).to(self.device)
            x_t = torch.zeros(cond.shape[0], 256).to(self.device)
            for j in range(self.cfg.sample_steps):
                t = j * dt
                t = torch.tensor([t]).to(self.device).repeat(cond.shape[0])

                if cond is not None:
                    v_pred_uncond = self.diffuse_net(x_t, torch.zeros_like(cond), t=t*1000)
                    v_pred_cond = self.diffuse_net(x_t, cond, t=t*1000)
                    v_pred = v_pred_uncond + self.cfg.cond_strength * (v_pred_cond - v_pred_uncond)
                else:
                    v_pred = self.diffuse_net(x_t, cond, t=t*1000)
                ## euler
                x_t = x_t + v_pred * dt
        return x_t