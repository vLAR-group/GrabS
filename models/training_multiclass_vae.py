import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time

class Trainer(object):
    def __init__(self, model, logger, train_dataset, val_dataset, save_path, lr=1e-4, cfg=None):
        device = torch.device("cuda")
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr= lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.val_min = None
        # self.fg_seg_loss = torch.nn.BCELoss().cuda()
        self.cfg = cfg

    def euler_angles_to_rotation_matrix_batch(self, euler_angles):
        """
        Convert batched Euler angles to rotation matrices.
        Args:
        - euler_angles: Tensor of shape [bs, 3], where bs is the batch size.
                        Each row contains [yaw, pitch, roll] angles in radians.

        Returns:
        - Rotation matrices of shape [bs, 3, 3]
        """
        batch_size = euler_angles.shape[0]
        yaw = euler_angles[:, 0]  # Rotation around the z-axis
        pitch = euler_angles[:, 1]  # Rotation around the y-axis
        roll = euler_angles[:, 2]  # Rotation around the x-axis

        cos = torch.cos
        sin = torch.sin

        # Rotation matrices around the x, y, and z axes
        R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos(roll)
        R_x[:, 1, 2] = -sin(roll)
        R_x[:, 2, 1] = sin(roll)
        R_x[:, 2, 2] = cos(roll)

        R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_y[:, 0, 0] = cos(pitch)
        R_y[:, 0, 2] = sin(pitch)
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin(pitch)
        R_y[:, 2, 2] = cos(pitch)

        R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32)
        R_z[:, 0, 0] = cos(yaw)
        R_z[:, 0, 1] = -sin(yaw)
        R_z[:, 1, 0] = sin(yaw)
        R_z[:, 1, 1] = cos(yaw)
        R_z[:, 2, 2] = 1

        # Combined rotation matrix for each batch element
        R = torch.bmm(torch.bmm(R_x, R_y), R_z)
        return R

    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_aux1, loss_aux2 = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=4, norm_type=2)
        self.optimizer.step()
        return loss.item(), loss_aux1, loss_aux2

    def compute_loss(self, batch):
        device = self.device

        if self.cfg.stage == 'sdf':
            p = batch.get('off_surface_points').to(device)
            df_gt = batch.get('df').to(device) #(Batch,num_points)
            inputs = batch.get('on_surface_points').to(device)

            df_pred, mu, logvar = self.model(p, inputs)
            sdf_error = abs(df_pred - df_gt)
            sdf_near_mask = (sdf_error < 0.1).float().detach()
            sdf_w_error = sdf_error * sdf_near_mask + sdf_error * (1.0 - sdf_near_mask) * 0.5
            loss_sdf_uni, loss_sdf_nss = sdf_w_error[:, int(self.cfg.num_sample_points_training*0.9):], sdf_w_error[:, :int(self.cfg.num_sample_points_training*0.9)]
            loss_sdf = loss_sdf_uni.mean() * 0.5 + loss_sdf_nss.mean() * 0.5

            kl_div = - 0.5 * torch.mean(torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=-1))
            if self.model.training:
                loss = loss_sdf + 1e-4*kl_div
                return loss, loss_sdf.item(), kl_div.item()
            else:
                return loss_sdf.item()

        elif self.cfg.stage == 'pos':
            ## preds rota not rota-1
            inputs = batch.get('on_surface_points').to(device)
            so3 = batch.get('so3').to(device)
            predicted_euler, predicted_R = self.model(inputs) ## in [-1, 1]
            angle_distance = torch.remainder(predicted_euler - so3 + np.pi, 2 * np.pi) - np.pi

            R, pred_R = self.euler_angles_to_rotation_matrix_batch(so3).cuda(), self.euler_angles_to_rotation_matrix_batch(predicted_euler).cuda()
            R_inv, pred_R_inv = R.transpose(2, 1), pred_R.transpose(2, 1)
            point_loss = (torch.bmm(inputs, R_inv) - torch.bmm(inputs, pred_R_inv)).norm(p=2, dim=-1).mean(-1)

            loss_rata = abs(angle_distance)
            loss_rata = loss_rata.mean(-1)
            so3_diff = (torch.bmm(R, pred_R_inv) - torch.eye(3)[None, :, :].cuda()).norm(dim=(1, 2)).mean()

            loss = point_loss+loss_rata
            if self.model.training:
                return loss.mean(), loss_rata.mean(), point_loss.mean()
            else:
                return loss.mean()


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            loss_display, loss_aux1_display, loss_aux2_display, interval = 0, 0, 0, 140
            time_curr = time.time()

            for batch_idx, batch in enumerate(train_data_loader):
                iteration = epoch * len(train_data_loader) + batch_idx + 1

                loss, loss_aux1, loss_aux2 = self.train_step(batch)
                loss_display += loss
                loss_aux1_display += loss_aux1
                loss_aux2_display += loss_aux2

                if (batch_idx+1) % interval ==0:
                    loss_display /= interval
                    loss_aux1_display /= interval
                    loss_aux2_display /= interval
                    time_used = time.time() - time_curr
                    if self.cfg.stage == 'sdf':
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, LossSDF: {:.5f}, KL: {:.5f}'
                        ', lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(epoch, (batch_idx + 1), len(train_data_loader),
                        100. * (batch_idx + 1) / len(train_data_loader), iteration, loss_display, loss_aux1_display, loss_aux2_display,
                        self.optimizer.param_groups[0]['lr'], time_used, interval))
                    elif self.cfg.stage == 'pos':
                        self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, LossRota: {:.5f}, PointLoss: {:.5f}'
                        ', lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(epoch, (batch_idx + 1), len(train_data_loader),
                        100. * (batch_idx + 1) / len(train_data_loader), iteration, loss_display, loss_aux1_display, loss_aux2_display,
                        self.optimizer.param_groups[0]['lr'], time_used, interval))
                    time_curr = time.time()
                    loss_display, loss_aux1_display, loss_aux2_display = 0, 0, 0

                if (iteration) in [1000000]:#[800000, 1000000, 1100000]:
                    self.optimizer.param_groups[0]['lr'] = 0.3*self.optimizer.param_groups[0]['lr']

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
        path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 100
        with torch.no_grad():
            for _ in range(num_batches):
                try:
                    val_batch = self.val_data_iterator.next()
                except:
                    self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                    val_batch = self.val_data_iterator.next()
                sum_val_loss += self.compute_loss(val_batch)#.item()
        return sum_val_loss / num_batches