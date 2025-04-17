import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
import torch.nn.functional as F

# extract the appropriate t index for a batch of indices
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    #print("using LINEAR schedule")
    scale = 1000 / timesteps ## maybe scale=1 is enough, in default DDPM, beta is from 0.0001 to 0.02
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

class Trainer(object):
    def __init__(self, cond_net, diffuse_net, VAE, logger, train_dataset, val_dataset, save_path, lr=1e-4, cfg=None, num_timesteps=1000):
        device = torch.device("cuda")
        self.cond_net = cond_net.to(device)
        self.diffuse_net = diffuse_net.to(device)
        self.VAE = VAE.to(device)

        self.device = device
        self.optimizer = optim.Adam([{'params': self.cond_net.parameters()}, {'params': self.diffuse_net.parameters()}],
                                 lr=cfg.lr)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.val_min = None
        self.num_timesteps = num_timesteps
        self.cfg = cfg

        ### DDPM parameters
        self.betas = linear_beta_schedule(num_timesteps).cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).cuda()
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.).cuda()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).cuda()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).cuda()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).cuda()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).cuda()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).cuda()
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20)).cuda()

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).cuda()
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(self.alphas)/ (1.0 - self.alphas_cumprod)).cuda()

        # self.p2_loss_weight = (1 + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** - 0
        self.ddim_sampling_timesteps = 20
        self.ddim_sampling_eta = 0#1 # 0 means generate a fixed data from given noise and condition


    def train_step(self,batch):
        self.cond_net.train(), self.diffuse_net.train(), self.VAE.eval()
        self.optimizer.zero_grad()
        loss, loss_100, loss_1000 = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_100, loss_1000


    def compute_loss(self, batch):
        device = self.device

        inputs = batch.get('on_surface_points').to(device)

        # z_clean = mu#eps * std + mu
        # cond_feat = self.cond_net(inputs)
        z_clean, mu, logvar, pred_R_inv = self.VAE.encode(inputs, vae_training=True)
        # z_clean = self.VAE.reparameterize(mu, logvar)
        cond_feat = self.cond_net(torch.bmm(inputs, pred_R_inv).detach())

        # STEP 1: sample timestep
        t = torch.randint(0, self.num_timesteps, (z_clean.shape[0],), device=z_clean.device).long()
        noise = torch.randn_like(z_clean)

        # STEP 2: create noisy data
        z_noisy = self.q_sample(x_start=z_clean, t=t, noise=noise).float()
        target = z_clean

        # STEP 3: pass to forward function
        z_denoised = self.diffuse_net(z_noisy, cond_feat, t) ## [bs, 256], t:[bs]
        loss = F.mse_loss(z_denoised, target, reduction='none')

        # loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        unreduced_loss = loss.detach().clone().mean(dim=1)

        loss_100 = unreduced_loss[t<100].mean().detach()
        loss_1000 = unreduced_loss[t>100].mean().detach()

        if (t<100).sum()>0:
            return loss.mean(), loss_100.item(), loss_1000.item()
        else:
            return loss.mean(), None, loss_1000.item()

    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader()
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            loss_display, loss100_display, loss1000_display, interval, loss100_num = 0, 0, 0, 140, 0
            time_curr = time.time()

            for batch_idx, batch in enumerate(train_data_loader):
                iteration = epoch * len(train_data_loader) + batch_idx + 1

                loss, loss_100, loss_1000 = self.train_step(batch)
                loss_display += loss
                if loss_100 is not None:
                    loss100_display += loss_100
                    loss100_num += 1
                loss1000_display += loss_1000

                if (batch_idx+1) % interval ==0:
                    loss_display /= interval
                    loss100_display /= loss100_num
                    loss1000_display /= interval
                    time_used = time.time() - time_curr
                    self.logger.info(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.5f}, Loss100: {:.5f}, Loss1000: {:.5f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)'.format(
                            epoch, (batch_idx + 1), len(train_data_loader), 100. * (batch_idx + 1) / len(train_data_loader),
                            iteration, loss_display, loss100_display, loss1000_display, self.optimizer.param_groups[0]['lr'], time_used, interval))
                    time_curr = time.time()
                    loss_display, loss100_display, loss1000_display = 0, 0, 0

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
                        'diffuse_net_state_dict': self.diffuse_net.state_dict(),
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

                pred_clean_z = self.ddim_sample(cond_feat)

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

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def ddim_sample(self, cond):
        times = torch.tensor(list(reversed(range(0, self.num_timesteps, self.num_timesteps//self.ddim_sampling_timesteps)))).long()
        ### why not from 1000 to 0, current is from 900 to 0
        # + 1#torch.linspace(self.num_timesteps, 0, steps=self.ddim_sampling_timesteps + 1).long()#[:-1]
        # x_T = torch.randn(cond.shape[0], 256).cuda()
        x_T = torch.zeros(cond.shape[0], 256).cuda()
        for i in range(1, self.ddim_sampling_timesteps):
            # print(i)
            curr_t = times[i - 1]# - 1
            next_t = times[i]# - 1
            alpha = self.alphas_cumprod_prev[curr_t] ## 0
            alpha_next = self.alphas_cumprod_prev[next_t] ### 1

            batch_t = torch.full((cond.shape[0],), next_t).long().cuda()

            ### STEP 1: predict x_0 from x_t, but this x_0 cannot be the generated result, it should be used to produce x_t-1
            pred_x0 = self.diffuse_net(x_T, cond, batch_t.float())
            pred_noise = self.predict_noise_from_start(x_T, batch_t, pred_x0)

            # if clip_denoised:
            #     x_start.clamp_(-1., 1.)

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x_T) if next_t > 0 else 0.

            x_T = pred_x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise
            x_T = x_T.float()
        return x_T


    @torch.no_grad()
    def sample(self, cond):
        x_T = torch.randn(cond.shape[0], 256).cuda()
        for t in reversed(range(0, self.num_timesteps)):
            batch_t = torch.full((cond.shape[0],), t).long().cuda()

            ### STEP 1: predict x_0 from x_t, but this x_0 cannot be the generated result, it should be used to produce x_t-1
            pred_x0 = self.diffuse_net(x_T, cond, batch_t)
            pred_noise = self.predict_noise_from_start(x_T, batch_t, pred_x0)

            # if clip_denoised:
            #     x_start.clamp_(-1., 1.)

            ### STEP 2: get x_t-1 from predicted noise, x_t and predicted x_0
            model_mean, _, model_log_variance = self.q_posterior(x_start=pred_x0, x_t=x_T, t=batch_t)

            noise = torch.randn_like(x_T) if t > 0 else 0.  # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_T