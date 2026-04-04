import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pytorch_msssim import ms_ssim

def pad_to_match_size(x, target_shape):

    B_target, C_target, H_target, W_target = target_shape
    Bx, Cx, Hx, Wx = x.shape

    assert B_target == Bx, f"Batch size mismatch: {Bx} vs {B_target}"
    assert C_target == Cx, f"Channel size mismatch: {Cx} vs {C_target}"

    pad_h = H_target - Hx
    pad_w = W_target - Wx

    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Target size should be >= input size. Got input ({Hx}, {Wx}), target ({H_target}, {W_target})")

    x_padded = F.pad(x, (0, pad_w, 0, pad_h))

    return x_padded

def cosine_schedule(t, T_total):
    """
    Cosine schedule controlling the blur ratio alpha_t
    """
    return torch.cos((t / T_total) * math.pi / 2)

def impedance_to_reflectivity(z):
    r = (z[:, :, 1:, :] - z[:, :, :-1, :]) / (z[:, :, 1:, :] + z[:, :, :-1, :] + 1e-6)
    r = F.pad(r, (0, 0, 0, 1), mode='replicate')
    return r


class L1MSSSIMLoss(nn.Module):
    def __init__(self, l1_weight=0.3, ms_ssim_weight=0.7, win_size=5):
        super(L1MSSSIMLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.win_size = win_size

    def forward(self, output, target):

        l1_loss = F.l1_loss(output, target)
        ms_ssim_loss = 1 - ms_ssim(output, target, data_range=2.0, size_average=True, win_size=self.win_size)

        total_loss = self.l1_weight * l1_loss + self.ms_ssim_weight * ms_ssim_loss
        return total_loss

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            loss_type='l1_ms_ssim',
            *,
            image_size,
            channels=2,
            timesteps=1000,
            eta: float = 0.0,
            train_routine='Final',
            sampling_routine='default',
            ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.num_timesteps = int(timesteps)
        self.channels = channels
        self.image_size = image_size
        self.loss_type = loss_type
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.eta = eta
        self.T_total = timesteps
        self.device = 'cuda'
        self.wavelet_generator = TrainableWavelet(f0=20.0, length=0.096, dt=0.003, device=self.device)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        max_sigma = 0.05
        sigma_schedule = torch.linspace(0.0, max_sigma, timesteps)
        self.register_buffer('sigma_schedule', sigma_schedule)

        self.target_max = 12088
        self.target_min = 1136
        self.init_max = 10359
        self.init_min = 1671


    @torch.no_grad()
    def sample(self, condition, T_total,  device='cuda', t=None):
        """
        Inference process: take the condition (initial impedance + seismic data) as input
        and reconstruct the impedance image.

        Returns:
            img: final inverted normalized impedance
            direct_recons: first predicted normalized impedance
            seismic: input seismic section
        """
        self.denoise_fn.eval()

        init_impedance = condition[:, 0:1, :, :].to(device)
        seismic = condition[:, 1:2, :, :].to(device)
        B, C, H, W = seismic.shape
        init_norm = 2 * (init_impedance - self.init_min) / (self.init_max - self.init_min) - 1

        sigma_T = extract(self.sqrt_one_minus_alphas_cumprod,
                          torch.full((B,), self.num_timesteps - 1, device=device, dtype=torch.long), seismic.shape)
        img = seismic + sigma_T * torch.randn_like(seismic)

        direct_recons = None

        for step in reversed(range(self.num_timesteps)):
            step_t = torch.full((B,), step, dtype=torch.long, device=device)

            model_input = torch.cat([img, init_norm, seismic], dim=1)
            x0_pred_norm = self.denoise_fn(model_input, step_t)
            if direct_recons is None:
                direct_recons = x0_pred_norm.clone()

            if step > 0:
                prev_t = step - 1
                step_t1 = torch.full((B,), prev_t, dtype=torch.long, device=device)

                sqrt_abar_t = extract(self.sqrt_alphas_cumprod, step_t, img.shape)
                sqrt_1m_abar_t = extract(self.sqrt_one_minus_alphas_cumprod, step_t, img.shape)

                Z_pred = (x0_pred_norm + 1) * 0.5 * (self.target_max - self.target_min) \
                         + self.target_min
                x0 = Z_pred.squeeze(1)
                r = (x0[:, 1:, :] - x0[:, :-1, :]) / (x0[:, 1:, :] + x0[:, :-1, :] + 1e-6)  # [B, H-1, W]
                r_input = r.unsqueeze(1)  # [B, 1, H-1, W]

                K_t_list = []
                for i in range(B):
                    K_t_single = F.conv2d(r_input[i:i + 1], wavelet_t[i:i + 1], padding=(wavelet_t.shape[2] // 2, 0))
                K_t = torch.cat(K_t_list, dim=0)  # [B,1,H-1,W]
                # print(f"K_t shape:{K_t.shape}")
                S_syn_pred = pad_to_match_size(K_t, (B, 1, H, W))
                x_mix_t = sqrt_abar_t * x0_pred_norm + sqrt_1m_abar_t * S_syn_pred

                eps_pred = (img - x_mix_t) / sqrt_1m_abar_t
                sqrt_abar_t1 = extract(self.sqrt_alphas_cumprod, step_t1, img.shape)
                x_mix_t1 = sqrt_abar_t1 * x0_pred_norm + sqrt_1m_abar_t1 * S_syn_pred
                img = x_mix_t1 + sqrt_1m_abar_t1 * eps_pred

            else:
                img = x0_pred_norm

        self.denoise_fn.train()
        return img, direct_recons, S_syn_pred, seismic

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img1=None, img2=None, img3=None, t=None, times=None, eval=True):
        self.denoise_fn.eval()

        device = img1.device
        if t is None:
            t = self.num_timesteps

        init_impedance = img1.to(device)
        seismic = img2.to(device)
        target = img3.to(device)
        B, C, H, W = seismic.shape

        init_norm = 2 * (init_impedance - self.init_min) / (self.init_max - self.init_min) - 1
        target_norm = 2 * (target - self.target_min) / (self.target_max - self.target_min) - 1

        S_syn_list = []
        mix_list = []
        noisy_list = []
        S_syn_pred_list = []
        r_input_list = []

        for i in range(t):
            step = torch.full((B,), i, dtype=torch.long, device=device)
            x_mix, S_syn = self.q_sample(target=target, seismic=seismic, t=step)
            sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, step, x_mix.shape)
            x_noisy = x_mix + sigma_t * torch.randn_like(x_mix)
            S_syn_list.append(S_syn.clone())
            mix_list.append(x_mix.clone())
            noisy_list.append(x_noisy.clone())

        img = noisy_list[-1]
        Backward = []

        for step_val in reversed(range(t)):
            step_t = torch.full((B,), step_val, dtype=torch.long, device=device)
            model_input = torch.cat([img, init_norm, seismic], dim=1)
            x0_pred_norm = self.denoise_fn(model_input, step_t)
            x0_pred_norm = pad_to_match_size(x0_pred_norm, (B, 1, H, W))
            Z_pred = (x0_pred_norm + 1) * 0.5 * (self.target_max - self.target_min) + self.target_min
            x0 = Z_pred.squeeze(1)
            r = (x0[:, 1:, :] - x0[:, :-1, :]) / (x0[:, 1:, :] + x0[:, :-1, :] + 1e-6)

            wavelet_t = self.wavelet_generator(B)
            r_input = r.unsqueeze(1)
            r_input_list.append(r_input.clone())

            K_t_list = []
            for i in range(B):
                K_t_single = F.conv2d(r_input[i:i + 1], wavelet_t[i:i + 1], padding=(wavelet_t.shape[2] // 2, 0))
            K_t = torch.cat(K_t_list, dim=0)

            S_syn_pred = pad_to_match_size(K_t, (B, 1, H, W))
            S_syn_pred_list.append(S_syn_pred.clone())

            sqrt_abar_t = extract(self.sqrt_alphas_cumprod, step_t, img.shape)
            sqrt_1m_abar_t = extract(self.sqrt_one_minus_alphas_cumprod, step_t, img.shape)

            x_mix_t = sqrt_abar_t * x0_pred_norm + sqrt_1m_abar_t * S_syn_pred
            eps_pred = (img - x_mix_t) / sqrt_1m_abar_t

            if step_val > 0:
                step_t1 = torch.full((B,), step_val - 1, dtype=torch.long, device=device)
                sqrt_1m_abar_t1 = extract(self.sqrt_one_minus_alphas_cumprod, step_t1, img.shape)

                x_mix_t1 = sqrt_abar_t1 * x0_pred_norm + sqrt_1m_abar_t1 * S_syn_pred
                img = x_mix_t1 + sqrt_1m_abar_t1 * eps_pred
            else:
                img = x0_pred_norm
            Backward.append(img.clone())
        self.denoise_fn.train()
        return S_syn_list, mix_list, noisy_list, Backward, img, S_syn_pred_list, r_input_list


    def q_sample(self, target, seismic, t):

        B, C, H, W = target.shape
        target_norm = 2 * (target - self.target_min) / (self.target_max - self.target_min) - 1
        x0 = target.squeeze(1)
        r = (x0[:, 1:, :] - x0[:, :-1, :]) / (x0[:, 1:, :] + x0[:, :-1, :] + 1e-6)  # [B, H-1, W]
        r_input = r.unsqueeze(1)  # [B, 1, H-1, W]
        K_t_list = []
        for i in range(B):
            K_t_single = F.conv2d(r_input[i:i + 1], wavelet_t[i:i + 1], padding=(wavelet_t.shape[2] // 2, 0))
        K_t = torch.cat(K_t_list, dim=0)  # [B,1,H-1,W]

        # print(f"K_t shape:{K_t.shape}") # ([2, 1, 128, 128])
        S_syn = pad_to_match_size(K_t, (B, 1, H, W))

        a = extract(self.sqrt_alphas_cumprod, t, target.shape)
        b = extract(self.sqrt_one_minus_alphas_cumprod, t, target.shape)
        x_mix = a * target_norm + b * S_syn
        return x_mix, S_syn

    def p_losses(self, target, condition, t):
        init_impedance = condition[:, 0:1]
        seismic_data = condition[:, 1:2]

        init_max, init_min = self.init_max, self.init_min
        init_impedance_norm = 2 * (init_impedance - init_min) / (init_max - init_min) - 1

        target_max, target_min = self.target_max, self.target_min
        target_norm = 2 * (target - target_min) / (target_max - target_min) - 1

        x_mix, S_syn = self.q_sample(target=target, seismic=seismic_data, t=t)
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_mix.shape)
        x_mix_noisy = x_mix + sigma_t * torch.randn_like(x_mix)
        model_input = torch.cat([x_mix_noisy, init_impedance_norm, seismic_data], dim=1)
        x_recon = self.denoise_fn(model_input, t)

        pred_loss = L1MSSSIMLoss()(x_recon, target_norm)
        phys_loss = F.mse_loss(S_syn, seismic_data)

        λ_phys = 0.3
        loss = pred_loss + λ_phys * phys_loss

        return loss


    def forward(self, target, condition, *args, **kwargs):
        """
        Training entry of the model: takes seismic data and initial impedance as input
        and outputs the predicted impedance.
        """
        b, c, h, w = target.shape
        assert h == self.image_size and w == self.image_size, f'Expected {self.image_size}, got {h}x{w}'
        t = torch.randint(0, self.num_timesteps, (b,), device=target.device).long()
        return self.p_losses(target, condition, t, *args, **kwargs)
