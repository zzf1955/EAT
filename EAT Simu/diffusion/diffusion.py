import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from .utils import Progress, Silent

class Diffusion(nn.Module):
    def __init__(self, input_dim, output_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = input_dim
        self.action_dim = output_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({'t': i})
            if return_diffusion: diffusion.append(x)
        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x, None

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action, t_step_res = self.p_sample_loop(state, shape, *args, **kwargs)
        if t_step_res:
            return action, t_step_res
        else:
            return action, None

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise, weights)
        else:
            return self.loss_fn(x_recon, x_start, weights)

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        print("Stop!")
        input()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        if (isinstance(state,np.ndarray)):
            state = torch.from_numpy(state).float().to(self.device)
        return self.sample(state, *args, **kwargs)

class PostDiffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True):
        super(PostDiffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, x, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({'t': i})
            if return_diffusion: diffusion.append(x)
        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x, None

    def sample(self, state, x0, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action, t_step_res = self.p_sample_loop(x0, state, shape, *args, **kwargs)
        if t_step_res:
            return action, t_step_res
        else:
            return action, None

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            return self.loss_fn(x_recon, noise, weights)
        else:
            return self.loss_fn(x_recon, x_start, weights)

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        print("Stop!")
        input()
        return self.p_losses(x, state, t, weights)

    # def interpolate(self,tensor, output_dim):
    #     # 确保输入 tensor 的形状是 [1, 原始长度]
    #     input_dim = tensor.size(1)
        
    #     # 调整 tensor 形状为 [1, 1, 原始长度]，以便使用 interpolate 函数
    #     tensor = tensor.unsqueeze(1)  # [1, 1, input_dim]
        
    #     # 使用 F.interpolate 进行插值，调整到 [1, 1, output_dim]，然后移除多余维度
    #     interpolated_tensor = F.interpolate(tensor, size=output_dim, mode='linear', align_corners=True)
        
    #     # 恢复为 [1, output_dim] 的形状
    #     return interpolated_tensor.squeeze(1)

    def forward(self, x0, state, *args, **kwargs):

        # x0 = self.interpolate(x0,self.action_dim)

        if (isinstance(x0,np.ndarray)):
            state = torch.from_numpy(x0).float().to(self.device)

        if (isinstance(state,np.ndarray)):
            state = torch.from_numpy(state).float().to(self.device)
        
        return self.sample(state, x0, *args, **kwargs)
    
class DDIM(nn.Module):
    def __init__(self, input_dim, output_dim, model, n_timesteps):
        super(DDIM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        self.n_timesteps = n_timesteps

        # 设置每个时间步的噪声缩放因子
        self.betas = torch.linspace(0.0001, 0.02, self.n_timesteps)  # 线性变化的噪声
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累乘

    def forward(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.output_dim)
        x = torch.randn(shape).to(device = "cuda")

        # 将 numpy 数组转换为 PyTorch 张量
        state = torch.tensor(state, dtype=torch.float32).to(device = "cuda")

        # 初始化扩散过程，从 x_n_timesteps 开始，逐步去噪
        for t in reversed(range(self.n_timesteps)):
            t_tensor = torch.full((batch_size,), t, device="cuda", dtype=torch.long)

            # 使用模型预测去噪结果
            predicted_noise = self.model(x, t_tensor, state)

            # 计算无噪声图像的预测
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            
            x_pred = (x - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

            # 使用DDIM去噪公式更新 x
            x = alpha_t_prev.sqrt() * x_pred + (1 - alpha_t_prev).sqrt() * predicted_noise

        # 返回去噪后的输出，并调整形状为 output_dim
        return x.view(-1, self.output_dim), None
