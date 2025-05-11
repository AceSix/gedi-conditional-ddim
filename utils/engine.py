from typing import Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out



def cosine_beta_schedule(T: int, s: float = 0.008, eps: float = 1e-5):
    """
    Nichol-Dhariwal cosine schedule with extra clamping so that
    0 < beta_t < 1  and  0 < alpha_bar_t <= 1  (avoids NaNs).
    """
    steps = torch.arange(T + 1, dtype=torch.float32)

    # 1) compute normalized cosine alpha_bar
    alphas_bar = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # ensure alpha_bar[0] == 1

    # 2) clamp for numerical safety
    alphas_bar.clamp_(min=eps, max=1.0)

    # 3) recover betas
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas.clamp_(min=eps, max=1.0 - eps)

    return betas



def make_timesteps(T: int, S: int, method: str = "linear"):
    """Index sequence for DDIM/DDPM sampling."""
    if method == "linear":
        return np.linspace(0, T - 1, S, dtype=int)
    if method == "quadratic":            # √t spacing
        return (np.linspace(0, np.sqrt(T - 1), S) ** 2).astype(int)
    if method == "karras":               # ρ-schedule (Karras et al. 2022)
        rho, i = 7.0, np.linspace(0, 1, S)
        return ((1 - i ** rho) * (T - 1)).astype(int)
    raise ValueError(f"Unknown method {method}")



class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: Union[str, Tuple[int, int]], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        #self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        if isinstance(beta, str) and beta.lower() == "cosine":
            self.register_buffer("beta_t", cosine_beta_schedule(T))
        else:
            self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

        # Optional: store log(ᾱₜ) once to avoid log/exp in KL‑loss variants
        self.register_buffer("log_alpha_bar", torch.log(alpha_t_bar))


    def forward(self, x_0, cond, drop_prob=0.25):
        # get a random training step $t \sim Uniform({1, ..., T})$
        B, dtype, device = x_0.size(0), x_0.dtype, x_0.device
        
        # 1. sample time‑step t and add noise
        t = torch.randint(self.T, size=(B,), device=device)
        
        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0, dtype=dtype)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate , t, x_0.shape) * epsilon)

        # 2. decide which items are unconditional
        mask = torch.rand(B, device=device) < drop_prob
        cond = cond.clone()
        cond[mask] = torch.ones_like(cond[mask]) * -1.0  # cond vectors are 0-1, so -1 represents no condition

        # 3. predict the noise
        eps_theta = self.model(x_t, t, cond)
        loss = F.mse_loss(eps_theta, epsilon, reduction="mean")

        '''# v prediction
        v_target = (extract(self.signal_rate, t, x_0.shape) * epsilon - 
                    extract(self.noise_rate, t, x_0.shape) * x_0)
        v_theta = self.model(x_t, t, cond)
        base_mse = (v_theta - v_target).pow(2).mean(dim=[1, 2])

        # min-SNR weighting
        snr = (self.signal_rate[t] / self.noise_rate[t])**2
        weights = torch.minimum(snr, torch.tensor(gamma, device=x_0.device)) / snr
        loss = (weights * base_mse).mean()

        if torch.isnan(self.signal_rate).any() or torch.isnan(self.noise_rate).any():
            raise ValueError("NaNs in schedule – check β computation!")
'''

        return loss



class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t, cond=None):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$
        """
        epsilon_theta = self.model(x_t, t, cond)
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta

        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, cond=None):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        mean, var = self.cal_mean_variance(x_t, t, cond)

        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, only_return_x_0: bool = True, interval: int = 1, **kwargs):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.
            kwargs: no meaning, just for compatibility.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        x = [x_t]
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
            for time_step in sampling_steps:
                x_t = self.sample_one_step(x_t, time_step)

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]



class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], 
                 T: int, guidance_scale: float = 1.5):
        super().__init__()
        self.model = model
        self.T = T
        self.guidance_scale = guidance_scale # TODO: check cond

        # generate T steps of beta
        if isinstance(beta, str) and beta.lower() == "cosine":
            beta_t = cosine_beta_schedule(T)
        else:
            beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, 
                        eta: float, cond = None, guidance_scale = None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # two forward passes of predicting noise
        epsilon_cond = self.model(x_t, t, cond)
        null_cond = torch.full_like(cond, -1)
        epsilon_uncond = self.model(x_t, t, null_cond)
        # calculate the final epsilon
        epsilon_theta_t = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one


    @torch.no_grad()
    def forward(self, x_t, steps: int = 1, method="linear", eta=0.1,
                only_return_x_0: bool = True, interval: int = 1, cond = None, guidance_scale = None):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        
        '''if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int32)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")'''

        time_steps = make_timesteps(self.T, steps, method)

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        #time_steps = time_steps + 1
        # shift by +1 then clip so max index is T-1 (safe)
        time_steps  = np.clip(time_steps + 1, 1, self.T - 1)
        
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta, cond, guidance_scale)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clamp(x_t, -10.0, 200.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
