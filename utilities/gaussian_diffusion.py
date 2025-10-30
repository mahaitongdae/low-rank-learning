from typing import Callable
import math
import torch
from utilities.functions import normal_kl, discretized_gaussian_loglik, flat_mean
import copy
import numpy as np

def _warmup_beta(beta_start, beta_end, timesteps, warmup_frac, dtype):
    betas = beta_end * torch.ones(timesteps, dtype=dtype)
    warmup_time = int(timesteps * warmup_frac)
    betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=dtype)
    return betas


def get_beta_schedule(beta_schedule, beta_start, beta_end, timesteps, dtype=torch.float64):
    if beta_schedule == 'quad':
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=dtype) ** 2
    elif beta_schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.1, dtype=dtype)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.5, dtype=dtype)
    elif beta_schedule == 'cosine':
        # Nichol & Dhariwal cosine schedule (Improved DDPM)
        # alpha_bar(t) = cos^2(((t/T) + s)/(1+s) * pi/2)
        # beta_t = 1 - alpha_bar(t+1)/alpha_bar(t)
        s = 0.008
        def alpha_bar(t_frac: float) -> float:
            return math.cos((t_frac + s) / (1.0 + s) * math.pi / 2.0) ** 2
        betas_list = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            beta = 1.0 - alpha_bar(t2) / alpha_bar(t1)
            # Clamp to avoid singularities at the end of the schedule
            betas_list.append(min(max(beta, 1e-8), 0.999))
        betas = torch.tensor(betas_list, dtype=dtype)
    elif beta_schedule == 'const':
        betas = beta_end * torch.ones(timesteps, dtype=dtype)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(timesteps, 1, timesteps, dtype=dtype)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (timesteps, )
    return betas

# energy_func_gmm2 = lambda x: (0.8 * torch.exp(- torch.linalg.norm(x - 3., axis=1) ** 2 / 2)
#                   + 0.2 * torch.exp(- torch.linalg.norm(x + 3., axis=1) ** 2 /2 ))

def energy_func_gmm(x):
    """_summary_

    Args:
        x (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    return torch.log(0.8 * torch.exp(- torch.linalg.norm(x - 3., axis=-1) ** 2 / 2)
                  + 0.2 * torch.exp(- torch.linalg.norm(x + 3., axis=-1) ** 2 / 2))

class GaussianDiffusion:

    def __init__(self, betas, model_mean_type, model_var_type, loss_type,
                 **kwargs):
        assert isinstance(betas, torch.Tensor)
        assert (betas > 0).all() and (betas <= 1).all()
        self.betas = betas
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.sampling_dist = kwargs.get('sampling_dist')

        self.timesteps = len(betas)
        print("Timesteps:", self.timesteps)

        alphas = 1 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([
            torch.as_tensor([
                1.,
            ], dtype=torch.float64), self.alphas_bar[:-1]
        ])

        # q(x_t | x_0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        # q(x_{t-1} | x_t, x_0)
        # refer to the formula 1-3 in README.md
        sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / self.alphas_bar -
                                                   1.)  # m1: minus 1
        self.posterior_var = betas * (1. - alphas_bar_prev) / (1. -
                                                               self.alphas_bar)
        self.posterior_logvar_clipped = torch.log(
            torch.cat([self.posterior_var[[1]], self.posterior_var[1:]]))
        self.posterior_mean_coef1 = betas * sqrt_alphas_bar_prev / (
            1. - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (
            1. - alphas_bar_prev) / (1. - self.alphas_bar)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large":
            (self.betas,
             torch.log(torch.cat([self.posterior_var[[1]], self.betas[1:]]))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

    def log_expectation_reward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        energy_function,
        num_mc_samples: int,
    ):
        repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

        samples = self.reverse_sample(repeated_x, t)

        log_rewards = energy_function(samples)

        return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)

    def estimate_grad_Rt(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        energy_function,
        num_mc_samples: int = 100,
    ):
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(len(x))

        grad_fxn = torch.func.grad(self.log_expectation_reward, argnums=1)
        vmapped_fxn = torch.vmap(grad_fxn,
                                 in_dims=(0, 0, None, None),
                                 randomness="different")

        return vmapped_fxn(t, x, energy_function, num_mc_samples)

    @staticmethod
    def _extract(arr,
                 t,
                 x,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 ndim=4):
        """
        Extract noise schedule coefficients.
        
        Args:
            arr: (timesteps,)
            t: (t_dim,), usually [1, ] or [B, ]
            x: (B, ...)
        Returns:
            out: (t_dim, (x.ndim - 1) * (1,))
        """
        if x is not None:
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        out = torch.as_tensor(arr, dtype=dtype, device=device).gather(0, t)  # gather arr at dimension 0 and index t
        return out.reshape((-1, ) + (1, ) * (ndim - 1))

    def q_mean_var(self, x_0, t):
        mean = self._extract(self.sqrt_alphas_bar, t, x_0) * x_0
        var = self._extract(1. - self.alphas_bar, t, x_0)
        logvar = self._extract(torch.log(1 - self.alphas_bar), t, x_0)
        return mean, var, logvar

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        coef1 = self._extract(self.sqrt_alphas_bar, t, x_0)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0)
        return coef1 * x_0 + coef2 * noise

    def reverse_sample(self, x_t, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_t)
        coef1 = self._extract(self.sqrt_recip_alphas_bar, t, x_t)
        coef2 = self._extract(self.sqrt_recip_m1_alphas_bar, t, x_t)
        return coef1 * x_t - coef2 * noise

    def q_posterior_mean_var(self, x_0, x_t, t):
        """
        q(x_{t - 1} | x_t, x_0)
        Equation 6, 7 in Denoising Diffusion Probabilistic Models,
        https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

        """
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x_0)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x_0)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        posterior_var = self._extract(self.posterior_var, t, x_0)
        posterior_logvar = self._extract(self.posterior_logvar_clipped, t, x_0)
        return posterior_mean, posterior_var, posterior_logvar

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        """
        input is x_t, t
        output x_{t-1} ~ p(x_{t-1} | x_t)

        """
        if x_t.ndim == 2:
            B, N = x_t.shape
        else:
            B, C, H, W = x_t.shape
        out = denoise_fn(x_t, t)  # \epsilon(x_t, t)

        if self.model_var_type == "learned" and x_t.ndim == 4:
            assert all(out.shape == (B, 2 * C, H, W))
            out, model_logvar = out.chunk(2, dim=1)
            model_var = torch.exp(model_logvar)
        elif self.model_var_type in ["fixed-small", "fixed-large"]:
            model_var, model_logvar = self._extract(self.fixed_model_var, t, x_t),\
                                      self._extract(self.fixed_model_logvar, t, x_t)
        else:
            raise NotImplementedError(self.model_var_type)

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (
            lambda x: x)
        if self.model_mean_type == "mean":
            pred_x_0 = _clip(self._pred_x_0_from_mean(x_t=x_t, mean=out, t=t))
            model_mean = out
        elif self.model_mean_type == "x_0":
            pred_x_0 = _clip(out)
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0,
                                                       x_t=x_t,
                                                       t=t)
        elif self.model_mean_type == "eps":
            pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
            model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0,
                                                       x_t=x_t,
                                                       t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    def _pred_x_0_from_mean(self, x_t, mean, t):
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t)
        return mean / coef1 - coef2 / coef1 * x_t

    def _pred_x_0_from_eps(self, x_t, eps, t):
        """
        x_0 = 1 / \sqrt(\bar \alpha_t) x_t  - \sqrt(1 / \bar\alpha_t - 1) \eps
        """
        coef1 = self._extract(self.sqrt_recip_alphas_bar, t, x_t)
        coef2 = self._extract(self.sqrt_recip_m1_alphas_bar, t, x_t)
        return coef1 * x_t - coef2 * eps

    # === sample ===

    def p_sample_step(self,
                      denoise_fn,
                      x_t,
                      t,
                      clip_denoised=False,
                      return_pred=False,
                      generator=None):
        '''
        input: x_t, t
        output: x_{t - 1}
        '''
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t, t, clip_denoised=clip_denoised, return_pred=True)
        noise = torch.empty_like(x_t).normal_(generator=generator)
        nonzero_mask = (t > 0).reshape((-1, ) + (1, ) * (x_t.ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(
            0.5 * model_logvar) * noise
        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(self,
                 denoise_fn,
                 shape=None,
                 device=torch.device("cpu"),
                 noise=None,
                 seed=None):
        """
        Sample x through the reverse diffusion process.

        Args:
            denoise_fn: The denoising function.
            shape: The shape of the input.
            device: The device to use.
            noise: The noise to use.
            seed: The seed to use.

        Returns:
            x_t: The sampled x.
        """
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(denoise_fn, x_t, t, generator=rng)
        return x_t

    def p_sample_grad(self,
                      denoise_fn,
                      shape=None,
                      device=torch.device("cpu"),
                      noise=None,
                      seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(denoise_fn, x_t, t, generator=rng)
        return x_t
    
    def p_sample_idem_from_energy(self,
                      energy_fn,
                      shape=None,
                      device=torch.device("cpu"),
                      noise=None,
                      seed=None):
        assert self.model_mean_type == "eps"
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
            
        def denoise_fn(x_t, t):
            return self.get_idem_noise_from_clean_energy_fn(x_t, t, energy_fn)
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(denoise_fn, x_t, t, generator=rng)
        return x_t

    @torch.inference_mode()
    def p_sample_save_all(self,
                          denoise_fn,
                          shape=None,
                          device=torch.device("cpu"),
                          noise=None,
                          seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty((B, ), dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        noise_copy = copy.deepcopy(x_t)
        pred_x0_all = []
        xt_all = []
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t, pred_x0 = self.p_sample_step(denoise_fn,
                                              x_t,
                                              t,
                                              generator=rng,
                                              return_pred=True)
            pred_x0_all.append(pred_x0.clone().detach().cpu())
            xt_all.append(x_t.clone().detach().cpu())
        return x_t, noise_copy, xt_all, pred_x0_all

    @torch.inference_mode()
    def p_sample_progressive(self,
                             denoise_fn,
                             shape,
                             device=torch.device("cpu"),
                             noise=None,
                             pred_freq=10,
                             seed=None):
        B = (shape or noise.shape)[0]
        t = torch.empty(B, dtype=torch.int64, device=device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device).manual_seed(seed)
        if noise is None:
            x_t = torch.empty(shape, device=device).normal_(generator=rng)
        else:
            x_t = noise.to(device)
        L = self.timesteps // pred_freq
        preds = torch.zeros((L, B) + shape[1:], dtype=torch.float32)
        idx = L
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(denoise_fn,
                                           x_t,
                                           t,
                                           return_pred=True,
                                           generator=rng)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds

    # === log likelihood ===
    # bpd: bits per dimension

    def _loss_term_bpd(self, denoise_fn, x_0, x_t, t, clip_denoised,
                       return_pred):
        # calculate L_t
        # t = 0: negative log likelihood of decoder, -\log p(x_0 | x_1)
        # t > 0: variational lower bound loss term, KL term
        true_mean, _, true_logvar = self.q_posterior_mean_var(x_0=x_0,
                                                              x_t=x_t,
                                                              t=t)
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn,
            x_t=x_t,
            t=t,
            clip_denoised=clip_denoised,
            return_pred=True)
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        decoder_nll = discretized_gaussian_loglik(x_0,
                                                  model_mean,
                                                  log_scale=0.5 *
                                                  model_logvar).neg()
        decoder_nll = flat_mean(decoder_nll) / math.log(2.)
        output = torch.where(t.to(kl.device) > 0, kl, decoder_nll)
        return (output, pred_x_0) if return_pred else output

    def train_losses(self, denoise_fn, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise=noise)

        # calculate the loss
        # kl: weighted
        # mse: unweighted
        if self.loss_type == "kl":
            losses = self._loss_term_bpd(denoise_fn,
                                         x_0=x_0,
                                         x_t=x_t,
                                         t=t,
                                         clip_denoised=False,
                                         return_pred=False)
        elif self.loss_type == "mse":
            assert self.model_var_type != "learned"
            if self.model_mean_type == "mean":
                target = self.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]
            elif self.model_mean_type == "x_0":
                target = x_0
            elif self.model_mean_type == "eps":
                target = noise
            else:
                raise NotImplementedError(self.model_mean_type)
            model_out = denoise_fn(x_t, t)
            losses = flat_mean((target - model_out).pow(2))
        elif self.loss_type == "rssm":
            noise_2 = torch.randn_like(x_0)
            if self.sampling_dist == 'uniform':
                # x_t =
                sample_xt = torch.rand_like(x_0) * 10 - 5
            elif self.sampling_dist == 'Gaussian':
                sample_xt = torch.randn_like(x_0) * 3
            elif self.sampling_dist == 'pt':
                sample_xt = x_t.clone()
            else:
                raise NotImplementedError
            tilde_x_0 = self.reverse_sample(sample_xt, t, noise=noise_2)
            energy = 100 * (
                0.8 * torch.exp(-torch.linalg.norm(
                    tilde_x_0 - 3 * torch.ones_like(x_t), axis=1)**2 / 2) +
                0.2 * torch.exp(-torch.linalg.norm(
                    tilde_x_0 + 3 * torch.ones_like(x_t), axis=1)**2 / 2))
            model_out = denoise_fn(sample_xt, t)
            losses = energy * flat_mean((noise_2 - model_out).pow(2))

        elif self.loss_type == "idem":
            sample_xt = torch.rand_like(x_0) * 10 - 5

            # grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
            # vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None), randomness="different")
            score = self.estimate_grad_Rt(
                t,
                sample_xt,
                energy_func_gmm2,
            )
            coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0)
            model_out = denoise_fn(sample_xt, t)
            losses = flat_mean((score * coef2 + model_out).pow(2))
        else:
            raise NotImplementedError(self.loss_type)

        return losses, torch.linalg.norm(model_out, axis=1).mean().item()

    def _prior_bpd(self, x_0):
        B, T = len(x_0), self.timesteps
        T_mean, _, T_logvar = self.q_mean_var(x_0=x_0,
                                              t=(T - 1) * torch.ones(
                                                  (B, ), dtype=torch.int64))
        kl_prior = normal_kl(T_mean, T_logvar, mean2=0., logvar2=0.)
        return flat_mean(kl_prior) / math.log(2.)

    def calc_all_bpd(self, denoise_fn, x_0, clip_denoised=True):
        B, T = x_0.shape, self.timesteps
        t = torch.empty([
            B,
        ], dtype=torch.int64)
        t.fill_(T - 1)
        losses = torch.zeros([B, T], dtype=torch.float32)
        mses = torch.zeros([B, T], dtype=torch.float32)

        for i in range(T - 1, -1, -1):
            x_t = self.q_sample(x_0, t=t)
            loss, pred_x_0 = self._loss_term_bpd(denoise_fn,
                                                 x_0,
                                                 x_t=x_t,
                                                 t=t,
                                                 clip_denoised=clip_denoised,
                                                 return_pred=True)
            losses[:, i] = loss
            mses[:, i] = flat_mean((pred_x_0 - x_0).pow(2))

        prior_bpd = self._prior_bpd(x_0)
        total_bpd = torch.sum(losses, dim=1) + prior_bpd
        return total_bpd, losses, prior_bpd, mses
    
    ## IDEM score functioons
    def get_idem_score_single(self, x_t: torch.Tensor, t: float, recon_fn: Callable,
                          energy_fn: Callable, num_mc_samples: int = 100):
        """
        x_t: (x_shape,)
        recon_fn: (x_shape,) -> (x_shape,)
        energy: (x_shape,) -> (x_shape,)
        """
        assert x_t.ndim == 1
        x_shape = x_t.shape[0]
        size = num_mc_samples
        noise = torch.randn([size, x_shape]) * t
        samples = self.reverse_sample(x_t, t, noise)
        energy = energy_fn(samples)
        lse = torch.logsumexp(energy, dim=-1)
        return lse

    def get_idem_noise_from_clean_energy_fn(self, x_t, t, energy_fn, num_mc_samples: int = 100):
        """
        Get the score function from energy funtion using IDEM, https://arxiv.org/pdf/2402.06121.
        
        """
        x_t = x_t.detach_().requires_grad_(True)
        # lse = torch.vmap(self.get_idem_score_single, (0, 0, None, None),
        #                  randomness="different")(x_t, t, energy_fn)
        noise = torch.randn([num_mc_samples, *x_t.shape])  # (num_mc_samples, B, x_dim)
        x_0 = self.reverse_sample(x_t, t, noise)
        energy = energy_fn(x_0)
        assert energy.ndim == x_0.ndim - 1
        lse = torch.logsumexp(energy, dim=0)  # (B, )
        score = torch.autograd.grad(lse.sum(), x_t)[0]  # score function
        scale = self._extract(self.sqrt_one_minus_alphas_bar, t,
                              x_t)  # predicted noise
        return -scale * score


def test_gaussian_diffusion():
    betas = torch.linspace(0.0001, 0.02, 1000)
    gd = GaussianDiffusion(betas,
                           model_mean_type="mean",
                           model_var_type="fixed-large",
                           loss_type="mse")
    x_t = gd.p_sample(denoise_fn=lambda x, t: x, shape=(1, 2, 28, 28))
    print(x_t.shape)

def test_idem_score_unbalanced_gmm(beta_scale: float = 0.3):
    import matplotlib.pyplot as plt
    betas = beta_scale * get_beta_schedule(beta_schedule="cosine", beta_start=0.0, beta_end=1.0, timesteps=20)
    gd = GaussianDiffusion(betas,
                           model_mean_type="eps",
                           model_var_type="fixed-large",
                           loss_type="mse")
    x_t = torch.randn(16, 2)
    print(x_t.shape)
    score = gd.get_idem_noise_from_clean_energy_fn(x_t, torch.tensor([5]), energy_func_gmm)
    print(score.shape)
    print("Test the full generation process")
    x_t = gd.p_sample_idem_from_energy(energy_func_gmm, shape=(1024, 2))
    print(x_t.shape)
    x_t = x_t.detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.scatter(x_t[:, 0], x_t[:, 1], s=5, alpha=0.5)
    plt.axis("equal")
    plt.grid(True)
    plt.title("IDEM generated samples")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"./figures/idem_generated_samples_beta_scale_{beta_scale}.jpg")

def test_energy_func_gmm():
    x = torch.randn(10, 10, 2)
    energy = energy_func_gmm(x)
    assert energy.ndim == x.ndim - 1
    print(energy.shape)

if __name__ == '__main__':
    for beta_scale in [0.1, 0.3, 0.5, 0.7, 0.9]:
        test_idem_score_unbalanced_gmm(beta_scale)
