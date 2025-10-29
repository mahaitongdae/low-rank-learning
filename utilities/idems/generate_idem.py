from typing import Callable
import numpy as np
from utilities.gaussian_diffusion import *
import matplotlib.pyplot as plt
import torch



energy_func_gmm2 = lambda x: torch.log(0.8 * torch.exp(- torch.linalg.norm(x - 3., axis=1) ** 2 / 2)
                  + 0.2 * torch.exp(- torch.linalg.norm(x + 3., axis=1) ** 2 /2 )).sum(dim=-1)


def get_idem_score_single(x_t: torch.Tensor, t: float, recon_fn: Callable,
                          energy_fn: Callable):
    """
    x_t: (x_shape,)
    recon_fn: (x_shape,) -> (x_shape,)
    energy: (x_shape,) -> (x_shape,)
    """
    assert x_t.ndim == 1
    x_shape = x_t.shape[0]
    size = 100
    noise = torch.randn([size, x_shape]) * t
    samples = recon_fn(x_t, t)
    energy = energy_fn(samples)
    lse = torch.logsumexp(energy, dim=-1)
    return lse


def get_idem_score_unbalanced_gmm(self, x_t, t):
    x_t = x_t.detach_().requires_grad_(True)
    lse = torch.vmap(get_idem_score_single, (0, 0, None), randomness="different")(x_t, t, energy_func_gmm2)
    score = torch.autograd.grad(lse.sum(), x_t)[0]  # score function
    scale = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t)  # predicted noise
    return - scale * score

if __name__ == '__main__':
    betas = torch.linspace(0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type="eps", model_var_type="fixed-large", loss_type="mse")


    def denoise_fn(x_t, t):
        return diffusion.get_idem_score_unbalanced_gmm(x_t, t, energy_func_gmm2)


    def sample_fn(n):
        shape = (n,) + (2,)
        sample = diffusion.p_sample_grad(
            denoise_fn=denoise_fn, shape=shape, device="cpu", noise=None)
        return sample.detach().cpu().numpy()

        # if evaluator is not None:


    eval_results = evaluator.eval(sample_fn)
    gen_data = eval_results['x_gen']

    from utils.plot import plot_hist

    plot_hist(gen_data, 'iDEM', 'generate_idem_202510.pdf')
