import torch
import torch.nn.functional as F


def get_targets(q_t, r_t, done_t, discount_t, n, lambda_t):
    """
    Compute the strided n-step bootstrap return targets over a sequence. An analogy of rlax.n_step_bootstrap_return.
    
        G_t = r_t + discount_t * (lambda_t * G_{t+1} + (1 - lambda_t) * q_t)
        
    """
    T = q_t.shape[0]
    device = q_t.device
    # discounts per step: gamma * (1 - done)
    discounts = discount_t * (1.0 - done_t)

    # Right pad time with n zeros so slicing k:k+T works for k in [0, n]
    pad_size = min(n - 1, T)
    # if T < n - 1, q_values[n - 1:] is [], all the targets are padded with the last value of q_values
    targets = torch.cat([q_t[n - 1:], torch.tile(q_t[-1], [pad_size])])
    # If
    rewards_pad = torch.cat([r_t, torch.zeros(n - 1, device=device)])
    discounts_pad = torch.cat([discounts, torch.ones(n - 1, device=device)])
    values_pad = torch.cat([q_t, torch.tile(q_t[-1], [n - 1])])

    # Backward accumulate n rewards: r_{t+k} + discount_{t+k} * (...)
    for k in range(n - 1, -1, -1):
        r_slice = rewards_pad[k:k + T]
        disc_slice = discounts_pad[k:k + T]
        values_slice = values_pad[k:k + T]
        targets = r_slice + disc_slice * (lambda_t * targets + (1 - lambda_t) * values_slice)
    return targets


def td_n_loss(target_q_values: torch.Tensor,
              predicted_q_values: torch.Tensor,
              rewards: torch.Tensor,
              dones: torch.Tensor,
              gamma: float,
              n: int,
              lambda_: float = 1.0,
              mask: torch.Tensor | None = None,
              reduction: str = 'mean') -> tuple[torch.Tensor, dict]:
    """
    Compute TD-n loss for time-major tensors.

    Parameters
    ----------
    target_q_values : torch.Tensor
        Target Q values for each (s_t, a_t) with shape [T, B, 1].
    predicted_q_values : torch.Tensor
        Predicted Q values for each (s_t, a_t) with shape [T, B, 1].
    rewards : torch.Tensor
        Rewards r_t with shape [T, B, 1].
    dones : torch.Tensor
        Episode termination flags for transition at t, shape [T, B, 1].
        Non-zero value indicates episode ended after time t.
    gamma : float
        Discount factor.
    n : int
        N-step horizon.
    lambda_ : float
        Lambda parameter for the lambda-return.
    mask : torch.Tensor | None
        Optional validity mask of shape [T, B, 1], 1 for valid steps, 0 for padded.
        If None, all steps are treated as valid.
    reduction : str
        'mean' or 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Scalar TD-n loss.
    info : dict
        Diagnostic scalars.
    """
    if predicted_q_values.dim() == 3:  # [T, B, 1] -> [T, B]
        predicted_q_values = predicted_q_values.squeeze(-1)
    if target_q_values.dim() == 3:  # [T, B, 1] -> [T, B]
        target_q_values = target_q_values.squeeze(-1)
    if rewards.dim() == 3:  # [T, B, 1] -> [T, B]
        rewards = rewards.squeeze(-1)
    if dones.dim() == 3:  # [T, B, 1] -> [T, B]
        dones = dones.squeeze(-1)
    assert predicted_q_values.dim() == rewards.dim() == dones.dim(
    ) == 2, "Expect [T, B] tensors"
    assert predicted_q_values.shape == rewards.shape == dones.shape, "Shape mismatch among predicted_q_values, rewards, dones"
    device = predicted_q_values.device
    gamma = gamma * torch.ones_like(rewards, device=device)

    if mask is None:
        mask = torch.ones_like(rewards, device=device)
    else:
        assert mask.shape == rewards.shape, "mask must have shape [T, B, 1]"

    # User-requested implementation:
    # 1) Pad rewards, values, discounts, dones with n values (right-pad on time)
    # 2) Compute TD-n targets via a clean backward horizon loop

    with torch.no_grad():
        targets = torch.vmap(get_targets,
                             in_dims=(1, 1, 1, 1, None, None),
                             out_dims=(1, ))(target_q_values, rewards, dones, gamma,
                                             n, lambda_)

    # TD error and masked loss
    td_error = predicted_q_values - targets
    if reduction == 'mean':
        # denom = mask.sum().clamp_min(1.0)
        # loss = (td_error.pow(2) * mask).sum() / denom
        loss = (td_error.pow(2) * mask).mean()
    elif reduction == 'sum':
        loss = (td_error.pow(2) * mask).sum()
    else:
        raise ValueError("Unsupported reduction: %s" % reduction)

    info = {
        'td_n_loss': loss.item(),
        'q_mean': predicted_q_values.mean().item(),
        'target_mean': targets.mean().item(),
        'target_std': targets.std(dim=1).mean().item(),
        'target_max': targets.max(dim=1).values.mean().item(),
        'target_min': targets.min(dim=1).values.mean().item(),
        'td_error_abs_mean': td_error.abs().mean().item(),
    }
    return loss, info


def test_td_n_loss():
    import numpy as np
    import rlax
    import jax
    q_values = np.random.randn(10, 32)
    rewards = np.random.randn(10, 32)
    dones = np.zeros((10, 32))
    gamma = 0.99 * np.ones((10, 32))
    n = 10
    lambda_ = 1.0
    mask = np.ones((10, 32))
    target_torch = torch.vmap(get_targets, in_dims=(1, 1, 1, 1, None, None), out_dims=(1,))(torch.from_numpy(q_values),
                           torch.from_numpy(rewards),
                           torch.from_numpy(dones),
                           torch.from_numpy(gamma), n, lambda_)
    # print()
    rlax_fn = jax.vmap(rlax.n_step_bootstrapped_returns, in_axes=(1, 1, 1, None, None), out_axes=1)
    target_jax = rlax_fn(rewards, (1 - dones) * gamma, q_values, n, lambda_)
    print(np.allclose(target_torch.numpy(), np.array(target_jax)))

if __name__ == '__main__':
    test_td_n_loss()
