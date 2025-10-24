import torch
import torch.nn.functional as F


def td_n_loss(q_values: torch.Tensor,
              rewards: torch.Tensor,
              dones: torch.Tensor,
              gamma: float,
              n: int,
              mask: torch.Tensor | None = None,
              reduction: str = 'mean') -> tuple[torch.Tensor, dict]:
    """
    Compute TD-n loss for time-major tensors.

    Parameters
    ----------
    q_values : torch.Tensor
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
    assert q_values.dim() == rewards.dim() == dones.dim(
    ) == 2, "Expect [T, B,] tensors"
    assert q_values.shape == rewards.shape == dones.shape, "Shape mismatch among q_values, rewards, dones"
    T, B = q_values.shape
    device = q_values.device

    if mask is None:
        mask = torch.ones_like(rewards, device=device)
    else:
        assert mask.shape == rewards.shape, "mask must have shape [T, B, 1]"

    # User-requested implementation:
    # 1) Pad rewards, values, discounts, dones with n values (right-pad on time)
    # 2) Compute TD-n targets via a clean backward horizon loop

    def get_targets(q_values, rewards, dones, gamma, n):
        # discounts per step: gamma * (1 - done)
        discounts = gamma * (1.0 - dones)

        # Right pad time with n zeros so slicing k:k+T works for k in [0, n]
        pad_size = min(n - 1, T)
        targets = torch.cat([q_values[-pad_size:], rewards], dim=0)

        # Start from bootstrap term q_{t+n}
        targets = q_pad[n:n + T].clone()
        # Backward accumulate n rewards: r_{t+k} + discount_{t+k} * (...)
        for k in range(n - 1, -1, -1):
            r_slice = rewards_pad[k:k + T]
            disc_slice = discounts_pad[k:k + T]
            targets = r_slice + disc_slice * targets
        return targets
    targets = torch.vmap(get_targets, in)(q_values, rewards, dones, gamma, n)

    # TD error and masked loss
    td_error = q_values - targets
    if reduction == 'mean':
        denom = mask.sum().clamp_min(1.0)
        loss = (td_error.pow(2) * mask).sum() / denom
    elif reduction == 'sum':
        loss = (td_error.pow(2) * mask).sum()
    else:
        raise ValueError("Unsupported reduction: %s" % reduction)

    info = {
        'td_n_loss': loss.item(),
        'q_mean': q_values.mean().item(),
        'target_mean': targets.mean().item(),
        'td_error_abs_mean': td_error.abs().mean().item(),
    }
    return loss, info


def test_td_n_loss():
    q_values = torch.randn(10, 32, 1)
    rewards = torch.randn(10, 32, 1)
    dones = torch.randint(0, 2, (10, 32, 1))
    gamma = 0.99
    n = 10
    mask = torch.ones(10, 32, 1)
    loss, info = td_n_loss(q_values, rewards, dones, gamma, n, mask)
    print(loss)
    print(info)

if __name__ == '__main__':
    test_td_n_loss()
