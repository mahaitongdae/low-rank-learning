"""
EBM Estimator.

"""

from typing import Optional, Union
import os

import torch
import torch.nn as nn
from agents.estimator.estimator import DensityEstimator
from networks.networks import MLP, NormalizedMLP, LearnableRandomFeature, randMu2
from agents.estimator.estimation_loss import td_n_loss
import numpy as np
EPS = 1e-6

torch.nn.Softplus

def mlp_elu(input_dim: int,
            hidden_dim: int,
            output_dim: int,
            hidden_depth: int,
            output_mod: Union[nn.Module, None] = None,
            output_bias: Optional[bool] = True) -> nn.Sequential:
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim, bias=output_bias))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class LearnableFRandomFeatureEstimator(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 hidden_depth: int,
                 dt: float = 0.05,
                 output_mod: Union[nn.Module, None] = None,
                 mc_dim: int = 1024,
                 learning_rate: float = 1e-3,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        # We need to save this epsilon parameters but not train it
        self.epsilon = torch.nn.Parameter(
            torch.normal(0, 1., size=(mc_dim, state_dim),
                         requires_grad=False).to(device))  # [N, n]
        self.trunk = mlp_elu(state_dim + action_dim, hidden_dim, 2 * state_dim,
                             hidden_depth, output_mod)
        self.reward_trunk = mlp_elu(state_dim, hidden_dim, 1, hidden_depth,
                                    output_mod)
        self.apply(weight_init)
        self.b = 2 * torch.pi * torch.rand(size=(mc_dim, 1)).to(device)
        self.trunk.to(device)
        self.reward_trunk.to(device)
        self.trunk_optimizer = torch.optim.Adam(self.trunk.parameters(),
                                                lr=learning_rate)
        self.reward_trunk_optimizer = torch.optim.Adam(
            self.reward_trunk.parameters(), lr=learning_rate)
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_delta_and_std(
            self, state: torch.Tensor,
            action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the delta and std of the transition.
        Parameters
        ----------
        state : torch.Tensor [B, state_dim]
        action : torch.Tensor [B, action_dim]
        Returns
        -------
        delta : torch.Tensor [B, n]
            The delta of the transition.
        std : torch.Tensor [B, n]
            The std of the transition.
        """
        assert state.shape[0] == action.shape[0]
        assert state.shape[1] == self.state_dim
        assert action.shape[1] == self.action_dim
        output = self.trunk(torch.cat([state, action], dim=-1))
        delta, std = torch.chunk(output, 2, dim=-1)  # [B, n]
        std = nn.Softplus()(std) + EPS
        return delta, std

    def forward(self, state, action, s_tp1=None):
        delta, std = self.get_delta_and_std(state, action)
        f_sa = state + delta * self.dt
        # Reparameterization to get w
        w = std[:, None, :] * self.epsilon[
            None, :, :]  # [B, 1, n] * [1, N, n] = [B, N, n]

        def get_random_feature(x):
            """
            Get the random feature of a state.
            Parameters
            ----------
            x : torch.Tensor [B, state_dim]
                The state to get the random feature of.
            Returns
            -------
            random_feature : torch.Tensor
                The random feature of the state.
            """
            wx_p_b = w @ x.unsqueeze(-1) + self.b[None, :, :]
            # [B, N, n] * [B, n, 1] + [1, N, 1] = [B, N, 1]
            return torch.cos(wx_p_b.squeeze(-1))  # [B, N]

        random_phi = get_random_feature(f_sa)  # [B, N]
        if s_tp1 is not None:
            random_mu = get_random_feature(s_tp1)  # [B, N]
        else:
            random_mu = None
        return random_phi, random_mu

    def get_log_likelihood(self, state: torch.Tensor, action: torch.Tensor,
                           s_tp1: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Get the log likelihood of the transition.
        Parameters
        ----------
        state : torch.Tensor [B, state_dim]
        action : torch.Tensor [B, action_dim]
        s_tp1 : torch.Tensor [B, state_dim]
        Returns
        -------
        log_likelihood : torch.Tensor [B]
            The log likelihood of the transition.
        """
        delta, std = self.get_delta_and_std(state, action)
        mu = state + self.dt * delta
        diff = s_tp1 - mu
        per_dim = -0.5 * (np.log(2 * np.pi) + 2.0 * torch.log(std) +
                          (diff / std)**2)
        return torch.sum(per_dim, dim=-1), {
            'delta_mean': torch.mean(delta, dim=-1).mean(),
            'std_mean': torch.mean(std, dim=-1).mean(),
            'delta_std': torch.std(delta, dim=-1).mean(),
            'std_std': torch.std(std, dim=-1).mean(),
            'prediction_error_mean': torch.mean(diff, dim=-1).mean(),
            'prediction_error_std': torch.std(diff, dim=-1).mean(),
        }

    def get_loss(self, state: torch.Tensor, action: torch.Tensor,
                 s_tp1: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Get the loss of the transition.
        """
        log_likelihood, info = self.get_log_likelihood(state, action, s_tp1)
        loss = -torch.mean(log_likelihood)
        info.update({'est_loss': loss.item()})
        return loss, info

    def estimate(self, state: torch.Tensor, action: torch.Tensor,
                 s_tp1: torch.Tensor) -> dict:
        """
        Estimate the log likelihood of the transition.
        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The batch of transitions.
        Returns
        -------
        info : dict
            The information of the estimation.
        """
        loss, info = self.get_loss(state, action, s_tp1)
        self.trunk_optimizer.zero_grad()
        loss.backward()
        self.trunk_optimizer.step()
        return info

    def fit_reward(self, state: torch.Tensor, action: torch.Tensor,
                   reward: torch.Tensor) -> dict:
        """
        Fit the reward function.
        """
        reward_pred = self.reward_trunk(state)
        loss = torch.nn.MSELoss()(reward_pred, reward)
        self.reward_trunk_optimizer.zero_grad()
        loss.backward()
        self.reward_trunk_optimizer.step()
        return {'reward_loss': loss.item()}

    def train(self, state: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, s_tp1: torch.Tensor) -> dict:
        """
        Train the estimator.
        """
        info_estimate = self.estimate(state, action, s_tp1)
        info_reward = self.fit_reward(state, action, reward)
        info = {**info_estimate, **info_reward}
        return info

    def save(self, path: str):
        """
        Save the estimator.
        """
        torch.save(self.state_dict(), os.path.join(path, 'estimator.pth'))

    def load(self, path: str):
        """
        Load the estimator.
        """
        self.load_state_dict(torch.load(os.path.join(path, 'estimator.pth')))


class RandomFeatureQNet(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, hidden_depth, mc_dim,
                 device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.mc_dim = mc_dim
        self.device = device
        # Feature extractor over concatenated [state, action]
        self.rf = LearnableRandomFeature(input_dim=state_dim + action_dim,
                                         output_dim=mc_dim,
                                         hidden_dim=hidden_dim,
                                         hidden_depth=hidden_depth,
                                         batch_size=mc_dim,
                                         sigma=1.,
                                         learnable_w=True,
                                         device=device)
        # Linear head from random features to scalar Q
        self.q_head = nn.Linear(mc_dim, 1)
        self.apply(weight_init)
        self.to(device)
        self.optimizer = torch.optim.Adam(
            list(self.rf.parameters()) + list(self.q_head.parameters()), lr=1e-3)

    def forward(self, state, action):
        """
        Compute Q(s, a) for batch inputs.
        state: [B, state_dim]
        action: [B, action_dim]
        returns: [B, 1]
        """
        x = torch.cat([state, action], dim=-1)
        phi = self.rf(x)
        q = self.q_head(phi)
        return q

    def forward_time_major(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s_t, a_t) for time-major inputs.
        states: [T, B, state_dim]
        actions: [T, B, action_dim]
        returns: [T, B, 1]
        """
        T, B, _ = states.shape
        x = torch.cat([states, actions], dim=-1).reshape(T * B, -1)
        phi = self.rf(x)
        q = self.q_head(phi).reshape(T, B, 1)
        return q

    def load_pretrained_reprsentation(self, path: str):
        """
        Load pretrained random feature parameters if available at path/rf.pth
        """
        rf_path = os.path.join(path, 'rf.pth')
        if os.path.exists(rf_path):
            self.rf.load_state_dict(torch.load(rf_path, map_location=self.device))

    def train(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, s_tp1: torch.Tensor) -> dict:
        """
        Legacy single-step regression on immediate reward (kept for compatibility).
        """
        q = self.forward(state, action)
        loss = torch.nn.MSELoss()(q, reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'qnet_loss': loss.item()}

    def train_td_n(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   dones: torch.Tensor,
                   gamma: float,
                   n: int,
                   mask: torch.Tensor | None = None,
                   reduction: str = 'mean') -> dict:
        """
        Train Q-network with TD-n loss on time-major inputs.
        - states, actions, rewards, dones: [T, B, ...] with rewards/dones shaped [T, B, 1]
        Returns logging dict.
        """
        q_values = self.forward_time_major(states, actions)  # [T, B, 1]
        loss, info = td_n_loss(q_values=q_values,
                               rewards=rewards,
                               dones=dones,
                               gamma=gamma,
                               n=n,
                               mask=mask,
                               reduction=reduction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return info

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def test_get_state_dict():
    estimator = LearnableFRandomFeatureEstimator(state_dim=10, action_dim=10, hidden_dim=10, hidden_depth=10, mc_dim=10)
    state_dict = estimator.state_dict()
    print(state_dict.keys())

if __name__ == '__main__':
    test_get_state_dict()
