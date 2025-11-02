"""
EBM Estimator.

"""

from typing import Optional, Union
import os
import json

import gymnasium
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
                 regularizer: float = 1e-3,
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
        self.reward_trunk = mlp_elu(state_dim + action_dim, hidden_dim, 1, hidden_depth,
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
        self.regularizer = regularizer
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
        std_inv = 1. / std
        # Reparameterization to get w
        w = std_inv[:, None, :] * self.epsilon[
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
            'delta_mean': torch.mean(delta,
                                     dim=-1).mean(),  # mean of the output
            'std_mean': torch.mean(std, dim=-1).mean(),  # mean of the std
            'delta_std': torch.std(delta, dim=-1).mean(),  # std of the output
            'std_std': torch.std(std, dim=-1).mean(),  # std of the std
            'prediction_error_mean':
            torch.mean(diff, dim=-1).mean(),  # mean of the prediction error
            'prediction_error_std':
            torch.std(diff, dim=-1).mean(),  # std of the prediction error
        }

    def get_loss(self, state: torch.Tensor, action: torch.Tensor,
                 s_tp1: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Get the loss of the transition.
        """
        delta, std = self.get_delta_and_std(state, action)
        mu = state + self.dt * delta
        diff = s_tp1 - mu
        per_dim = -0.5 * (np.log(2 * np.pi) + 2.0 * torch.log(std) +
                          (diff / std)**2)
        loss = -torch.mean(per_dim)
        regularization = self.regularizer * (torch.linalg.norm(
            delta, dim=-1).mean() + torch.linalg.norm(std, dim=-1).mean())
        info = {
            'delta_mean': torch.mean(delta,
                                     dim=-1).mean(),  # mean of the output
            'std_mean': torch.mean(std, dim=-1).mean(),  # mean of the std
            'delta_std': torch.std(delta, dim=-1).mean(),  # std of the output
            'std_std': torch.std(std, dim=-1).mean(),  # std of the std
            'prediction_error_mean':
            torch.mean(diff, dim=-1).mean(),  # mean of the prediction error
            'prediction_error_std':
            torch.std(diff, dim=-1).mean(),  # std of the prediction error
            'est_loss': loss.item(),
            'regularization': regularization.item()
        }
        return loss + regularization, info

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
        reward_pred = self.reward_trunk(torch.cat([state, action], dim=-1))
        loss = torch.nn.MSELoss()(reward_pred, reward)
        self.reward_trunk_optimizer.zero_grad()
        loss.backward()
        self.reward_trunk_optimizer.step()
        return {'reward_loss': loss.item()}

    def train_(self, state: torch.Tensor, action: torch.Tensor,
              reward: torch.Tensor, s_tp1: torch.Tensor) -> dict:
        """
        Train the estimator.
        """
        info_estimate = self.estimate(state, action, s_tp1)
        info_reward = self.fit_reward(state, action, reward)
        info = {**info_estimate, **info_reward}
        return info

    def save(self, path: str, iter: int | None = None):
        """
        Save the estimator.
        """
        fname = f'estimator_{iter}.pth' if iter is not None else 'estimator.pth'
        torch.save(self.state_dict(), os.path.join(path, fname))

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
        self.rf = LearnableFRandomFeatureEstimator(state_dim=state_dim,
                                                   action_dim=action_dim,
                                                   hidden_dim=hidden_dim,
                                                   hidden_depth=hidden_depth,
                                                   mc_dim=mc_dim,
                                                   device=device,
                                                   dt=0.05)
        # Linear head from random features to scalar Q
        self.q_head = nn.Linear(mc_dim, 1)
        self.apply(weight_init)
        self.to(device)
        self.optimizer = torch.optim.Adam(self.q_head.parameters(), lr=1e-3)

    def forward(self, state, action):
        """
        Compute Q(s, a) for batch inputs.
        state: [B, state_dim]
        action: [B, action_dim]
        returns: [B, 1]
        """
        # x = torch.cat([state, action], dim=-1)
        phi, _ = self.rf(state, action)
        q = self.rf.reward_trunk(torch.cat([state, action],
                                           dim=-1)) + self.q_head(phi)
        return q

    def forward_time_major(self, states: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s_t, a_t) for time-major inputs.
        states: [T, B, state_dim]
        actions: [T, B, action_dim]
        returns: [T, B, 1]
        """
        T, B, _ = states.shape
        # x = torch.cat([states, actions], dim=-1).reshape(T * B, -1)
        states = states.reshape(T * B, -1)
        actions = actions.reshape(T * B, -1)
        phi_sa, _ = self.rf(states, actions)
        q = self.q_head(phi_sa).reshape(T, B, 1)
        return q

    def load_pretrained_reprsentation(self,
                                      path: str,
                                      args: dict,
                                      epoch: int | None = None):
        """
        Load pretrained random feature parameters.
        - If `epoch` is provided, loads `estimator_{epoch}.pth`.
        - If `epoch` is None, finds the latest `estimator_*.pth` by max epoch.
          Falls back to `estimator.pth` if no numbered checkpoints are found.
        """
        rf_path = None
        if epoch is not None:
            candidate = os.path.join(path, f'estimator_{epoch}.pth')
            if os.path.exists(candidate):
                rf_path = candidate
        else:
            # discover the latest estimator_*.pth by numeric suffix
            try:
                files = os.listdir(path)
            except FileNotFoundError:
                files = []
            epochs = []
            prefix = 'estimator_'
            suffix = '.pth'
            for fname in files:
                if fname.startswith(prefix) and fname.endswith(suffix):
                    num_str = fname[len(prefix):-len(suffix)]
                    if num_str.isdigit():
                        epochs.append(int(num_str))
            if len(epochs) > 0:
                latest = max(epochs)
                candidate = os.path.join(path, f'estimator_{latest}.pth')
                if os.path.exists(candidate):
                    rf_path = candidate
            # fallback to plain estimator.pth
            if rf_path is None:
                candidate = os.path.join(path, 'estimator.pth')
                if os.path.exists(candidate):
                    rf_path = candidate

        if rf_path is None:
            raise FileNotFoundError(
                f"No pretrained representation found in {path}. "
                f"Expected 'estimator_<epoch>.pth' or 'estimator.pth'.")

        pretrained_state_dict = torch.load(rf_path, map_location='cpu')
        if 'epsilon' in pretrained_state_dict.keys():
            pretrained_state_dict.pop('epsilon')
            print(f"[Warning]: poping epsilong out, since it is not trained in the pretrained representation.")
        self.rf.load_state_dict(pretrained_state_dict, strict=False)
        self.rf.dt = args['estimator']['dt']
        # Optionally load normalizer stats saved by pretraining
        stats_path = os.path.join(path, 'normalizer_stats.pth')
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, map_location='cpu')
                # lightweight attach for downstream usage
                self.normalizer_stats = stats
            except Exception as e:
                print(f"Warning: failed to load normalizer_stats.pth: {e}")
        # # Prefer JSON for better interpretability; fallback to legacy .pth
        # obs_norm_json = os.path.join(path, 'obs_normalizer.json')
        # if os.path.exists(obs_norm_json):
        #     try:
        #         with open(obs_norm_json, 'r') as f:
        #             self.obs_normalizer = json.load(f)  # {'shift': [...], 'scale': [...]}
        #     except Exception as e:
        #         print(f"Warning: failed to load obs_normalizer.json: {e}")
        # else:
        #     obs_norm_path = os.path.join(path, 'obs_normalizer.pth')
        #     if os.path.exists(obs_norm_path):
        #         try:
        #             obs_norm = torch.load(obs_norm_path, map_location='cpu')
        #             self.obs_normalizer = obs_norm
        #         except Exception as e:
        #             print(f"Warning: failed to load obs_normalizer.pth: {e}")

    def train_(self, state: torch.Tensor, action: torch.Tensor,
               reward: torch.Tensor, s_tp1: torch.Tensor) -> dict:
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
                   gamma: float,
                   n: int,
                   dones: torch.Tensor | None = None,
                   mask: torch.Tensor | None = None,
                   reduction: str = 'mean') -> dict:
        """
        Train Q-network with TD-n loss on time-major inputs.
        - states, actions, rewards, dones: [T, B, ...] with rewards/dones shaped [T, B, 1]
        Returns logging dict.
        """
        if dones is None:
            dones = torch.zeros_like(rewards)
        predicted_q_values = self.forward_time_major(states[:-1],
                                                     actions[:-1])  # [T, B, 1]
        with torch.no_grad():
            target_q_values = self.forward_time_major(states[1:],
                                                      actions[1:])  # [T, B, 1]
        loss, info = td_n_loss(target_q_values=target_q_values,
                               predicted_q_values=predicted_q_values,
                               rewards=rewards[:-1],
                               dones=dones[:-1],
                               gamma=gamma,
                               n=n,
                               reduction=reduction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return info

    def save(self, path: str, iter: int | None = None):
        """
        Save the Q-network.
        """
        fname = f'qnet_{iter}.pth' if iter is not None else 'qnet.pth'
        torch.save(self.state_dict(), os.path.join(path, fname))

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def test_get_state_dict():
    estimator = LearnableFRandomFeatureEstimator(state_dim=10,
                                                 action_dim=10,
                                                 hidden_dim=10,
                                                 hidden_depth=10,
                                                 mc_dim=10)
    state_dict = estimator.state_dict()
    print(state_dict.keys())


def test_get_random_feature():
    env = gymnasium.make('HalfCheetah-v5')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    q = RandomFeatureQNet(state_dim=state_dim,
                          action_dim=action_dim,
                          hidden_dim=256,
                          hidden_depth=2,
                          mc_dim=1024,
                          device=torch.device('cuda'))
    state = env.reset()
    action = env.action_space.sample()
    random_feature = q.rf(state, action)
    print(random_feature.shape)

def test_qnet_state_dict():
    q = RandomFeatureQNet(state_dim=10,
                          action_dim=10,
                          hidden_dim=10,
                          hidden_depth=3,
                          mc_dim=10,
                          device=torch.device('cuda'))
    state_dict = q.state_dict()
    print(state_dict.keys())

if __name__ == '__main__':
    test_qnet_state_dict()
