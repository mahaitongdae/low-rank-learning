import os
import json
import math
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import minari
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from agents.estimator.estimation_loss import td_n_loss
from agents.policies import DiagGaussianPolicy, build_mlp
from exp_logger.log_git import log_git_details
from exp_logger.logger import TensorboardOrWandBLogger
from utilities.dataset_utils import (
    collate_fn,
    compute_stats_over_episodes,
    create_normalizers_from_stats,
    normalize_batch_dict,
    ConcatMinariDataset,
    RandomInterleavedConcatSampler,
)


def process_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith('mujoco_'):
        return dataset_name.replace('_', '/') + '-v0'
    else:
        return dataset_name


class SimpleQNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 hidden_depth: int,
                 device: torch.device,
                 lr: float = 3e-4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.trunk = build_mlp(state_dim + action_dim, hidden_dim, 1, hidden_depth).to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.trunk(x)

    def forward_time_major(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states/actions: [T, B, *]
        T, B, _ = states.shape
        states_flat = states.reshape(T * B, -1)
        actions_flat = actions.reshape(T * B, -1)
        q_flat = self.forward(states_flat, actions_flat)
        return q_flat.reshape(T, B, 1)

    @torch.no_grad()
    def _target_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward_time_major(states, actions)

    def train_td_n(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   gamma: float,
                   n: int,
                   dones: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None,
                   reduction: str = 'mean') -> dict:
        if dones is None:
            dones = torch.zeros_like(rewards)
        pred_q = self.forward_time_major(states[:-1], actions[:-1])
        with torch.no_grad():
            tgt_q = self._target_values(states[1:], actions[1:])
        loss, info = td_n_loss(
            target_q_values=tgt_q,
            predicted_q_values=pred_q,
            rewards=rewards[:-1],
            dones=dones[:-1],
            gamma=gamma,
            n=n,
            mask=mask[:-1] if mask is not None else None,
            reduction=reduction,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return info

    def save(self, path: str, iter: Optional[str] = None) -> None:
        fname = f'qnet_{iter}.pth' if iter is not None else 'qnet.pth'
        torch.save(self.state_dict(), os.path.join(path, fname))


@hydra.main(config_path='../config', config_name='lrl_offline_baseline')
def run(args: DictConfig):
    # Logger setup
    if args.logger == 'wandb':
        run = wandb.init(
            project="low_rank_learning",
            name=HydraConfig.get().run.dir.split('/')[-1],
            config=OmegaConf.to_container(args, resolve=True),
            group='offline_baseline_q' if not getattr(args, 'train_policy', False) else 'offline_baseline_q_policy',
        )
        logger = TensorboardOrWandBLogger(
            logger='wandb', logger_instance=run, log_interval=100
        )
    else:
        summary_writer = SummaryWriter(os.getcwd())
        logger = TensorboardOrWandBLogger(
            logger='tensorboard', logger_instance=summary_writer, log_interval=100
        )
    log_git_details(log_file=os.path.join(os.getcwd(), 'git.diff'))

    # Dataset loading
    dataset_names = getattr(args, 'dataset_names', None)
    if dataset_names and len(dataset_names) > 0:
        loaded_datasets = [minari.load_dataset(process_dataset_name(name), download=True)
                           for name in dataset_names]
        dataset = ConcatMinariDataset(loaded_datasets)
        env_source = loaded_datasets[0]
    else:
        print(f"dataset_name: {process_dataset_name(args.dataset_name)}")
        env_source = minari.load_dataset(process_dataset_name(args.dataset_name), download=True)
        dataset = env_source

    env = env_source.recover_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Dataloader
    if isinstance(dataset, ConcatMinariDataset):
        sampler = RandomInterleavedConcatSampler(dataset, seed=args.seed)
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=partial(collate_fn, shuffle_trajectories=True),
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, shuffle_trajectories=True),
            num_workers=4,
            pin_memory=True,
        )

    # Normalization stats
    stats = compute_stats_over_episodes(dataset, keys=("observations",), num_episodes=5)
    normalizers = create_normalizers_from_stats(stats)

    try:
        torch.save(stats, os.path.join(os.getcwd(), 'normalizer_stats.pth'))
        if 'observations' in normalizers:
            obs_shift, obs_scale = normalizers['observations'].shift_scale()
            obs_norm = {'shift': obs_shift.tolist(), 'scale': obs_scale.tolist()}
            with open(os.path.join(os.getcwd(), 'obs_normalizer.json'), 'w') as f:
                json.dump(obs_norm, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save normalizer stats: {e}")

    # Q-network (simple MLP)
    qnet = SimpleQNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        device=torch.device(args.device),
        lr=args.lr,
    )

    # Optional policy training
    policy = None
    actor_optimizer = None
    if getattr(args, 'train_policy', False):
        policy = DiagGaussianPolicy(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=getattr(args, 'policy_hidden_dim', args.hidden_dim),
            hidden_depth=getattr(args, 'policy_hidden_depth', max(1, args.hidden_depth // 2)),
            log_std_bounds=tuple(getattr(args, 'log_std_bounds', [-5.0, 2.0]))
        ).to(args.device)
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=getattr(args, 'actor_lr', args.lr))

    # Training loop
    global_step = 0
    max_batches = getattr(args, 'train_batches_per_epoch', None)
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.train_epochs}')
        for batch, transition in enumerate(pbar):
            # Normalize observations
            if getattr(args, 'normalize_data', True):
                transition = normalize_batch_dict(
                    transition, normalizers, keys=("observations", "next_observations")
                )
            state = transition['observations'].float().to(args.device)
            action = transition['actions'].float().to(args.device)
            reward = (transition['rewards'] - 6.0).float().to(args.device)  # temporary reward shift
            # TD-n critic update, optionally with CQL regularization
            use_cql = hasattr(args, 'cql') and getattr(args.cql, 'enabled', False)
            if not use_cql:
                info = qnet.train_td_n(state, action, reward, gamma=0.99, n=args.td_n_horizon)
                loss_to_log = info.get('td_n_loss')
            else:
                # Compute TD-n loss (no step here)
                pred_q = qnet.forward_time_major(state[:-1], action[:-1])
                with torch.no_grad():
                    tgt_q = qnet._target_values(state[1:], action[1:])
                td_loss, info = td_n_loss(
                    target_q_values=tgt_q,
                    predicted_q_values=pred_q,
                    rewards=reward[:-1],
                    dones=torch.zeros_like(reward[:-1]),
                    gamma=0.99,
                    n=args.td_n_horizon,
                    reduction='mean',
                )

                # CQL loss: alpha * ( E_s[ logsumexp_a Q(s,a) - logK ] - E_{(s,a)~D}[Q(s,a)] )
                alpha = torch.as_tensor(getattr(args.cql, 'alpha', 1.0), device=state.device, dtype=pred_q.dtype)
                temperature = torch.as_tensor(getattr(args.cql, 'temperature', 1.0), device=state.device, dtype=pred_q.dtype)
                num_uniform = int(getattr(args.cql, 'num_uniform_actions', 10))
                num_policy = int(getattr(args.cql, 'num_policy_actions', 10))
                subtract_log_num = bool(getattr(args.cql, 'subtract_log_num_actions', True))

                # Use states at time t to evaluate Q(s_t, .)
                states_tm = state[:-1]  # [T-1, B, S]
                Tm, B, Sdim = states_tm.shape
                states_flat = states_tm.reshape(Tm * B, Sdim)

                # Dataset actions at time t
                actions_data = action[:-1].reshape(Tm * B, -1)
                q_data = qnet.forward(states_flat, actions_data).reshape(Tm * B, 1)

                q_samples_list = []
                # Uniform samples in [-1, 1]
                if num_uniform > 0:
                    a_uniform = torch.rand(Tm * B, num_uniform, action.shape[-1], device=state.device, dtype=state.dtype) * 2.0 - 1.0
                    s_uniform = states_flat.unsqueeze(1).expand(-1, num_uniform, -1).reshape(Tm * B * num_uniform, Sdim)
                    a_uniform_flat = a_uniform.reshape(Tm * B * num_uniform, -1)
                    q_uniform = qnet.forward(s_uniform, a_uniform_flat).reshape(Tm * B, num_uniform, 1)
                    q_samples_list.append(q_uniform)

                # Policy samples if available
                if num_policy > 0:
                    if policy is not None:
                        # Sample num_policy actions per state
                        dist = policy(states_flat)
                        a_pol = []
                        for _ in range(num_policy):
                            a_pol.append(dist.rsample())
                        a_pol = torch.stack(a_pol, dim=1)  # [N, K, A]
                    else:
                        # Fallback: Gaussian then tanh to [-1,1]
                        a_pol = torch.tanh(torch.randn(Tm * B, num_policy, action.shape[-1], device=state.device, dtype=state.dtype))
                    s_pol = states_flat.unsqueeze(1).expand(-1, num_policy, -1).reshape(Tm * B * num_policy, Sdim)
                    a_pol_flat = a_pol.reshape(Tm * B * num_policy, -1)
                    q_pol = qnet.forward(s_pol, a_pol_flat).reshape(Tm * B, num_policy, 1)
                    q_samples_list.append(q_pol)

                if len(q_samples_list) > 0:
                    q_samples = torch.cat(q_samples_list, dim=1)  # [N, K, 1]
                    K = q_samples.shape[1]
                    # temperature-scaled logsumexp across sampled actions
                    lse = torch.logsumexp(q_samples.squeeze(-1) / temperature, dim=1, keepdim=True) * temperature
                    if subtract_log_num and K > 0:
                        lse = lse - math.log(K)
                    cql_loss = (lse.mean() - q_data.mean()) * alpha
                else:
                    cql_loss = torch.zeros((), device=state.device, dtype=pred_q.dtype)

                total_loss = td_loss + cql_loss
                qnet.optimizer.zero_grad()
                total_loss.backward()
                qnet.optimizer.step()

                info.update({
                    'cql_loss': cql_loss.item() if cql_loss.numel() == 1 else float(cql_loss.mean().item()),
                    'total_loss': total_loss.item(),
                })
                loss_to_log = info.get('total_loss')

            global_step += 1
            logger.log(info, global_step, group='train')
            pbar.set_postfix(loss=loss_to_log)

            # Optional actor update (maximize Q + entropy)
            if getattr(args, 'train_policy', False) and policy is not None:
                T, B, _ = state.shape
                states_tm = state[:-1]
                states_flat = states_tm.reshape(-1, state_dim)
                dist = policy(states_flat)
                actions_sample = dist.rsample()
                log_prob = dist.log_prob(actions_sample).sum(dim=-1, keepdim=True)
                for p in qnet.parameters():
                    p.requires_grad = False
                q_values = qnet.forward(states_flat, actions_sample)
                alpha = torch.as_tensor(getattr(args, 'policy_alpha', 0.2), device=q_values.device, dtype=q_values.dtype)
                actor_loss = (alpha * log_prob - q_values).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                for p in qnet.parameters():
                    p.requires_grad = True
                logger.log({'actor_loss': actor_loss.item(), 'actor_entropy': (-log_prob).mean().item()}, global_step, group='train')

            if max_batches is not None and (batch + 1) >= max_batches:
                break

        # Save checkpoints
        qnet.save(os.getcwd(), iter=str(epoch))
        if getattr(args, 'train_policy', False) and policy is not None:
            torch.save(policy.state_dict(), os.path.join(os.getcwd(), f'policy_{epoch}.pth'))


if __name__ == '__main__':
    run()
