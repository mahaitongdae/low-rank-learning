"""
This script is used to pretrain the representations from a offline dataset.
"""

import argparse
import hydra
import json
import minari
import numpy as np
import os
import torch
import yaml
import math

from agents.estimator.random_feature import LearnableFRandomFeatureEstimator
from agents.estimator.random_feature import RandomFeatureQNet
from datetime import datetime
from exp_logger.log_git import log_git_details
from exp_logger.logger import TensorboardOrWandBLogger
from functools import partial
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tensorboardX import SummaryWriter
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities.dataset_utils import collate_fn
from utilities.dataset_utils import compute_stats_over_episodes
from utilities.dataset_utils import create_normalizers_from_stats
from utilities.dataset_utils import normalize_batch_dict
from utilities.dataset_utils import ConcatMinariDataset
from utilities.dataset_utils import RandomInterleavedConcatSampler

# Lightweight squashed Gaussian policy (self-contained)
from torch import nn
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - nn.functional.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_depth: int) -> nn.Sequential:
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    layers = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class DiagGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, hidden_depth: int, log_std_bounds: tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.trunk = build_mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

    def forward(self, obs: torch.Tensor) -> SquashedNormal:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)

# from evaluation import evaluate

def process_dataset_name(dataset_name: str) -> str:
    if dataset_name.startswith('mujoco_'):
        return dataset_name.replace('_', '/') + '-v0'
    else:
        return dataset_name


@hydra.main(config_path='../config', config_name='lrl_train_q')
def run(args):

    if args.logger == 'wandb':
        run = wandb.init(
            project="low_rank_learning",
            name=HydraConfig.get().run.dir.split('/')[-1],
            config=OmegaConf.to_container(args, resolve=True),
            group='train_qnet_only' if not getattr(args, 'train_policy', False) else 'train_qnet_and_policy',
        )
        logger = TensorboardOrWandBLogger(logger='wandb',
                                          logger_instance=run,
                                          log_interval=100)
    else:
        summary_writer = SummaryWriter(os.getcwd())
        logger = TensorboardOrWandBLogger(logger='tensorboard',
                                          logger_instance=summary_writer,
                                          log_interval=100)
    log_git_details(log_file=os.path.join(os.getcwd(), 'git.diff'))

    dataset_names = getattr(args, 'dataset_names', None)
    if dataset_names and len(dataset_names) > 0:
        loaded_datasets = [
            minari.load_dataset(process_dataset_name(name), download=True)
            for name in dataset_names
        ]
        dataset = ConcatMinariDataset(loaded_datasets)
        env_source = loaded_datasets[0]
    else:
        print(f"dataset_name: {process_dataset_name(args.dataset_name)}")
        env_source = minari.load_dataset(process_dataset_name(
            args.dataset_name),
                                         download=True)
        dataset = env_source

    env = env_source.recover_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ### initial training

    if isinstance(dataset, ConcatMinariDataset):
        sampler = RandomInterleavedConcatSampler(dataset, seed=args.seed)
        train_dataloader = DataLoader(dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=False,
                                      sampler=sampler,
                                      collate_fn=partial(
                                          collate_fn,
                                          shuffle_trajectories=True),
                                      num_workers=4,
                                      pin_memory=True)
    else:
        train_dataloader = DataLoader(dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      collate_fn=partial(
                                          collate_fn,
                                          shuffle_trajectories=True),
                                      num_workers=4,
                                      pin_memory=True)

    # normalize the data
    stats = compute_stats_over_episodes(dataset,
                                        keys=("observations", ),
                                        num_episodes=5)
    normalizers = create_normalizers_from_stats(stats)

    try:
        # Persist raw stats and a light normalizer snapshot
        torch.save(stats, os.path.join(os.getcwd(), 'normalizer_stats.pth'))
        # For interpretability, save observation normalizer as JSON
        if 'observations' in normalizers:
            obs_shift, obs_scale = normalizers['observations'].shift_scale()
            obs_norm = {
                'shift': obs_shift.tolist(),
                'scale': obs_scale.tolist()
            }
            with open(os.path.join(os.getcwd(), 'obs_normalizer.json'),
                      'w') as f:
                json.dump(obs_norm, f, indent=2)
    except Exception as e:
        # Non-fatal: training succeeded even if saving stats failed
        print(f"Warning: failed to save normalizer stats: {e}")

    if args.estimator.name == 'random_feature':
        estimator = LearnableFRandomFeatureEstimator(
            hidden_dim=args.hidden_dim,
            hidden_depth=args.hidden_depth,
            state_dim=state_dim,
            action_dim=action_dim,
            mc_dim=args.feature_dim,
            device=args.device,
            dt=args.estimator.dt,
            learning_rate=args.estimator.lr)
    else:
        raise NotImplementedError

    # dataloader = zip(
    #     train_dataloader,
    #     neg_dataloader) if 'contrastive' in args.dynamics else train_dataloader
    if args.task == 'train_qnet_only':
        assert args.pretrained_representation_path is not None
        print(args.pretrained_representation_path)
        with open(
                os.path.join(args.pretrained_representation_path,
                             '.hydra/config.yaml'), 'r') as f:
            saved_args = yaml.safe_load(f)
        qnet = RandomFeatureQNet(state_dim=state_dim,
                                 action_dim=action_dim,
                                 hidden_dim=args.hidden_dim,
                                 hidden_depth=args.hidden_depth,
                                 mc_dim=args.feature_dim,
                                 device=args.device)

        qnet.load_pretrained_reprsentation(args.pretrained_representation_path,
                                           saved_args,
                                           epoch=9)
        # Optional policy to maximize learned Q
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
    global_step = 0
    max_batches = getattr(args, 'train_batches_per_epoch', None)
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_dataloader,
                    desc=f'Epoch {epoch + 1}/{args.train_epochs}')
        for batch, transition in enumerate(pbar):

            # Train the estimator
            assert args.task == 'train_qnet_only'
            if qnet.normalizer_stats:
                normalizers = create_normalizers_from_stats(
                    qnet.normalizer_stats)
                transition = normalize_batch_dict(transition,
                                                  normalizers,
                                                  keys=("observations",
                                                        "next_observations"))
            state = transition['observations'].float().to(args.device)
            action = transition['actions'].float().to(args.device)
            # reward = torch.nn.functional.sigmoid(
            #     transition['rewards']).float().to(args.device)
            reward = (transition['rewards'] - 6.0).float().to(
                args.device)  # a temporary fix for the reward normalization
            # s_tp1 = transition['next_observations'].float().to(args.device)
            info = qnet.train_td_n(state,
                                   action,
                                   reward,
                                   gamma=0.99,
                                   n=args.td_n_horizon)
            global_step += 1
            logger.log(info, global_step, group='train')
            pbar.set_postfix(loss=info.get('td_n_loss'))

            # Optional: policy improvement step maximizing Q(s,a) + alpha * H
            if getattr(args, 'train_policy', False) and policy is not None:
                T, B, _ = state.shape
                states_tm = state[:-1]  # match TD target horizon
                states_flat = states_tm.reshape(-1, state_dim)
                dist = policy(states_flat)
                actions_sample = dist.rsample()
                log_prob = dist.log_prob(actions_sample).sum(dim=-1, keepdim=True)
                # Freeze critic params during actor update while keeping dQ/da
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

                logger.log({
                    'actor_loss': actor_loss.item(),
                    'actor_entropy': (-log_prob).mean().item()
                }, global_step, group='train')
            if max_batches is not None and (batch + 1) >= max_batches:
                break

        # Save the Q-network
        qnet.save(os.getcwd(), iter=str(epoch))
        # Save policy if trained
        if getattr(args, 'train_policy', False) and policy is not None:
            torch.save(policy.state_dict(), os.path.join(os.getcwd(), f'policy_{epoch}.pth'))


if __name__ == '__main__':
    run()
