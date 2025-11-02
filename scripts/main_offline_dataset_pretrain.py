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

from agents.estimator.random_feature import LearnableFRandomFeatureEstimator
from agents.estimator.random_feature import RandomFeatureQNet
from datetime import datetime
from exp_logger.log_git import log_git_details
from functools import partial
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities.dataset_utils import collate_fn
from utilities.dataset_utils import compute_stats_over_episodes
from utilities.dataset_utils import create_normalizers_from_stats
from utilities.dataset_utils import normalize_batch_dict
from utilities.dataset_utils import ConcatMinariDataset
from utilities.dataset_utils import RandomInterleavedConcatSampler
from utils import TransitionDataset, LabeledTransitionDataset
from exp_logger.logger import TensorboardOrWandBLogger
import wandb
# from evaluation import evaluate


@hydra.main(config_path='../config', config_name='lrl_train_repr')
def run(args):

    if args.logger == 'wandb':
        run = wandb.init(
            project="low_rank_learning",
            name="train_repr_" + args.suffix,
            config=OmegaConf.to_container(args, resolve=True),
            group='train_repr',
        )
        logger = TensorboardOrWandBLogger(logger='wandb', logger_instance=run)
    else:
        summary_writer = SummaryWriter(os.getcwd())
        logger = TensorboardOrWandBLogger(logger='tensorboard', logger_instance=summary_writer)
    log_git_details(log_file=os.path.join(os.getcwd(), 'git.diff'))

    dataset_names = getattr(args, 'dataset_names', None)
    if dataset_names and len(dataset_names) > 0:
        loaded_datasets = [
            minari.load_dataset(name, download=True) for name in dataset_names
        ]
        dataset = ConcatMinariDataset(loaded_datasets)
        env_source = loaded_datasets[0]
    else:
        env_source = minari.load_dataset(args.dataset_name, download=True)
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
            device=args.device,
            dt=args.estimator.dt,
            learning_rate=args.estimator.lr)
    else:
        raise NotImplementedError

    # dataloader = zip(
    #     train_dataloader,
    #     neg_dataloader) if 'contrastive' in args.dynamics else train_dataloader

    global_step = 0
    max_batches = getattr(args, 'train_batches_per_epoch', None)
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_dataloader,
                    desc=f'Epoch {epoch + 1}/{args.train_epochs}')
        for batch, transition in enumerate(pbar):

            # Train the estimator
            assert args.task == 'train_estimator_only'
            if args.normalize_data:
                transition = normalize_batch_dict(
                    transition,
                    normalizers,
                    keys=("observations", "next_observations"))
            # Flat the whole batch and train the estimator
            state = transition['observations'].reshape(
                -1, state_dim).float().to(args.device)
            action = transition['actions'].reshape(
                -1, action_dim).float().to(args.device)
            reward = transition['rewards'].reshape(-1, 1).float().to(
                args.device)
            s_tp1 = transition['next_observations'].reshape(
                -1, state_dim).float().to(args.device)
            info = estimator.train_(state, action, reward, s_tp1)
            global_step += 1
            logger.log(info, global_step, group='train')
            pbar.set_postfix(loss=info.get('est_loss'))
            if max_batches is not None and (batch + 1) >= max_batches:
                break

        # Save estimator and normalization stats (for later reuse)
        estimator.save(os.getcwd(), iter=str(epoch))


if __name__ == '__main__':

    run()
