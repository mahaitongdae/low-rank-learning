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
from utils import TransitionDataset, LabeledTransitionDataset

# from evaluation import evaluate


@hydra.main(config_path='../config', config_name='lrl_dataset')
def run(args):

    ### set file path
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # log_dir = os.path.join(root_dir, 'log')
    # alg_dir = os.path.join(log_dir, f'{args.dynamics}/{args.estimator}')
    # exp_dir = os.path.join(alg_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    # os.makedirs(exp_dir, exist_ok=True)
    # exp_dir = Path(HydraConfig.get().run.dir)
    summary_writer = SummaryWriter(os.getcwd())
    log_git_details(log_file=os.path.join(os.getcwd(), 'git.diff'))
    dataset = minari.load_dataset(args.dataset_name, download=True)
    env = dataset.recover_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ### initial training

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, shuffle_trajectories=False),
        num_workers=4,
        pin_memory=True  # Set to True if you are training on a CUDA GPU
    )
    if args.estimator.name == 'random_feature':
        estimator = LearnableFRandomFeatureEstimator(
            hidden_dim=args.hidden_dim,
            hidden_depth=args.hidden_depth,
            state_dim=state_dim,
            action_dim=action_dim,
            mc_dim=args.feature_dim,
            device=args.device)
        qnet = RandomFeatureQNet(state_dim=state_dim,
                                 action_dim=action_dim,
                                 hidden_dim=args.hidden_dim,
                                 hidden_depth=args.hidden_depth,
                                 mc_dim=args.feature_dim,
                                 device=args.device)
    else:
        raise NotImplementedError

    # dataloader = zip(
    #     train_dataloader,
    #     neg_dataloader) if 'contrastive' in args.dynamics else train_dataloader
    if task == 'train_qnet_only':
        assert args.pretrained_representation_path is not None
        qnet.load_pretrained_reprsentation(args.pretrained_representation_path)
    global_step = 0
    max_batches = getattr(args, 'train_batches_per_epoch', None)
    for epoch in range(args.train_epochs):
        pbar = tqdm(train_dataloader,
                    desc=f'Epoch {epoch + 1}/{args.train_epochs}')
        for batch, transition in enumerate(pbar):
            state = transition['observations'].reshape(
                -1, state_dim).float().to(args.device)
            action = transition['actions'].reshape(-1, action_dim).float().to(
                args.device)
            reward = transition['rewards'].reshape(-1,
                                                   1).float().to(args.device)
            s_tp1 = transition['next_observations'].reshape(
                -1, state_dim).float().to(args.device)
            # Train the estimator
            if args.task == 'train_estimator_only':
                info = estimator.train(state, action, reward, s_tp1)
            elif args.task == 'train_qnet_only':
                info = qnet.train(state, action, reward, s_tp1)
            else:
                raise NotImplementedError
            global_step += 1
            for key, value in info.items():
                if 'dist' in key:
                    key = 'dist/' + key
                    summary_writer.add_histogram(key, value, global_step)
                else:
                    key = 'train/' + key
                    summary_writer.add_scalar(key, value, global_step)
            summary_writer.flush()
            pbar.set_postfix(loss=info.get('est_loss'))
            if max_batches is not None and (batch + 1) >= max_batches:
                break

    estimator.save(os.getcwd())


if __name__ == '__main__':

    run()
