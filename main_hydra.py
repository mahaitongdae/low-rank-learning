from envs.noisy_pendulum import ParallelNoisyPendulum
from envs.mvn import *
from utils import TransitionDataset, LabeledTransitionDataset
import torch
from torch.utils.data import DataLoader
from agents.estimator import MLEEstimator, NCEEstimator, SupervisedEstimator, SupervisedLearnableRandomFeatureEstimator
from agents.single_network_estimator import NCESingleNetwork
from tensorboardX import SummaryWriter
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
# from evaluation import evaluate
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from exp_logger.log_git import log_git_details


@hydra.main(config_path='./config', config_name='lrl')
def run(args):

    ### set file path
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # log_dir = os.path.join(root_dir, 'log')
    # alg_dir = os.path.join(log_dir, f'{args.dynamics}/{args.estimator}')
    # exp_dir = os.path.join(alg_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    # os.makedirs(exp_dir, exist_ok=True)
    # exp_dir = Path(HydraConfig.get().run.dir)
    summary_writer = SummaryWriter(os.getcwd())
    log_git_details(log_file='git.diff')

    ### set env and collect data

    if args.dynamics == 'NoisyPendulum':
        data_generator = ParallelNoisyPendulum(
            # sigma=args.sigma,
            rollout_batch_size=args.train_batch_size,
            # sin_cos_obs=args.sin_cos_obs,
            prob=args.prob_labels,
            **vars(args))
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='./datasets', dist=args.sample)
    elif args.dynamics == 'mvn_contrastive_normal':
        data_generator = MVNContrasiveNormal(rollout_batch_size=args.train_batch_size, )
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='./datasets')
        neg_dataset, neg_prob = data_generator.noise_sample(batches=args.train_batches, store_path='./datasets')
    elif args.dynamics == 'mvn_uniform_contrastive_normal':
        data_generator = MVNUniformContrastiveNormal(rollout_batch_size=args.train_batch_size, )
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='/home/haitong/PycharmProjects/low_rank_learning/datasets')
        neg_dataset, neg_prob = data_generator.noise_sample(batches=args.train_batches, store_path='/home/haitong/PycharmProjects/low_rank_learning/datasets')
    elif args.dynamics == 'mvn_contrastive_unifrom':
        data_generator = MVNContrastiveUnifrom(rollout_batch_size=args.train_batch_size, )
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='/home/haitong/PycharmProjects/low_rank_learning/datasets')
        neg_dataset, neg_prob = data_generator.noise_sample(batches=args.train_batches, store_path='/home/haitong/PycharmProjects/low_rank_learning/datasets')

    elif args.dynamics == 'mvn_uniform':
        data_generator = MVNUniform(rollout_batch_size=args.train_batch_size, )
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='./datasets')
    else:
        raise NotImplementedError
    # if not args.estimator.startswith('supervised'):
    #     dataset = TransitionDataset(data=dataset, device=torch.device(args.device))
    # else:
    dataset = LabeledTransitionDataset(data=dataset, prob=prob, device=torch.device(args.device))


    ### initial training

    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    if 'contrastive' in args.dynamics:
        neg_dataset = LabeledTransitionDataset(data=neg_dataset, prob=neg_prob, device=torch.device(args.device))
        neg_dataloader = DataLoader(neg_dataset, batch_size=args.train_batch_size, shuffle=True)
    epoch = 10
    OmegaConf.set_struct(args.estimator, False)
    args.estimator['dynamics'] = args.dynamics
    args.estimator['device'] = args.device
    if args.estimator.name == 'mle':
        estimator = MLEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=data_generator.action_dim,
                                 **vars(args))
    elif args.estimator.name == 'nce':
        if args.estimator.noise_dist == 'uniform':
            noise_args = {'dist': "uniform",
                          'uniform_scale': [1.0, 1.0, 8.0]}
        else:
            raise NotImplementedError
        estimator = NCEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=data_generator.action_dim,
                                 noise_args=noise_args,
                                 **args.estimator)
    elif args.estimator.name == 'supervised':
        estimator = SupervisedEstimator(embedding_dim=args.feature_dim,
                                        state_dim=data_generator.state_dim,
                                        action_dim=data_generator.action_dim,
                                        **vars(args))
    elif args.estimator.name == 'supervised_rf':
        estimator = SupervisedLearnableRandomFeatureEstimator(embedding_dim=args.feature_dim,
                                                              state_dim=data_generator.state_dim,
                                                              action_dim=data_generator.action_dim,
                                                              **vars(args))
    else:
        raise NotImplementedError

    dataloader = zip(train_dataloader, neg_dataloader) if 'contrastive' in args.dynamics else train_dataloader

    pbar = tqdm(dataloader, desc='Epoch')
    for batch, transition in enumerate(pbar):
        info = estimator.estimate(transition)
        for key, value in info.items():
            if 'dist' in key:
                summary_writer.add_histogram(key, value, batch + 1)
            else:
                summary_writer.add_scalar(key, value, batch + 1)
        summary_writer.flush()
        # print(f"Epoch {batch + 1}, loss {info.get('est_loss')}")
        pbar.set_postfix(loss=info.get('est_loss'))

    estimator.save(os.getcwd())

    # save dicts
    # args_dict = vars(args)

    # with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
    #     json.dump(args_dict, json_file, indent=4)

    ## Evaluations

    # generating test set
    # evaluate(args, estimator=estimator)

if __name__ == '__main__':

    run()





