from envs.noisy_pendulum import ParallelNoisyPendulum
from envs.mvn import MVN, MVNUniform
from utils import TransitionDataset, LabeledTransitionDataset, TransitionDatasetfromD4RL
from data_utils import load_d4rl_data
import torch
from torch.utils.data import DataLoader
from agents.estimator import MLEEstimator, NCEEstimator, SupervisedEstimator, SupervisedLearnableRandomFeatureEstimator, SpectralSVDEstimator
from agents.single_network_estimator import NCESingleNetwork
from tensorboardX import SummaryWriter
import argparse
import os
from datetime import datetime
import json
import numpy as np
# from evaluation import evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Pipelines
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--train_batches", default=20000, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)

    # Tasks
    parser.add_argument('--env_id', default='HalfCheetah-v2', type=str)
    parser.add_argument('--expert_dataset_name', default="random-v2")
    parser.add_argument('--expert_num_traj', default=500, type=int)
    # parser.add_argument('--logprob_regularization', action='store_true')
    # parser.set_defaults(logprob_regularization=True)
    # parser.add_argument("--logprob_regularization_weights", default=1., type=float)
    # parser.add_argument("--integral_normalization", action='store_true')
    # parser.set_defaults(integral_normalization=False)
    # parser.add_argument("--integral_normalization_weights", default=0.1, type=float)
    # parser.add_argument('--sigma', default=4.0, type=float)
    # parser.add_argument("--sample", default='uniform_theta', type=str,
    #                     help="how the s, a distribution is sampled, uniform_theta or uniform_sin_theta")
    # parser.add_argument("--sin_cos_obs", action='store_true')
    # parser.set_defaults(sin_cos_obs=False)
    # parser.add_argument("--prob_labels", default='conditional', type=str,
    #                     help="what probability returned by data generators. joint for P(s, a, sprime) and " +
    #                     "conditional for P(sprime | s, a)")
    # parser.add_argument("--pendulum_noise_dist", default='gaussian', type=str,
    #                     help="what is the noise distribution. be careful that this might be different with")

    ## Sanity check arguments
    parser.add_argument("--layer_normalization", action='store_true')
    parser.set_defaults(layer_normalization=True)
    parser.add_argument("--preprocess", default='none', type=str)

    ## Estimators general
    parser.add_argument('--estimator', default='spectral_svd', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--feature_dim', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--hidden_depth', default=3, type=int)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=True)
    parser.add_argument("--logprob_regularization_weights", default=1., type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=False)
    parser.add_argument("--integral_normalization_weights", default=0.1, type=float)

    args = parser.parse_args()

    ### set file path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir =os.path.join(root_dir, 'log')
    alg_dir = os.path.join(log_dir, f'{args.env_id}/{args.estimator}')
    exp_dir = os.path.join(alg_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(exp_dir, exist_ok=True)
    summary_writer = SummaryWriter(exp_dir)
    dataset_dir = os.path.join(root_dir, 'datasets/imitation')

    ### set env and collect data

    assert args.env_id == 'HalfCheetah-v2' # we currently only use this one
    (expert_initial_states, expert_states, expert_actions, expert_next_states, expert_dones) = load_d4rl_data(
        dataset_dir, args.env_id,
        args.expert_dataset_name,
        args.expert_num_traj, start_idx=0)
    shift = - np.mean(expert_states, 0)
    scale = 1.0 / (np.std(expert_states, 0) + 1e-6)


    dataset = TransitionDatasetfromD4RL(expert_states=expert_states,
                                        expert_actions=expert_actions,
                                        expert_next_states=expert_next_states,
                                        device=torch.device(args.device))

    ### initial training

    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    # len(train_dataloader)
    epoch = 10
    assert args.estimator == 'spectral_svd'
    estimator = SpectralSVDEstimator(embedding_dim=args.feature_dim,
                                     state_dim=expert_initial_states.shape[-1],
                                     action_dim=expert_actions.shape[-1],
                                     shift=shift,
                                     scale=scale,
                                     **vars(args))

    for batch, transition in enumerate(train_dataloader):
        info = estimator.estimate(transition)
        for key, value in info.items():
            if 'dist' in key:
                summary_writer.add_histogram(key, value, batch+1)
            else:
                summary_writer.add_scalar(key, value, batch + 1)
        summary_writer.flush()
        print(f"Epoch {batch + 1}, loss {info.get('est_loss')}")

    estimator.save(exp_dir)

    # save dicts
    args_dict = vars(args)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)



