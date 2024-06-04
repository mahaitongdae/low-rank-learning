from envs.noisy_pendulum import ParallelNoisyPendulum
from utils import TransitionDataset, LabeledTransitionDataset
import torch
from torch.utils.data import DataLoader
from agents.estimator import MLEEstimator, NCEEstimator, SupervisedEstimator, SupervisedLearnableRandomFeatureEstimator
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
    parser.add_argument("--train_batches", default=50000, type=int)
    parser.add_argument("--train_batch_size", default=1024, type=int)

    # Tasks
    parser.add_argument('--dynamics', default='NoisyPendulum', type=str)
    parser.add_argument('--sigma', default=4.0, type=float)
    parser.add_argument("--sample", default='uniform_theta', type=str,
                        help="how the s, a distribution is sampled, uniform_theta or uniform_sin_theta")
    parser.add_argument("--sin_cos_obs", action='store_true')
    parser.set_defaults(sin_cos_obs=False)
    parser.add_argument("--prob_labels", default='conditional', type=str,
                        help="what probability returned by data generators. joint for P(s, a, sprime) and " +
                        "conditional for P(sprime | s, a)")

    ## Sanity check arguments
    parser.add_argument("--layer_normalization", action='store_true')
    parser.set_defaults(layer_normalization=True)
    parser.add_argument("--preprocess", default='none', type=str)

    ## Estimators general
    parser.add_argument('--estimator', default='nce', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)

    parser.add_argument('--feature_dim', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--hidden_depth', default=3, type=int)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=False)
    parser.add_argument("--logprob_regularization_weights", default=1., type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=True)
    parser.add_argument("--integral_normalization_weights", default=0.1, type=float)

    # MLE
    parser.add_argument('--sigmoid_output', action='store_true')
    parser.set_defaults(sigmoid_output=False)

    # NCE
    parser.add_argument("--nce_loss", default='ranking', type=str,
                        help="loss function for noise contrastive learning, either binary or ranking or self_contrastive.")
    parser.add_argument("--nce_lr", default=3e-5, type=float)
    parser.add_argument("--noise_dist", default='uniform', type=str,
                        help="noise distribution")
    parser.add_argument("--num_classes", default=3, type=int,
                        help="number of classes in the NCE, K in the paper.")


    args = parser.parse_args()

    ### set file path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir =os.path.join(root_dir, 'log')
    alg_dir = os.path.join(log_dir, f'{args.dynamics}/{args.estimator}')
    exp_dir = os.path.join(alg_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(exp_dir, exist_ok=True)
    summary_writer = SummaryWriter(exp_dir)

    ### set env and collect data

    if args.dynamics == 'NoisyPendulum':
        data_generator = ParallelNoisyPendulum(sigma=args.sigma,
                                               rollout_batch_size=args.train_batch_size,
                                               sin_cos_obs=args.sin_cos_obs,
                                               prob=args.prob_labels)
        dataset, prob = data_generator.sample(batches=args.train_batches, store_path='./datasets',dist=args.sample)
    else:
        raise NotImplementedError
    # if not args.estimator.startswith('supervised'):
    #     dataset = TransitionDataset(data=dataset, device=torch.device(args.device))
    # else:
    dataset = LabeledTransitionDataset(data=dataset, prob=prob, device=torch.device(args.device))

    ### initial training

    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    # len(train_dataloader)
    epoch = 10
    if args.estimator == 'mle':
        estimator = MLEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=1,
                                 **vars(args))
    elif args.estimator == 'nce':
        if args.noise_dist == 'uniform':
            noise_args = {'dist': "uniform",
                          'uniform_scale': [1.0, 1.0, 8.0]}
        else:
            raise NotImplementedError
        estimator = NCEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=1,
                                 noise_args=noise_args,
                                 **vars(args))
    elif args.estimator == 'supervised':
        estimator = SupervisedEstimator(embedding_dim=args.feature_dim,
                                        state_dim=data_generator.state_dim,
                                        action_dim=1,
                                        **vars(args))
    elif args.estimator == 'supervised_rf':
        estimator = SupervisedLearnableRandomFeatureEstimator(embedding_dim=args.feature_dim,
                                                              state_dim=data_generator.state_dim,
                                                              action_dim=1,
                                                              **vars(args))
    else:
        raise NotImplementedError

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

    ## Evaluations

    # generating test set
    # evaluate(args, estimator=estimator)



