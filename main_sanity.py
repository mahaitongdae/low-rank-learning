from envs.noisy_pendulum import ParallelNoisyPendulum
from utils import TransitionDataset, LabeledTransitionDataset
import torch
from torch.utils.data import DataLoader
from agents.estimator import (MLEEstimator,
                              NCEEstimator,
                              SupervisedEstimator,
                              SupervisedLearnableRandomFeatureEstimator)
from agents.single_network_estimator import (SupervisedSingleNetwork,
                                             NCESingleNetwork,
                                             MLESingleNetwork,
                                             ScoreMatchingSingleNetwork)
from tensorboardX import SummaryWriter
import argparse
import os
from datetime import datetime
import json
import numpy as np
from evaluation import evaluation_saved_features, evaluate_saved_single_networks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Pipelines
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--train_batches", default=5000, type=int)
    parser.add_argument("--train_batch_size", default=512, type=int)

    # Tasks
    parser.add_argument('--dynamics', default='NoisyPendulum', type=str)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument("--sample", default='gaussian', type=str,
                        help="how the s, a distribution is sampled, uniform_theta, uniform_sin_theta, gaussian")

    ## Sanity check arguments
    parser.add_argument("--noise_input", action='store_true')
    parser.set_defaults(noise_input=True)
    # parser.add_argument("--layer_normalization", action='store_true')
    # parser.set_defaults(layer_normalization=False)
    parser.add_argument("--preprocess", default='scale', type=str)
    parser.add_argument("--output_log_prob", action='store_true')
    parser.set_defaults(output_log_prob=True)
    parser.add_argument("--true_parametric_model", action='store_true')
    parser.set_defaults(true_parametric_model=False)


    ## Networks
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)

    ## Estimators and regularization
    parser.add_argument('--estimator', default='score_matching_single_network', type=str)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=True)
    parser.add_argument("--logprob_regularization_weights", default=100., type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=False)
    parser.add_argument("--integral_normalization_weights", default=1., type=float)

    # MLE
    parser.add_argument('--sigmoid_output', action='store_true')
    parser.set_defaults(sigmoid_output=False)

    # NCE
    parser.add_argument("--nce_loss", default='binary', type=str,
                        help="loss function for noise contrastive learning, either binary or ranking or self_contrastive.")
    parser.add_argument("--noise_dist", default='ranking', type=str,
                        help="noise distribution")
    parser.add_argument("--num_classes", default=5, type=int,
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
        data_generator = ParallelNoisyPendulum(sigma=args.sigma)
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
    # epoch = 10

    if args.estimator == 'nce_single_network':
        # if args.noise_dist == 'gaussian':
        #     noise_args = {'dist': "uniform",
        #                   'uniform_scale': [1.0, 1.0]}
        # else:
        #     raise NotImplementedError
        assert args.noise_input
        estimator = NCESingleNetwork(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=1,
                                 # noise_args=noise_args,
                                 **vars(args))
    elif args.estimator == 'mle_single_network':
        estimator = MLESingleNetwork(embedding_dim=args.feature_dim,
                                 state_dim=data_generator.state_dim,
                                 action_dim=1,
                                 # noise_args=noise_args,
                                 **vars(args))
    elif args.estimator == 'supervised_single_network':
        estimator = SupervisedSingleNetwork(embedding_dim=-1,
                                            state_dim=data_generator.state_dim,
                                            action_dim=1,
                                            **vars(args))
    elif args.estimator == 'score_matching_single_network':
        estimator = ScoreMatchingSingleNetwork(embedding_dim=-1,
                                               state_dim=data_generator.state_dim,
                                               action_dim=1,
                                               **vars(args))
    else:
        raise NotImplementedError


    for batch, transition in enumerate(train_dataloader):
        # for _ in range(5):
        info = estimator.estimate(transition)
        for key, value in info.items():
            if 'dist' in key:
                summary_writer.add_histogram(key, value, batch+1)
            else:
                summary_writer.add_scalar(key, value, batch + 1)
        summary_writer.flush()
        print(f"Epoch {batch + 1}, loss {info.get('est_loss')}")

    # if 'rf' not in args.estimator:
    #     torch.save(estimator.phi.state_dict(), os.path.join(exp_dir, 'feature_phi.pth'))
    #     torch.save(estimator.mu.state_dict(), os.path.join(exp_dir, 'feature_mu.pth'))
    # else:
    #     torch.save(estimator.rf.state_dict(), os.path.join(exp_dir, 'rf.pth'))
    #     torch.save(estimator.f.state_dict(), os.path.join(exp_dir, 'f.pth'))
    estimator.save(exp_dir)

    # save dicts
    args_dict = vars(args)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    ## Evaluations

    # generating test set
    # evaluate(args, estimator=estimator)



