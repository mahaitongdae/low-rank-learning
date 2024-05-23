from envs.noisy_pendulum import ParallelNoisyPendulum
from utils import TransitionDataset, LabeledTransitionDataset
import torch
from torch.utils.data import DataLoader
from agents.estimator import MLEEstimator, NCEEstimator
from tensorboardX import SummaryWriter
import argparse
import os
from datetime import datetime
import json
import numpy as np
from evaluation import evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Pipelines
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--train_batches", default=10000, type=int)

    # Tasks
    parser.add_argument('--dynamics', default='NoisyPendulum', type=str)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument("--sample", default='uniform_sin_theta', type=str,
                        help="how the s, a distribution is sampled, uniform_theta or uniform_sin_theta")

    ## Estimators general
    parser.add_argument('--estimator', default='nce', type=str)
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=True)
    parser.add_argument("--logprob_regularization_weights", default=1., type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=False)
    parser.add_argument("--integral_normalization_weights", default=10, type=float)

    # MLE
    parser.add_argument('--sigmoid_output', action='store_true')
    parser.set_defaults(sigmoid_output=False)

    # NCE
    parser.add_argument("--nce_loss", default='self_contrastive', type=str,
                        help="loss function for noise contrastive learning, either binary or ranking or self_contrastive.")
    parser.add_argument("--noise_dist", default='uniform', type=str,
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
        dataset, _ = data_generator.sample(batches=args.train_batches, store_path='./datasets',dist=args.sample)
        dataset = TransitionDataset(data=dataset, device=torch.device(args.device))

    ### initial training

    train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    # len(train_dataloader)
    epoch = 10
    if args.estimator == 'mle':
        estimator = MLEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=3,
                                 action_dim=1,
                                 **vars(args))
    elif args.estimator == 'nce':
        if args.noise_dist == 'uniform':
            noise_args = {'dist': "uniform",
                          'uniform_scale': [1.0, 1.0, 8.0]}
        else:
            raise NotImplementedError
        estimator = NCEEstimator(embedding_dim=args.feature_dim,
                                 state_dim=3,
                                 action_dim=1,
                                 noise_args=noise_args,
                                 **vars(args))

    for batch, transition in enumerate(train_dataloader):
        info = estimator.estimate(transition)
        for key, value in info.items():
            summary_writer.add_scalar(key, value, batch + 1)
        summary_writer.flush()
        print(f"Epoch {batch + 1}, loss {info.get('est_loss')}")

        torch.save(estimator.phi.state_dict(), os.path.join(exp_dir, 'feature_phi.pth'))
        torch.save(estimator.mu.state_dict(), os.path.join(exp_dir, 'feature_mu.pth'))

    # save dicts
    args_dict = vars(args)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    ## Evaluations

    # generating test set
    # if args.d
    evaluate(args, estimator=estimator)
    # if args.dynamics == 'NoisyPendulum':
    #     data_generator = ParallelNoisyPendulum(sigma=args.sigma)
    #     test_dataset, test_prob = data_generator.sample(batches=10, seed=201,dist=args.sample)
    #     test_dataset = LabeledTransitionDataset(data=test_dataset, prob=test_prob, device=torch.device(args.device))
    #     test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
    #
    # evaluation_mse = []
    #
    # for batch, (sasp, prob) in enumerate(test_dataloader):
    #     st_at, s_tp1 = sasp[:, :4], sasp[:, 4:]
    #     predicted_joint = estimator.get_prob(st_at, s_tp1).cpu()
    #     print(f"mean predicted joint: {torch.mean(predicted_joint).item()}")
    #     true_marginal = data_generator.get_true_marginal(st_at.cpu())
    #     true_joint = np.multiply(prob.cpu(), true_marginal)
    #     eval_loss_fn = torch.nn.MSELoss()
    #     mse = eval_loss_fn(predicted_joint, true_joint)
    #     if mse.item() > 1:
    #         pass
    #     else:
    #         evaluation_mse.append(mse.item())
    # print(f"Evaluation MSE: {np.mean(evaluation_mse)}")



