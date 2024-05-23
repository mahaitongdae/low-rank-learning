import seaborn
import torch
import numpy as np
import os
import json
from envs.noisy_pendulum import ParallelNoisyPendulum
from utils import LabeledTransitionDataset
from torch.utils.data import DataLoader
import argparse
from agents.estimator import NCEEstimator, MLEEstimator
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(args, estimator=None, test_dataloader=None, data_generator=None, exp_dir=None):
    if test_dataloader is None:
        if args.dynamics == 'NoisyPendulum':
            data_generator = ParallelNoisyPendulum(sigma=args.sigma)
            test_dataset, test_prob = data_generator.sample(batches=10, seed=201, non_zero_initial=True,
                                                            dist=args.sample)
            test_dataset = LabeledTransitionDataset(data=test_dataset, prob=test_prob, device=torch.device(args.device))
            test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)
        else:
            raise NotImplementedError
    else:
        assert data_generator

    if estimator is None:
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
        else:
            raise NotImplementedError

    if exp_dir:
        estimator.phi.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_phi.pth')))
        estimator.mu.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_mu.pth')))

    compare_joint_prob(args, estimator, test_dataloader, data_generator)


def compare_joint_prob(args, estimator, test_dataloader, data_generator):
    evaluation_mse = []
    errors = []

    for batch, (sasp, prob) in enumerate(test_dataloader):
        st_at, s_tp1 = sasp[:, :4], sasp[:, 4:]
        predicted_joint = estimator.get_prob(st_at, s_tp1).cpu()
        print(f"mean predicted joint: {torch.mean(predicted_joint).item()}")
        sin_theta = st_at[:, 1].cpu()
        true_marginal = data_generator.get_true_marginal(st_at.cpu(), dist=args.sample)
        true_joint = np.multiply(prob.cpu(), true_marginal)

        normalized_true_joint = true_joint / torch.mean(true_joint)
        print(f"mean true joint: {torch.mean(normalized_true_joint).item()}")
        # true_marginal = (1 / np.pi / 16) * np.reciprocal(np.sqrt(1 - sin_theta ** 2) + 1e-8)  # arcsine distribution
        # true_joint = np.multiply(prob.cpu(), true_marginal)
        eval_loss_fn = torch.nn.MSELoss()

        mse = eval_loss_fn(predicted_joint, normalized_true_joint)
        errors += predicted_joint.tolist()
        if mse.item() > 1e6: # stability stuff
            pass
        else:
            evaluation_mse.append(mse.item())
    print(f"Evaluation MSE: {np.mean(evaluation_mse)}")
    print(evaluation_mse)
    plt.figure()
    sns.catplot(errors)
    plt.show()


def evaluation_saved_features(exp_dir):
    with open(os.path.join(exp_dir, 'args.json'), 'r') as json_file:
        args_dict = json.load(json_file)
    args = argparse.Namespace(**args_dict)
    evaluate(args, exp_dir=exp_dir)



if __name__ == '__main__':
    evaluation_saved_features('/home/haitong/PycharmProjects/low_rank_learning/log/NoisyPendulum/mle/2024-05-22-22-16-20')
