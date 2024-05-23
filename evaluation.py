import seaborn
import torch
import numpy as np
import os
import json
from envs.noisy_pendulum import ParallelNoisyPendulum
from utils import LabeledTransitionDataset
from torch.utils.data import DataLoader
import argparse
from agents.estimator import NCEEstimator, MLEEstimator, SupervisedEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        elif args.estimator == 'supervised':
            estimator = SupervisedEstimator(embedding_dim=args.feature_dim,
                                            state_dim=3,
                                            action_dim=1,
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
    data_predicted = {"prob":[]}
    data_true = {"prob": []}

    for batch, (sasp, prob) in enumerate(test_dataloader):
        st_at, s_tp1 = sasp[:, :4], sasp[:, 4:]
        eval_loss_fn = torch.nn.MSELoss()
        true_conditional = prob.cpu()
        if not isinstance(estimator, SupervisedEstimator):
            predicted_joint = estimator.get_prob(st_at, s_tp1).cpu()
            print(f"mean predicted joint: {torch.mean(predicted_joint).item()}")
            true_marginal = data_generator.get_true_marginal(st_at.cpu(), dist=args.sample)
            predicted_conditional = np.divide(predicted_joint, true_marginal)

        else:
            predicted_conditional = estimator.get_prob(st_at, s_tp1).cpu()
        mse = eval_loss_fn(predicted_conditional, true_conditional)
        data_predicted['prob'] += predicted_conditional.tolist()
        data_true['prob'] += true_conditional.tolist()
        evaluation_mse.append(mse.item())
    print(f"Evaluation MSE: {np.mean(evaluation_mse)}")

    # plot
    df_predicted = pd.DataFrame.from_dict(data_predicted)
    df_true = pd.DataFrame.from_dict(data_true)
    df_predicted['label'] = 'predicted'
    df_true['label'] = 'true'
    plt.figure()
    sns.catplot(data=pd.concat([df_predicted, df_true], ignore_index=True),
                x='label', y='prob')
    plt.show()


def evaluation_saved_features(exp_dir):
    with open(os.path.join(exp_dir, 'args.json'), 'r') as json_file:
        args_dict = json.load(json_file)
    args = argparse.Namespace(**args_dict)
    evaluate(args, exp_dir=exp_dir)



if __name__ == '__main__':
    evaluation_saved_features('/home/haitong/PycharmProjects/low_rank_learning/log/NoisyPendulum/supervised/2024-05-23-11-11-45')
