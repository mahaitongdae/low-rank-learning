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
from agents.single_network_estimator import NCESingleNetwork, SupervisedSingleNetwork, MLESingleNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_helper(args, estimator=None, test_dataloader=None, data_generator=None, exp_dir=None):
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
        elif args.estimator == 'mle_single_network':
            estimator = MLESingleNetwork(embedding_dim=args.feature_dim,
                                         state_dim=data_generator.state_dim,
                                         action_dim=1,
                                         # noise_args=noise_args,
                                         **vars(args))
        elif args.estimator == 'nce_single_network':
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
        elif args.estimator == 'supervised_single_network':
            estimator = SupervisedSingleNetwork(embedding_dim=-1,
                                                state_dim=data_generator.state_dim,
                                                action_dim=1,
                                                **vars(args))

        else:
            raise NotImplementedError

    if exp_dir:
        estimator.load(exp_dir)

    return estimator, test_dataloader, data_generator

    #


def compare_joint_prob(args, estimator, test_dataloader, data_generator):
    evaluation_mse = []
    errors = []
    data_predicted = {"prob":[]}
    data_true = {"prob": []}

    for batch, (sasp, prob) in enumerate(test_dataloader):
        st_at, s_tp1 = (sasp[:, :data_generator.state_dim + data_generator.action_dim],
                        sasp[:, data_generator.state_dim + data_generator.action_dim:])
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

def plot_learned_kernel(args, estimator, test_dataloader, data_generator):
    assert args.noise_input
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    x = np.linspace(-0.1, 0.1, 1000)
    y = np.linspace(-0.1, 0.1, 1000)
    X, Y = np.meshgrid(x, y)
    inputs = torch.from_numpy(np.vstack([X.ravel(), Y.ravel()]).T).float().to(torch.device('cuda'))
    prob = estimator.get_prob(inputs).detach().cpu().numpy()
    Z = prob.reshape(X.shape)
    cf1 = ax1.contourf(X, Y, Z)
    fig.colorbar(cf1, ax=ax1)
    ax1.set_title('Learned kernel')
    # ax1.colorbar()
    dist = torch.distributions.normal.Normal(torch.tensor([0.0, 0.0]), torch.tensor([0.05, 0.05]))
    true_prob = torch.prod(torch.exp(dist.log_prob(inputs.cpu())), dim=-1).detach().numpy()
    true_z = true_prob.reshape(X.shape)
    # ax.plot_surface(X, Y, Z)
    cf2 = ax2.contourf(X, Y, true_z)
    fig.colorbar(cf2, ax=ax2)
    ax2.set_title('True kernel')
    plt.show()


def evaluation_saved_features(exp_dir):
    with open(os.path.join(exp_dir, 'args.json'), 'r') as json_file:
        args_dict = json.load(json_file)
    args = argparse.Namespace(**args_dict)
    estimator, test_dataloader, data_generator = evaluate_helper(args, exp_dir=exp_dir)
    compare_joint_prob(args, estimator, test_dataloader, data_generator)

def evaluate_saved_single_networks(exp_dir):
    with open(os.path.join(exp_dir, 'args.json'), 'r') as json_file:
        args_dict = json.load(json_file)
    args = argparse.Namespace(**args_dict)
    estimator, test_dataloader, data_generator = evaluate_helper(args, exp_dir=exp_dir)
    assert "single_network" in args.estimator
    plot_learned_kernel(args, estimator, test_dataloader, data_generator)




if __name__ == '__main__':
    # evaluation_saved_features('/home/haitong/PycharmProjects/low_rank_learning/log/NoisyPendulum/supervised/2024-05-23-11-11-45')
    evaluate_saved_single_networks('/home/haitong/PycharmProjects/low_rank_learning/log/NoisyPendulum/mle_single_network/2024-05-30-00-09-55')