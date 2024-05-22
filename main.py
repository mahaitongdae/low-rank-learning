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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Tasks
    parser.add_argument('--dynamics', default='NoisyPendulum', type=str)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument("--sample", default='uniform', type=str)

    ## Estimators general
    parser.add_argument('--estimator', default='nce', type=str)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=True)
    parser.add_argument("--logprob_regularization_weights", default=10, type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=False)
    parser.add_argument("--integral_normalization_weights", default=10, type=float)

    # MLE
    parser.add_argument('--sigmoid_output', action='store_true')
    parser.set_defaults(sigmoid_output=True)


    # NCE
    parser.add_argument("--nce_loss", default='ranking', type=str,
                        help="loss function for noise contrastive learning, either binary or ranking.")
    parser.add_argument("--noise_dist", default='uniform', type=str,
                        help="noise distribution")
    parser.add_argument("--num_classes", default=1, type=int,
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
        dataset, _ = data_generator.uniform_sample(batches=200, store_path='./datasets')
        dataset = TransitionDataset(data=dataset)

    ### initial training

    train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    # len(train_dataloader)
    epoch = 10
    if args.estimator == 'mle':
        estimator = MLEEstimator(embedding_dim=128,
                                 state_dim=3,
                                 action_dim=1,
                                 sigmoid_output=args.sigmoid_output,
                                 integral_normalization=args.integral_normalization)
    elif args.estimator == 'nce':
        if args.noise_dist == 'uniform':
            noise_args = {'dist': "uniform",
                          'uniform_scale': [1.0, 1.0, 8.0]}
        else:
            raise NotImplementedError
        estimator = NCEEstimator(embedding_dim=128,
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

    # Step 3: Write the extracted data to a JSON file
    with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    ## Evaluations

    # generating test set
    # if args.d
    if args.dynamics == 'NoisyPendulum':
        data_generator = ParallelNoisyPendulum(sigma=args.sigma)
        test_dataset, test_prob = data_generator.uniform_sample(batches=10, seed=201)
        test_dataset = LabeledTransitionDataset(data=test_dataset, prob=test_prob)
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    for batch, (sasp, prob) in enumerate(test_dataloader):
        st_at, s_tp1 = sasp[:, :4], sasp[:, 4:]
        predicted_joint = estimator.get_prob(st_at, s_tp1)
        sin_theta = st_at[:, 0]
        true_marginal = (1 / np.pi / 16) * np.reciprocal(np.sqrt(1 - sin_theta ** 2) + 1e-8) # arcsine distribution
        true_joint = np.multiply(prob, true_marginal)
        eval_loss = torch.nn.MSELoss()
        mse = eval_loss(predicted_joint, true_joint)
        print(f"MSE: {mse.item()}")



