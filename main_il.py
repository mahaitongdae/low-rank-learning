from envs.noisy_pendulum import ParallelNoisyPendulum
from envs.mvn import MVN, MVNUniform
from utils import TransitionDataset, LabeledTransitionDataset, TransitionDatasetfromD4RL
from data_utils import load_d4rl_data
import torch
from torch.utils.data import DataLoader
from agents.estimator import MLEEstimator, NCEEstimator, SupervisedEstimator, SupervisedLearnableRandomFeatureEstimator, SpectralSVDEstimator
from agents.single_network_estimator import NCESingleNetwork
from agents.imitator import SpectralSVDImitator
from tensorboardX import SummaryWriter
import argparse
import os
from datetime import datetime
import json
import numpy as np
# from evaluation import evaluate
from buffer import ReplayBuffer
import gymnasium
from utils import Timer
from utilities.eval import eval_policy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Pipelines
    parser.add_argument("--device", default='mps', type=str)
    parser.add_argument("--train_batches", default=20000, type=int)
    parser.add_argument("--train_batch_size", default=256, type=int)

    # Tasks
    parser.add_argument('--env_id', default='HalfCheetah-v2', type=str)
    parser.add_argument('--expert_dataset_name', default="expert-v2")
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
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--hidden_depth', default=2, type=int)
    parser.add_argument('--logprob_regularization', action='store_true')
    parser.set_defaults(logprob_regularization=True)
    parser.add_argument("--logprob_regularization_weights", default=10., type=float)
    parser.add_argument("--integral_normalization", action='store_true')
    parser.set_defaults(integral_normalization=False)
    parser.add_argument("--integral_normalization_weights", default=0.1, type=float)

    ## imitation learning
    parser.add_argument("--start_timesteps", default=5000, type=float,
                        help='the number of initial steps that collects data via random sampled actions.')  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e4, type=int,
                        help='number of iterations as the interval to evaluate trained policy.')  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=float,
                        help='the total training time steps / iterations.')  # Max time steps to run environment

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
    imitator = SpectralSVDImitator(embedding_dim=args.feature_dim,
                                   state_dim=expert_initial_states.shape[-1],
                                   action_dim=expert_actions.shape[-1],
                                   shift=shift,
                                   scale=scale,
                                   **vars(args))

    for batch, transition in enumerate(train_dataloader):
        info = imitator.estimate(transition)
        for key, value in info.items():
            if 'dist' in key:
                summary_writer.add_histogram(key, value, batch+1)
            else:
                summary_writer.add_scalar(key, value, batch + 1)
        summary_writer.flush()
        print(f"Epoch {batch + 1}, loss {info.get('est_loss')}")

    imitator.save(exp_dir)

    # save dicts
    args_dict = vars(args)

    with open(os.path.join(exp_dir, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


    # imitation learning phase

    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    env = gymnasium.make('HalfCheetah-v4')
    eval_env = gymnasium.make('HalfCheetah-v4')

    # Evaluate untrained policy
    evaluations = []
    replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=args.device)

    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    timer = Timer()

    # keep track of best eval model's state dict
    best_eval_reward = -1e6
    best_actor = None

    # # save parameters
    # # kwargs.update({"action_space": None}) # action space might not be serializable
    # with open(os.path.join(log_path, 'train_params.yaml'), 'w') as fp:
    #     yaml.dump(kwargs, fp, default_flow_style=False)

    for t in range(int(args.max_timesteps + args.start_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = imitator.select_action(state, explore=True)

        # Perform action
        next_state, reward, terminated, truncated, rollout_info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, next_state, reward, done)

        prev_state = np.copy(state)
        state = next_state
        episode_reward += reward
        info = {}

        if t >= args.start_timesteps:
            rb_batch = replay_buffer.sample(args.train_batch_size)
            info = imitator.imitate(train_dataloader, rb_batch, 0.99)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Info: {rollout_info}")
            # Reset environment
            info.update({'ep_len': episode_timesteps})
            state, _ = env.reset()
            done = False
            # prev_state = np.copy(state)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eval_len, eval_ret, _, _ = eval_policy(imitator, eval_env, eval_episodes=50)
            evaluations.append(eval_ret)

            if t >= args.start_timesteps:
                info.update({'eval_len': eval_len,
                             'eval_ret': eval_ret})

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

            if eval_ret > best_eval_reward:
                best_actor = imitator.actor.state_dict()

                # save best actor/best critic
                torch.save(best_actor, exp_dir + "/best_actor.pth")

            best_eval_reward = max(evaluations)

        if (t + 1) % 20 == 0:
            for key, value in info.items():
                if 'dist' not in key:
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                else:
                    for dist_key, dist_val in value.items():
                        summary_writer.add_histogram(dist_key, dist_val, t + 1)
            summary_writer.flush()

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(imitator.actor.state_dict(), exp_dir + "/actor_last.pth")



