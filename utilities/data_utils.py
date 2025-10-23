# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for data loading and preprocessing."""

import random
import numpy as np
from urllib import request
import os
import h5py
KEYS = ['observations', 'actions', 'rewards', 'terminals']
import gym


def load_expert_data(filename):
    """Loads expert trajectoris from a file.

    Args:
      filename: a filename to load the data.

    Returns:
      Numpy arrays that contain states, actions, next_states and dones
    """
    with open(filename, 'rb') as fin:
        expert_data = np.load(fin)
        expert_data = {key: expert_data [key] for key in expert_data.files}

        expert_states = expert_data ['states']
        expert_actions = expert_data ['actions']
        expert_next_states = expert_data ['next_states']
        expert_dones = expert_data ['dones']
    return expert_states, expert_actions, expert_next_states, expert_dones


def subsample_trajectories(expert_states, expert_actions, expert_next_states,
                           expert_dones, num_trajectories):
    """Extracts a random subset of trajectories.

    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      num_trajectories: A number of trajectories to extract.

    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """
    expert_states_traj = [[]]
    expert_actions_traj = [[]]
    expert_next_states_traj = [[]]
    expert_dones_traj = [[]]

    for i in range(expert_states.shape [0]):
        expert_states_traj [-1].append(expert_states [i])
        expert_actions_traj [-1].append(expert_actions [i])
        expert_next_states_traj [-1].append(expert_next_states [i])
        expert_dones_traj [-1].append(expert_dones [i])

        if expert_dones [i] and i < expert_states.shape [0] - 1:
            expert_states_traj.append([])
            expert_actions_traj.append([])
            expert_next_states_traj.append([])
            expert_dones_traj.append([])

    shuffle_inds = list(range(len(expert_states_traj)))
    random.shuffle(shuffle_inds)
    shuffle_inds = shuffle_inds [:num_trajectories]
    expert_states_traj = [expert_states_traj [i] for i in shuffle_inds]
    expert_actions_traj = [expert_actions_traj [i] for i in shuffle_inds]
    expert_next_states_traj = [expert_next_states_traj [i] for i in shuffle_inds]
    expert_dones_traj = [expert_dones_traj [i] for i in shuffle_inds]

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    expert_states = concat_trajectories(expert_states_traj)
    expert_actions = concat_trajectories(expert_actions_traj)
    expert_next_states = concat_trajectories(expert_next_states_traj)
    expert_dones = concat_trajectories(expert_dones_traj)

    return expert_states, expert_actions, expert_next_states, expert_dones

def load_d4rl_data(dirname, env_id, dataname, num_trajectories, start_idx=0, dtype=np.float32):
    MAX_EPISODE_STEPS = 1000

    original_env_id = env_id
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()

    filename = f'{env_id}_{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')
    # if not exists
    if not os.path.exists(filepath):
        os.makedirs(dirname, exist_ok=True)
        # Download the dataset
        remote_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}.hdf5'
        print(f'Download dataset from {remote_url} into {filepath} ...')
        request.urlretrieve(remote_url, filepath)
        print(f'Done!')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()
    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, rew_, done_ = [], [], [], [], [], [], []
    episode_steps = 0
    num_episodes = 0
    for i in range(N - 1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            done_.append(done_bool)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break

    # env = gym.make(original_env_id)
    # if env.action_space.dtype == int:
    #     action_ = np.eye(env.action_space.n)[np.array(action_, dtype=np.int)]  # integer to one-hot encoding

    print(f'{num_episodes} trajectories are sampled')
    return np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype), np.array(action_, dtype=dtype), np.array(
        next_obs_, dtype=dtype), np.array(done_)


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env, dtype=np.float32):
    """Adds absorbing states to trajectories.
    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      env: A gym environment.
    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """

    # First add 0 indicator to all non-absorbing states.
    expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
    expert_next_states = np.pad(
        expert_next_states, ((0, 0), (0, 1)), mode='constant')

    expert_states = [x for x in expert_states]
    expert_next_states = [x for x in expert_next_states]
    expert_actions = [x for x in expert_actions]
    expert_dones = [x for x in expert_dones]

    # Add absorbing states.
    i = 0
    current_len = 0
    while i < len(expert_states):
        current_len += 1
        if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
            current_len = 0
            expert_states.insert(i + 1, env.get_absorbing_state())
            expert_next_states[i] = env.get_absorbing_state()
            expert_next_states.insert(i + 1, env.get_absorbing_state())
            action_dim = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            expert_actions.insert(i + 1, np.zeros((action_dim,), dtype=dtype))
            expert_dones[i] = 0.0
            expert_dones.insert(i + 1, 1.0)
            i += 1
        i += 1

    expert_states = np.stack(expert_states)
    expert_next_states = np.stack(expert_next_states)
    expert_actions = np.stack(expert_actions)
    expert_dones = np.stack(expert_dones)

    return expert_states.astype(dtype), expert_actions.astype(dtype), expert_next_states.astype(dtype), expert_dones.astype(dtype)

if __name__ == '__main__':
    # import os
    # filename = os.path.join('/home/haitong/PycharmProjects/low_rank_learning/datasets/imitation', 'HalfCheetah-v2.npz')
    # (expert_states, expert_actions, expert_next_states,
    #  expert_dones) = load_expert_data(filename)
    #
    # (expert_states, expert_actions, expert_next_states,
    #  expert_dones) = subsample_trajectories(expert_states,
    #                                                    expert_actions,
    #                                                    expert_next_states,
    #                                                    expert_dones,
    #                                                    1)
    # print('# of demonstraions: {}'.format(expert_states.shape [0]))
    # print(expert_states.max(axis=0), expert_states.min(axis=0))
    (expert_initial_states, expert_states, expert_actions, expert_next_states, expert_dones) = load_d4rl_data(
        '/home/haitong/PycharmProjects/low_rank_learning/datasets/imitation', 'HalfCheetah-v2',
        'expert-v2',
        1000, start_idx=0)
    shift = - np.mean(expert_states, 0)
    scale = 1.0 / (np.std(expert_states, 0) + 1e-6)
    print(shift, scale)
    post_processing = (expert_states + shift) * scale
    print(post_processing.std(axis=0))
    print(expert_initial_states.shape, expert_states.shape, expert_actions.shape)
    print(np.concatenate([expert_states, expert_actions, expert_next_states], axis=1).shape)