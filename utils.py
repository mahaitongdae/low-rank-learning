import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
import torch.nn.functional as F
import time

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Timer:

	def __init__(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def reset(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def set_step(self, step):
		self._step = step
		self._step_time = time.time()

	def time_cost(self):
		return time.time() - self._start_time

	def steps_per_sec(self, step):
		sps = (step - self._step) / (time.time() - self._step_time)
		self._step = step
		self._step_time = time.time()
		return sps

class TransitionDataset(Dataset):
    def __init__(self, file_path=None, data=None, device=None):
        if file_path is not None:
            self.data = np.load(file_path, allow_pickle=True)
        elif data is not None:
            self.data = data
        self.data = torch.from_numpy(self.data).float()
        if device:
            self.data=self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TransitionDatasetfromD4RL(Dataset):
    def __init__(self, expert_states, expert_actions, expert_next_states, device=None):
        transitions = np.concatenate([expert_states, expert_actions, expert_next_states], axis=1)
        self.data = torch.from_numpy(transitions).float()
        if device:
            self.data=self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class NoiseDataset4Repr(Dataset):
    def __init__(self, noise_states, device=None):
        # transitions = np.concatenate([expert_states, expert_actions, expert_next_states], axis=1)
        self.data = torch.from_numpy(noise_states).float()
        if device:
            self.data=self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LabeledTransitionDataset(Dataset):

    def __init__(self, file_path=None, data=None, prob=None, device=None):
        if file_path is not None:

            self.data = np.load('tran_' + file_path, allow_pickle=True)
        elif data is not None:
            self.data = data
            self.prob = prob
        self.data = torch.from_numpy(self.data).float()
        self.prob = torch.from_numpy(self.prob).float()
        if device:
            self.data = self.data.to(device)
            self.prob = self.prob.to(device)

    def __len__(self):
        assert len(self.data) == len(self.prob)
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.prob[idx]


class NoiseContrastiveDataset(TransitionDataset):

    def __init__(self, noise_distribution_scale, K, file_path=None, data=None):
        super().__init__(file_path, data)

        ## Now we assume uniform distributions.
        original_noise = torch.rand(len(self.data) * K, 1)
        noise = torch.kron(2 * noise_distribution_scale, original_noise) - noise_distribution_scale
        self.positive_label = torch.ones(len(self.data))
        self.negative_label = torch.zeros(len(noise))
        self.data = torch.vstack([self.data, noise])
        self.label = torch.hstack([self.positive_label, self.negative_label])

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, output_bias=True):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim, bias=output_bias))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
