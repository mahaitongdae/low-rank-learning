import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn


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


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

class LearnableRandomFeature(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 batch_size,
                 output_mod=None,
                 device=torch.device('cpu')
                 ):
        super().__init__()
        weights_dim = input_dim # TODO: we can also change here
        self.n = torch.normal(0, 1., size=(output_dim, input_dim)).to(device) # RF dim * s_dim
        self.trunk = mlp(input_dim, hidden_dim, input_dim, hidden_depth, output_mod)
        self.apply(weight_init)
        self.b = 2 * np.pi * torch.rand(size=(batch_size, output_dim)).to(device)
        self.trunk.to(device)


    def forward(self, x):
        w = self.trunk(self.n) # RF_dim * s_dim
        wx_p_b = x @ w.T + self.b
        return torch.cos(wx_p_b)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
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
