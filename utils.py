import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

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
                 preprocess=None,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)
        if preprocess == 'norm':
            self.preprocess = torch.nn.BatchNorm1d(input_dim)
        elif preprocess == 'scale':
            if input_dim == 2:
                self.preprocess = lambda x: 20 * x
            elif input_dim == 5:
                self.preprocess = lambda x: torch.tensor([1 /np.pi, 1/8., 1 / 2., 1 /np.pi, 1/8.],
                                                         device=torch.device('cuda')) * x
        elif preprocess == 'diff_scale':
            self.preprocess = lambda x: torch.tensor([1 /np.pi, 1/8., 1 / 2., 20., 20.,],
                                                         device=torch.device('cuda')) * x
        elif preprocess == 'none' or preprocess is None:
            self.preprocess = lambda x: x
        else:
            raise NotImplementedError('preprocess not implemented')

    def forward(self, x):
        x = self.preprocess(x)
        return self.trunk(x)

class NormalizedMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.larer_normlization = torch.nn.LayerNorm(input_dim)
        self.batch_norm = torch.nn.BatchNorm1d(input_dim)
        self.apply(weight_init)

    def forward(self, x):
        x = self.batch_norm(x)
        return self.trunk(x)

class Encoder(nn.Module):
    def __init__(self,input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth=2,):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):

        z = F.elu(self.l1(input))
        z = F.elu(self.l2(z))
        mean = self.mean_linear(z)
        log_std = self.log_std_linear(z)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, input):
        """
        """
        mean, log_std = self.forward(input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization
        return z

class Decoder(nn.Module):
  """
  Deterministic decoder (Gaussian with identify covariance)

  z -> s for conditional models
  z -> x for common models.
  """
  def __init__(
    self,
    output_dim,
    feature_dim=256,
    hidden_dim=256,):

    super(Decoder, self).__init__()

    self.l1 = nn.Linear(feature_dim, hidden_dim)
    self.state_linear = nn.Linear(hidden_dim, output_dim)
    # self.reward_linear = nn.Linear(hidden_dim, 1)


  def forward(self, feature):
    """
    Decode an input feature to observation
    """
    x = F.relu(self.l1(feature)) #F.relu(self.l1(feature))
    s = self.state_linear(x)
    # r = self.reward_linear(x)
    return s # , r


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
