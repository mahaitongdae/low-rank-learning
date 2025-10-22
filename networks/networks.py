import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import mlp, weight_init


class randMu2(nn.Module):
    """mu(s') random function, trying to sample from posterior"""

    def __init__(self, obs_dim, rf_dim, output_dim, sigma=1.):
        """
        Random, W cos(wx+b) where W is output_dim * rf_dim

        Parameters
        ----------
        obs_dim
        rf_dim
        output_dim
        sigma
        """
        super().__init__()
        fourier_feats = nn.Linear(obs_dim, rf_dim)
        init.normal_(fourier_feats.weight, std=1. / sigma)
        init.uniform_(fourier_feats.bias, 0, 2 * np.pi)
        fourier_feats.weight.requires_grad = False
        fourier_feats.bias.requires_grad = False
        self.fourier = fourier_feats
        print("self.fourier weights", self.fourier.weight)
        rand_weights = nn.Linear(rf_dim, output_dim)
        init.normal_(rand_weights.weight, std=1.0)
        init.constant_(rand_weights.bias, 0)
        rand_weights.weight.requires_grad = False
        rand_weights.bias.requires_grad = False
        self.rand_weights = rand_weights
        self.rf_dim = rf_dim

    def forward(self, states: torch.Tensor):
        output = math.sqrt(1. / self.rf_dim) * self.rand_weights(torch.cos(self.fourier(states)))
        return output



class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 preprocess=None,
                 output_mod=None,
                 output_bias=True):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod, output_bias)
        self.apply(weight_init)
        # if preprocess == 'norm':
        #     self.preprocess = torch.nn.BatchNorm1d(input_dim)
        # elif preprocess == 'scale':
        #     if input_dim == 2:
        #         self.preprocess = lambda x: 20 * x
        #     elif input_dim == 5:
        #         self.preprocess = lambda x: torch.tensor([1 /np.pi, 1/8., 1 / 2., 1 /np.pi, 1/8.],
        #                                                  device=torch.device('cuda')) * x
        # elif preprocess == 'diff_scale':
        #     self.preprocess = lambda x: torch.tensor([1 /np.pi, 1/8., 1 / 2., 20., 20.,],
        #                                                  device=torch.device('cuda')) * x
        # elif preprocess == 'none' or preprocess is None:
        #     self.preprocess = lambda x: x
        # else:
        #     raise NotImplementedError('preprocess not implemented')

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
                 sigma,
                 output_mod=None,
                 learnable_w=True,
                 device=torch.device('cpu')
                 ):
        super().__init__()
        weights_dim = input_dim # TODO: we can also change here
        sigma = 0.05 * sigma
        self.n = torch.nn.Parameter(torch.normal(0, 1. / sigma,
                                                 size=(output_dim, input_dim),
                                                 requires_grad=learnable_w).to(device)) # RF dim * s_dim
        self.trunk = mlp(input_dim, hidden_dim, input_dim, hidden_depth, output_mod)
        self.apply(weight_init)
        self.b = 2 * np.pi * torch.rand(size=(batch_size, output_dim)).to(device)
        self.trunk.to(device)


    def forward(self, x):
        w = self.trunk(self.n) # RF_dim * s_dim
        wx_p_b = x @ w.T + self.b
        return torch.cos(wx_p_b)
