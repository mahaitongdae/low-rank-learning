import torch
import numpy as np
from networks.networks import MLP, NormalizedMLP, LearnableRandomFeature, randMu2
from agents.estimator.estimator import DensityEstimator
EPS = 1e-6
from scipy.stats import norm
import os
from torch.func import vmap

torch.autograd.set_detect_anomaly(True)

LOG_PROB_MIN = -10
LOG_PROB_MAX = 6

class SupervisedEstimator(DensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)


    def estimate(self, batch):

        transition, labels = batch
        # st_at, s_tp1 = (transition[:, :self.state_dim + self.action_dim],
        #                 transition[:, self.state_dim + self.action_dim:])
        prob = self.get_prob(transition)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prob, labels)
        self.phi_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        loss.backward()
        self.phi_optimizer.step()
        self.mu_optimizer.step()

        info = {'est_loss': loss.item(),
                'dist_predicted': prob.detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy(),
                'dist_error': (prob - labels).detach().cpu().numpy()
                }

        return info

class SupervisedLearnableRandomFeatureEstimator(object):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        self.device = torch.device(kwargs.get('device'))
        self.rf = LearnableRandomFeature(input_dim=state_dim,
                                         output_dim=embedding_dim,
                                         hidden_dim=kwargs.get('hidden_dim', 256),
                                         hidden_depth=kwargs.get('hidden_depth', 2),
                                         batch_size=kwargs.get('train_batch_size', 512),
                                         sigma=kwargs.get('sigma', 1.),
                                         learnable_w=kwargs.get('learnable_w', True),
                                         device=self.device
                                         )
        nets = MLP if kwargs.get('layer_normalization', False) else NormalizedMLP
        self.f = nets(input_dim=state_dim + action_dim,
                     output_dim=state_dim,
                     hidden_dim=kwargs.get('hidden_dim', 256),
                     hidden_depth=kwargs.get('hidden_depth', 2),
                     ).to(self.device)


        self.rf_optimizer = torch.optim.Adam(params=self.rf.parameters(),
                                              lr=kwargs.get('lr', 1e-3),
                                              betas=(0.9, 0.999))
        self.f_optimizer = torch.optim.Adam(params=self.f.parameters(),
                                             lr=kwargs.get('lr', 1e-3),
                                             betas=(0.9, 0.999))
        self.kwargs = kwargs

        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_noise_with_model(self, transition):
        """
        Only for verification.

        Parameters
        ----------
        transition

        Returns
        -------

        """

        st, at, s_tp1 = (transition[:, :self.state_dim],
                         transition[:, self.state_dim:self.state_dim + self.action_dim],
                         transition[:, self.state_dim + self.action_dim:])
        th = st[:, 0]
        thdot = st[:, 1]
        max_speed = 8
        max_torque = 2.0
        dt = 0.05
        g = 10.0
        m = 1.0
        l = 1.0
        theta_ddot = 3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * at.squeeze()
        new_th = th + dt * thdot
        new_thdot = thdot + dt * theta_ddot
        # new_th = ((new_th + np.pi) % (2 * np.pi)) - np.pi
        new_thdot = torch.clamp(new_thdot, -max_speed, max_speed)
        f_sa = torch.vstack([new_th, new_thdot]).T
        noise = s_tp1 - f_sa
        return noise


    def get_prob(self, transition):
        st_at, s_tp1 = (transition[:, :self.state_dim + self.action_dim],
                        transition[:, self.state_dim + self.action_dim:])
        fsa = self.f(st_at)
        phi_fsa = self.rf(fsa)
        phi_stp1 = self.rf(s_tp1)

        prob = 16 * torch.mean(phi_fsa * phi_stp1, dim=-1)
        return prob

    def estimate(self, batch):
        transition, labels = batch
        prob = self.get_prob(transition)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prob, labels)
        self.rf_optimizer.zero_grad()
        self.f_optimizer.zero_grad()
        loss.backward()
        self.rf_optimizer.step()
        self.f_optimizer.step()

        info = {'est_loss': loss.item(),
                'dist_predicted': prob.detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy(),
                'dist_error': (prob-labels).detach().cpu().numpy()
                }

        return info

    def save(self, exp_dir):
        # else:
        torch.save(self.rf.state_dict(), os.path.join(exp_dir, 'rf.pth'))
        torch.save(self.f.state_dict(), os.path.join(exp_dir, 'f.pth'))

    def load(self, exp_dir):
        self.rf.load_state_dict(torch.load(os.path.join(exp_dir, 'rf.pth')))
        self.f.load_state_dict(torch.load(os.path.join(exp_dir, 'f.pth')))