import torch
import numpy as np
from utils import MLP, LearnableRandomFeature, NormalizedMLP, Encoder, Decoder
EPS = 1e-6
from scipy.stats import norm
import os
from torch.func import vmap, jacrev, functional_call, hessian

class SingleNetworkDensityEstimator(object):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        self.device = torch.device(kwargs.get('device'))
        input_dim = state_dim if kwargs.get('noise_input', False) else state_dim + action_dim + state_dim
        if kwargs.get("true_parametric_model", False):
            assert kwargs.get('noise_input', False)
            assert kwargs.get('preprocess', 'none') == 'none'
            self.sigma = torch.nn.Parameter(torch.tensor(0.1))
            if kwargs.get('output_log_prob', False):
                self.f = lambda x: torch.log(1 / (2 * np.pi * self.sigma ** 2) * torch.exp(
                    -0.5 * torch.norm(x, dim=1) ** 2 / (self.sigma ** 2)))
            else:
                self.f = lambda x: 1 / (2 * np.pi * self.sigma ** 2) * torch.exp(
                    -0.5 * torch.norm(x, dim=1) ** 2 / (self.sigma ** 2))
            # self.c = torch.nn.Parameter(torch.tensor(0., device=self.device))
            self.f_optimizer = torch.optim.Adam(
                [self.sigma], # , self.c
                # self.f.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999))
        else:
            output_mod = None if kwargs.get('output_log_prob', False) else torch.nn.Softplus()
            self.f = MLP(input_dim=input_dim,
                          output_dim=1,
                          hidden_dim=kwargs.get('hidden_dim', 256),
                          hidden_depth=kwargs.get('hidden_depth', 2),
                          preprocess=kwargs.get('preprocess', None),
                          output_mod=output_mod
                          ).to(self.device)


            # else:
            self.f_optimizer = torch.optim.Adam(self.f.parameters(),
                                                lr=3e-4,
                                                betas=(0.9, 0.999))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kwargs = kwargs

    def get_prob(self, inputs):
        return torch.exp(self.f(inputs)) if self.kwargs.get('output_log_prob', False) else self.f(inputs)

    def get_log_prob(self, inputs):
        return self.f(inputs) if self.kwargs.get('output_log_prob', False) else torch.log(self.f(inputs))


    def ground_truth_estimate(self, batch):
        transition, labels = batch
        transition = transition.cpu().numpy()
        labels = labels.cpu().numpy()
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
        theta_ddot = 3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * at.squeeze()
        new_th = th + dt * thdot
        new_thdot = thdot + dt * theta_ddot
        # new_th = ((new_th + np.pi) % (2 * np.pi)) - np.pi
        new_thdot = np.clip(new_thdot, -max_speed, max_speed)
        f_sa = np.vstack([new_th, new_thdot]).T
        noise = s_tp1 - f_sa
        dens = norm.pdf(noise, loc = [0.0, 0.0], scale=[0.05, 0.05])
        dens_joint = np.prod(dens, axis=1)

        loss = np.mean((dens_joint - labels) ** 2)

        info = {'est_loss': loss,
                # 'dist_predicted': prob.detach().cpu().numpy(),
                # 'dist_true': labels.detach().cpu().numpy()
                }

        return info

    def get_noise_with_model(self, transition):

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

    def save(self, exp_dir):
        if self.kwargs.get('true_parametric_model', False):
            print(self.sigma)
        else:
            torch.save(self.f.state_dict(), os.path.join(exp_dir, 'f.pth'))
    def load(self, exp_dir):
        if self.kwargs.get('true_parametric_model', False):
            raise AssertionError('call load on true parametric model')
        else:
            self.f.load_state_dict(torch.load(os.path.join(exp_dir, 'f.pth')))

    def normalize_or_regularize(self, log_prob):
        if self.kwargs.get('integral_normalization', False):
            norm_weights = self.kwargs.get('integral_normalization_weights', 1.)
            normalization_loss = norm_weights * (torch.mean(torch.exp(log_prob)) - 1) ** 2
            return normalization_loss

        elif self.kwargs.get('logprob_regularization', False):
            # log prob regularization from Making Linear MDP practical via NCE, section 4.2
            regularization_loss = torch.mean(log_prob ** 2)
            regularization_weights = self.kwargs.get('logprob_regularization_weights', 1.)
            return regularization_weights * regularization_loss
        else:
            return torch.tensor(0., device=self.device)


class SupervisedSingleNetwork(SingleNetworkDensityEstimator):


    def estimate(self, batch):
        transition, labels = batch
        transition = transition
        # labels = labels.cpu().numpy()
        if self.kwargs.get("noise_input", False):
            inputs = self.get_noise_with_model(transition)
        else:
            inputs = transition
        prob = self.get_prob(inputs).squeeze()

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prob, labels / 64)
        self.f_optimizer.zero_grad()
        loss.backward()
        self.f_optimizer.step()

        info = {'est_loss': loss,
                'dist_predicted': prob.detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy()
                }

        return info



class NCESingleNetwork(SingleNetworkDensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)
        # use biased normal as noise distribution
        self.noise_dist = torch.distributions.normal.Normal(loc = torch.tensor([0., 0.]).to(self.device),
                                                              scale= torch.tensor([0.1, 0.1]).to(self.device))

        self.K = self.kwargs.get('num_classes', 1)
        if self.kwargs.get('nce_loss', None) == 'binary':
            self.nce_loss_fn = self._binaryClassificationLoss
            self.c = torch.nn.Parameter(torch.tensor([0.], device=self.device))
            if not self.kwargs.get('true_parametric_model', False):
                assert self.kwargs.get('output_log_prob', False)

                self.f_optimizer = torch.optim.Adam([{'params':self.f.parameters()},
                                                            {'params':(self.c)}],
                                                    lr=3e-4,
                                                    betas=(0.9, 0.999))
            else:
                pass
        elif self.kwargs.get('nce_loss', None) == 'ranking':
            self.nce_loss_fn = self._rankingClassificationLoss
        else:
            raise NotImplementedError('nce_loss: {} is not implemented'.format(self.kwargs.get('nce_loss', None)))

    def estimate(self, batch):
        transition, labels = batch
        transition = transition
        if self.kwargs.get("noise_input", False):
            inputs = self.get_noise_with_model(transition)
        else:
            inputs = transition

            if self.kwargs.get('preprocess', None) == 'diff_scale':
                diff_state = transition[:, self.state_dim + self.action_dim:] - transition[:, :self.state_dim]
                inputs = torch.vstack([transition[:, :self.state_dim + self.action_dim], diff_state])

        nce_loss = self.nce_loss_fn(inputs)
        log_prob = torch.log(self.get_prob(inputs))
        reg_norm_loss = self.normalize_or_regularize(log_prob)
        loss = nce_loss + reg_norm_loss
        # info.update({})
        self.f_optimizer.zero_grad()
        loss.backward()
        self.f_optimizer.step()
        # print(self.sigma)

        info = {'est_loss': loss,
                'reg_norm_loss': reg_norm_loss.item(),

                # 'dist_predicted': prob.detach().cpu().numpy(),
                # 'dist_true': labels.detach().cpu().numpy()
                }
        if self.kwargs.get('true_parametric_model', False):
            info.update({'sigma': self.sigma.item()})
        if self.kwargs.get('nce_loss', None) == 'binary':
            info.update({'c': self.c.item()})

        return info

    def _rankingClassificationLoss(self, inputs):
        log_prob_positive = self.get_log_prob(inputs)
        labels_list = [torch.ones(len(inputs))]
        inputs_list = [inputs]
        weights_list = [torch.ones(len(inputs))]
        # prob_noise_positive = torch.prod(torch.exp(self.noise_dist.log_prob(inputs)), 1)

        # joint = torch.div(prob_positive, prob_noise_positive)
        # evidence = torch.div(prob_positive, prob_noise_positive)
        for k in range(self.K):
            noise = self.noise_dist.sample([len(inputs)]) # only numbers of samples in the batch
            inputs_list.append(noise)
            labels_list.append(torch.zeros(len(inputs)))
            weights_list.append(1 / self.K * torch.ones(len(inputs)))
            # prob_negative = self.get_prob(noise)
            # prob_noise_negative = torch.prod(torch.exp(self.noise_dist.log_prob(noise)), 1)
            # evidence = evidence + torch.div(prob_negative, prob_noise_negative)
        inputs_combined = torch.vstack(inputs_list)
        log_prob_positive = self.get_log_prob(inputs_combined).squeeze()
        log_prob_noise = torch.prod(self.noise_dist.log_prob(inputs_combined), dim=1)
        logits = log_prob_positive - log_prob_noise
        labels_combined = torch.cat(labels_list).to(self.device)
        weights = torch.cat(weights_list).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)

        ranking_loss = loss_fn(logits, labels_combined)
        return ranking_loss

    def _binaryClassificationLoss(self, inputs):
        noise = self.noise_dist.sample([len(inputs)])  # only numbers of samples in the batch

        labels_pos = torch.ones(len(inputs))
        labels_neg = torch.zeros(len(inputs))
        labels = torch.cat((labels_pos, labels_neg)).to(self.device)
        inputs_combined = torch.vstack((inputs, noise))
        log_prob_positive = self.get_log_prob(inputs_combined).squeeze()
        log_prob_noise = torch.prod(self.noise_dist.log_prob(inputs_combined), dim=1)
        logits = log_prob_positive - log_prob_noise
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return loss

    def get_log_prob(self, inputs):
        assert self.kwargs.get('output_log_prob', False)
        # assert self.kwargs.get('nce_loss', None) == 'binary'
        if self.kwargs.get('nce_loss', None) == 'binary':
            return self.f(inputs) - self.c
        else:
            return self.f(inputs)


class MLESingleNetwork(SingleNetworkDensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)

    def estimate(self, batch):
        transition, labels = batch
        if self.kwargs.get("noise_input", False):
            inputs = self.get_noise_with_model(transition)
        else:
            inputs = transition
        # prob = self.f(torch.from_numpy(20 * inputs).to(self.device)).squeeze()
        log_prob = self.get_log_prob(inputs)
        # loss_fn = torch.nn.MSELoss()
        loss = -1 * torch.mean(log_prob)
        reg_norm_loss = self.normalize_or_regularize(log_prob)
        est_loss = loss + reg_norm_loss
        self.f_optimizer.zero_grad()
        est_loss.backward()
        self.f_optimizer.step()
        # print(self.sigma)

        info = {'est_loss': loss.item(),
                'dist_predicted': torch.exp(log_prob).detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy(),
                'reg_norm_loss':reg_norm_loss.item()
                }

        if self.kwargs.get('true_parametric_model', False):
            info.update({'sigma': self.sigma.item()})

        return info

class VBSingleNetwork(SingleNetworkDensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super(VBSingleNetwork, self).__init__(embedding_dim, state_dim, action_dim, **kwargs)
        input_dim = state_dim if kwargs.get('noise_input', False) else state_dim + action_dim + state_dim
        self.encoder = Encoder(input_dim=input_dim,
                               hidden_dim=kwargs.get('hidden_dim', 256),
                               output_dim=embedding_dim,
                               hidden_depth=kwargs.get('hidden_depth', 2),)
        self.decoder = Decoder(output_dim=1,
                               hidden_dim=kwargs.get('hidden_dim', 256),
                               feature_dim=embedding_dim,)

        self.vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=3e-4,
                                              betas=(0.9, 0.999))
        output_mod = None if kwargs.get('output_log_prob', False) else torch.nn.Softplus()

        self.f = MLP(input_dim=embedding_dim,
                     output_dim=1,
                     hidden_dim=kwargs.get('hidden_dim', 256),
                     hidden_depth=kwargs.get('hidden_depth', 2),
                     preprocess=kwargs.get('preprocess', None),
                     output_mod=output_mod
                     ).to(self.device)
        # else:
        self.f_optimizer = torch.optim.Adam(self.f.parameters(),
                                            lr=3e-4,
                                            betas=(0.9, 0.999))

        self.mse_loss = torch.nn.MSELoss()
        # self.z_prior =

    def get_prob(self, inputs):
        z = self.encoder.sample(inputs)
        return torch.exp(self.f(z)) if self.kwargs.get('output_log_prob', False) else self.f(z)

    def vb_loss(self, inputs):
        """
        KL(q(z\mid X=x) || q(z)) = KL(N(mu, sigma) || N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        z = self.encoder.sample(inputs)
        x = self.decoder(z)

        reconstuction_loss = self.mse_loss(x, inputs)
        mean, log_std = self.encoder(inputs)
        log_var = 2 * log_std
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        return reconstuction_loss, kl_loss

    def estimate(self, batch):
        transition, labels = batch
        transition = transition
        if self.kwargs.get("noise_input", False):
            inputs = self.get_noise_with_model(transition)
        else:
            inputs = transition

            if self.kwargs.get('preprocess', None) == 'diff_scale':
                diff_state = transition[:, self.state_dim + self.action_dim:] - transition[:, :self.state_dim]
                inputs = torch.vstack([transition[:, :self.state_dim + self.action_dim], diff_state])

        # feature step
        recon_loss, kl_loss = self.vb_loss(inputs)
        vae_loss = recon_loss + kl_loss

        loss = vae_loss
        # info.update({})
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.f_optimizer.step()
        # print(self.sigma)

        info = {'est_loss': vae_loss.item(),
                'recon_loss' : recon_loss.item(),
                'kl_loss' : kl_loss.item(),
                # 'dist_predicted': prob.detach().cpu().numpy(),
                # 'dist_true': labels.detach().cpu().numpy()
                }

        # estimation step
        features = self.encoder(inputs)
        prob = self.f(features)
        log_prob = torch.log(prob)
        reg_norm_loss = self.normalize_or_regularize(log_prob)
        mse_loss = self.mse_loss(prob, labels)
        est_loss = mse_loss +reg_norm_loss
        self.f_optimizer.zero_grad()
        mse_loss.backward()
        self.f_optimizer.step()

        info.update({'mse_loss': mse_loss.item(),
                     'reg_norm_loss': reg_norm_loss.item(),})

        return info

class ScoreMatchingSingleNetwork(SingleNetworkDensityEstimator):


    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super(ScoreMatchingSingleNetwork, self).__init__(embedding_dim, state_dim, action_dim, **kwargs)


    def estimate(self, batch):
        transition, labels = batch
        transition = transition
        if self.kwargs.get("noise_input", False):
            inputs = self.get_noise_with_model(transition)
        else:
            inputs = transition
        def get_log_prob(inputs):
            log_prob = self.get_log_prob(inputs).squeeze()
            return log_prob
        jac = vmap(jacrev(get_log_prob))(inputs)
        hessians = vmap(hessian(get_log_prob))(inputs)
        # more on the torch.func and computing batch jacobian and hessian can see
        # https://pytorch.org/docs/stable/generated/torch.func.jacrev.html
        # https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/6
        score_matching_loss = 0.5 * torch.norm(jac, dim=1).mean() + vmap(torch.trace)(hessians).mean()

        log_prob = self.get_log_prob(inputs)
        reg_norm_loss = self.normalize_or_regularize(log_prob)
        loss = score_matching_loss + reg_norm_loss

        # loss_fn = torch.nn.MSELoss()
        # loss = loss_fn(prob, labels)
        self.f_optimizer.zero_grad()
        loss.backward()
        self.f_optimizer.step()
        # print(self.sigma)

        info = {'est_loss': score_matching_loss.item(),
                'dist_predicted': self.get_prob(inputs).detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy(),
                'reg_norm_loss': reg_norm_loss.item(),
                }

        return info











