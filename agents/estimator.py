import torch
import numpy as np
from utils import MLP, LearnableRandomFeature, NormalizedMLP
EPS = 1e-6
from scipy.stats import norm
import os

torch.autograd.set_detect_anomaly(True)

class DensityEstimator(object):
    """
	General class for density estimator P(s' | s, a)

	"""

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        self.embedding_dim = embedding_dim
        self.device = torch.device(kwargs.get('device'))
        hidden_dim = kwargs.get('hidden_dim', 256)
        hidden_depth = kwargs.get('hidden_depth', 2)
        out_mod = torch.nn.Sigmoid() if kwargs.get('sigmoid_output', False) else torch.nn.Softplus()
        self.phi = MLP(input_dim=state_dim + action_dim,
                       hidden_dim=hidden_dim,
                       hidden_depth=hidden_depth,
                       output_dim=embedding_dim,
                       output_mod=out_mod).to(device=self.device)
        self.mu = MLP(input_dim=state_dim,
                      hidden_dim=hidden_dim,
                      hidden_depth=hidden_depth,
                      output_dim=embedding_dim,
                      output_mod=out_mod).to(device=self.device)


        self.state_dim = state_dim
        self.action_dim = action_dim

        self.phi_optimizer = torch.optim.Adam(params=self.phi.parameters(),
                                              lr=1e-3,
                                              betas=(0.9, 0.999))
        self.mu_optimizer = torch.optim.Adam(params=self.mu.parameters(),
                                             lr=1e-3,
                                             betas=(0.9, 0.999))
        self.kwargs = kwargs
        self.mse_loss_fn = torch.nn.MSELoss()

    def estimate(self, batch):
        raise NotImplementedError

    def get_prob(self, st_at, s_tp1):

        # phi_sa = 1 / (self.embedding_dim ** 0.5) * self.phi(st_at)
        # mu_stp1 = 1 / (self.embedding_dim ** 0.5) * self.mu(s_tp1)
        phi_sa = 1 / (self.embedding_dim) * self.phi(st_at)
        mu_stp1 = 1 / (self.embedding_dim) * self.mu(s_tp1)
        prob = torch.sum(phi_sa * mu_stp1, dim=-1)

        return torch.clamp(prob, min=1e-6) # clamping for numerical stability
        # return prob

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

    def load(self, exp_dir):
        self.phi.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_phi.pth')))
        self.mu.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_mu.pth')))

    def save(self, exp_dir):
        # if 'rf' not in args.estimator:
        torch.save(self.phi.state_dict(), os.path.join(exp_dir, 'feature_phi.pth'))
        torch.save(self.mu.state_dict(), os.path.join(exp_dir, 'feature_mu.pth'))
        # else:
        #     torch.save(estimator.rf.state_dict(), os.path.join(exp_dir, 'rf.pth'))
        #     torch.save(estimator.f.state_dict(), os.path.join(exp_dir, 'f.pth'))



class MLEEstimator(DensityEstimator):
    
    def __init__(self,embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)
        self.phi_optimizer = torch.optim.Adam(params=self.phi.parameters(),
                                              lr=kwargs.get('mle_lr', 1e-3),
                                              betas=(0.9, 0.999))
        self.mu_optimizer = torch.optim.Adam(params=self.mu.parameters(),
                                             lr=kwargs.get('mle_lr', 1e-3),
                                             betas=(0.9, 0.999))


    def estimate(self, batch):

        transition, labels = batch
        info = {}
        st_at, s_tp1 = (transition[:, :self.state_dim+self.action_dim],
                        transition[:, self.state_dim + self.action_dim:])
        prob = self.get_prob(st_at, s_tp1)
        log_prob = torch.log(prob)
        mle_loss = -1 * torch.mean(log_prob)
        info.update({'est_loss': mle_loss.item()})

        reg_norm_loss = self.normalize_or_regularize(log_prob)
        loss = mle_loss + reg_norm_loss
        info.update({'reg_norm_loss': reg_norm_loss.item()})

        self.phi_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        loss.backward()
        self.phi_optimizer.step()
        self.mu_optimizer.step()

        mse_loss = self.mse_loss_fn(prob, labels)
        info.update({'mse_loss': mse_loss.item(),
                     'dist_predicted': prob.detach().cpu().numpy(),
                     'dist_true': labels.detach().cpu().numpy()
                     })

        return info


class NCEEstimator(DensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)
        self.noise_args = self.kwargs.get('noise_args', {})
        # if self.noise_args.get('dist') == 'uniform':
        #     uniform_scale = self.noise_args.get('uniform_scale')
        #     uniform_scale = torch.tensor(uniform_scale).float()
        if kwargs.get('prob_labels', 'conditional') == 'joint':
            self.noise_dist = torch.distributions.normal.Normal(loc=torch.tensor([0., 0., 0., 0., 0.]).to(self.device),
                                                                scale=torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0,]).to(self.device))
        elif kwargs.get('prob_labels', 'conditional') == 'conditional':
            self.noise_dist = torch.distributions.normal.Normal(loc=torch.tensor([0., 0.]).to(self.device),
                                                                scale=torch.tensor([1.0, 2.0]).to(self.device))
        else:
            raise NotImplementedError('noise dist for NCE not implemented')

        self.K = self.kwargs.get('num_classes', 1)

        self.phi_optimizer = torch.optim.Adam(params=self.phi.parameters(),
                                              lr=kwargs.get('nce_lr', 1e-3),
                                              betas=(0.9, 0.999))
        self.mu_optimizer = torch.optim.Adam(params=self.mu.parameters(),
                                             lr=kwargs.get('nce_lr', 1e-3),
                                             betas=(0.9, 0.999))

    def estimate(self, batch):

        transition, labels = batch

        info = {}
        st_at, s_tp1 = (transition[:, :self.state_dim + self.action_dim],
                        transition[:, self.state_dim + self.action_dim:])

        if self.kwargs.get('nce_loss') == 'ranking':
            nce_loss = self._rankingClassificationLoss(transition)
        elif self.kwargs.get('nce_loss') == 'binary':
            nce_loss = self._binaryClassificationLoss(transition)
        elif self.kwargs.get('nce_loss') == 'self_contrastive':
            nce_loss = self._self_contrastive_loss(st_at, s_tp1)
        else:
            raise NotImplementedError('NCE loss not implemented')
        info.update({'est_loss': nce_loss.item()})

        prob = self.get_prob(st_at, s_tp1)
        log_prob = torch.log(prob)
        reg_norm_loss = self.normalize_or_regularize(log_prob)
        loss = nce_loss + reg_norm_loss
        info.update({'reg_norm_loss': reg_norm_loss.item()})

        mse_loss = self.mse_loss_fn(prob, labels)
        info.update({'mse_loss': mse_loss.item(),
                     'dist_predicted': prob.detach().cpu().numpy(),
                     'dist_true': labels.detach().cpu().numpy()
                     })

        self.phi_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        loss.backward()
        self.phi_optimizer.step()
        self.mu_optimizer.step()

        return info

    def get_log_prob(self, inputs):
        st_at, s_tp1 = (inputs[:, :self.state_dim + self.action_dim],
                        inputs[:, self.state_dim + self.action_dim:])
        prob = self.get_prob(st_at, s_tp1)
        log_prob = torch.log(prob)
        return log_prob

    def _binaryClassificationLoss(self, inputs):
        st_at, s_tp1 = (inputs[:, :self.state_dim + self.action_dim],
                        inputs[:, self.state_dim + self.action_dim:])
        noise = self.noise_dist.sample([len(inputs)])  # only numbers of samples in the batch
        noised_transition = torch.hstack((st_at, noise))

        labels_pos = torch.ones(len(inputs))
        labels_neg = torch.zeros(len(inputs))
        labels = torch.cat((labels_pos, labels_neg)).to(self.device)
        inputs_combined = torch.vstack((inputs, noised_transition))
        log_prob_positive = self.get_log_prob(inputs_combined).squeeze()
        s_tp1_combined = inputs_combined[:, self.state_dim + self.action_dim:]
        log_prob_noise = torch.prod(self.noise_dist.log_prob(s_tp1_combined), dim=1)
        logits = log_prob_positive - log_prob_noise
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return loss

    def _rankingClassificationLoss(self, inputs):
        st_at, s_tp1 = (inputs[:, :self.state_dim + self.action_dim],
                        inputs[:, self.state_dim + self.action_dim:])
        log_prob_positive = self.get_log_prob(inputs)
        labels_list = [torch.ones(len(inputs))]
        inputs_list = [inputs]
        weights_list = [torch.ones(len(inputs))]
        s_tp1s = [s_tp1]
        for k in range(self.K):
            noise = self.noise_dist.sample([len(inputs)])# only numbers of samples in the batch
            if self.kwargs.get('prob_labels', 'conditional') == 'joint':
                noised_transition = torch.clamp(noise, torch.tensor([-np.pi, -4.0, -1.5, -np.pi, -4.0], device=self.device),
                                                 torch.tensor([np.pi, 4.0, 1.5, np.pi, 4.0], device=self.device))
            elif self.kwargs.get('prob_labels', 'conditional') == 'conditional':
                noised_transition = torch.vstack((st_at, noise))
            else:
                raise NotImplementedError('return prob types not implemented')
            inputs_list.append(noised_transition)
            labels_list.append(torch.zeros(len(inputs)))
            weights_list.append(1 / self.K * torch.ones(len(inputs)))
            s_tp1s.append(noise)
            # prob_negative = self.get_prob(noise)
            # prob_noise_negative = torch.prod(torch.exp(self.noise_dist.log_prob(noise)), 1)
            # evidence = evidence + torch.div(prob_negative, prob_noise_negative)
        inputs_combined = torch.vstack(inputs_list)
        log_prob_positive = self.get_log_prob(inputs_combined).squeeze()
        if self.kwargs.get('prob_labels', 'conditional') == 'joint':
            log_prob_noise = torch.prod(self.noise_dist.log_prob(inputs_combined), dim=1)
        elif self.kwargs.get('prob_labels', 'conditional') == 'conditional':
            s_tp1s_combined = torch.vstack(s_tp1s)
            log_prob_noise = torch.prod(self.noise_dist.log_prob(s_tp1s_combined), dim=1)
        else:
            raise NotImplementedError('return prob types not implemented')
        logits = log_prob_positive - log_prob_noise
        logits = torch.clamp(logits, -5., 5)
        labels_combined = torch.cat(labels_list).to(self.device)
        weights = torch.cat(weights_list).to(self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)

        ranking_loss = loss_fn(logits, labels_combined)
        if torch.isnan(ranking_loss):
            print('nan detected')
        return ranking_loss

    def _self_contrastive_loss(self, st_at, s_tp1):
        """
        Self contrastive loss, modified from
        https://github.com/shelowize/lvrep-rl/blob/main/agent/ctrlsac/ctrlsac_agent.py
        Parameters
        ----------
        st_at
        s_tp1

        Returns
        -------

        """
        phi_sa = self.phi(st_at)
        mu_stp1 = self.mu(s_tp1)

        labels = torch.eye(st_at.shape[0]).to(self.device)

        contrastive = (phi_sa[:, None, :] * mu_stp1[None, :, :]).sum(-1)
        model_loss = torch.nn.CrossEntropyLoss()
        model_loss = model_loss(contrastive, labels)
        return model_loss

class SupervisedEstimator(DensityEstimator):

    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(embedding_dim, state_dim, action_dim, **kwargs)


    def estimate(self, batch):

        transition, labels = batch
        st_at, s_tp1 = (transition[:, :self.state_dim + self.action_dim],
                        transition[:, self.state_dim + self.action_dim:])
        prob = self.get_prob(st_at, s_tp1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prob, labels)
        self.phi_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        loss.backward()
        self.phi_optimizer.step()
        self.mu_optimizer.step()

        info = {'est_loss': loss.item(),
                'dist_predicted': prob.detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy()
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
                                         device=self.device
                                         )
        nets = MLP if kwargs.get('layer_normalization', False) else NormalizedMLP
        self.f = nets(input_dim=state_dim + action_dim,
                     output_dim=state_dim,
                     hidden_dim=kwargs.get('hidden_dim', 256),
                     hidden_depth=kwargs.get('hidden_depth', 2),
                     ).to(self.device)


        self.rf_optimizer = torch.optim.Adam(params=self.rf.parameters(),
                                              lr=1e-3,
                                              betas=(0.9, 0.999))
        self.f_optimizer = torch.optim.Adam(params=self.f.parameters(),
                                             lr=1e-3,
                                             betas=(0.9, 0.999))
        self.kwargs = kwargs

        self.state_dim = state_dim
        self.action_dim = action_dim


    def get_prob(self, st_at, s_tp1):
        fsa = self.f(st_at)
        phi_fsa = self.rf(fsa)
        phi_stp1 = self.rf(s_tp1)

        prob = 16 * torch.mean(phi_fsa * phi_stp1, dim=-1)
        return prob

    def estimate(self, batch):
        transition, labels = batch
        st_at, s_tp1 = (transition[:, :self.state_dim + self.action_dim],
                        transition[:, self.state_dim + self.action_dim:])
        prob = self.get_prob(st_at, s_tp1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(prob, labels)
        self.rf_optimizer.zero_grad()
        self.f_optimizer.zero_grad()
        loss.backward()
        self.rf_optimizer.step()
        self.f_optimizer.step()

        info = {'est_loss': loss.item(),
                'dist_predicted': prob.detach().cpu().numpy(),
                'dist_true': labels.detach().cpu().numpy()
                }

        return info


