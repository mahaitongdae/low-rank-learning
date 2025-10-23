from agents.estimator import SpectralSVDEstimator
import torch
from utils import MLP, LearnableRandomFeature, NormalizedMLP
from agents.actor import DiagGaussianActor
import numpy as np
import os


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done

def weighted_softmax(x, weights, dim=0):
    x = x - torch.max(x)
    return weights * torch.exp(x) / torch.sum(weights * torch.exp(x), dim=dim, keepdim=True)

def orthogonal_regularization(model, device, reg=1e-4):
    with torch.enable_grad():
        orth_loss = torch.zeros(1).to(device)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                # sym -= torch.eye(param_flat.shape[0]).to(device)
                # orth_loss = orth_loss + (reg * sym.abs().sum())
                orth_loss += torch.sum(torch.square(sym * (1 - torch.eye(sym.shape[0]).to(device))))
    return reg * orth_loss

class ValueDICEImitator(torch.nn.Module):

    def __init__(self, state_dim, action_dim, **kwargs):
        super(ValueDICEImitator, self).__init__()
        nu_lr = kwargs.get('nu_lr', 1e-3)
        policy_lr = kwargs.get('policy_lr', 1e-5)
        hidden_dim = kwargs.get('hidden_dim', 256)
        hidden_depth = kwargs.get('hidden_depth', 2)
        self.device = torch.device(kwargs.get('device'))
        self.nu = MLP(input_dim=state_dim + action_dim,
                      hidden_dim=hidden_dim,
                      hidden_depth=hidden_depth,
                      output_dim=1,
                      output_bias=False).to(self.device)
        self.actor = DiagGaussianActor(obs_dim=state_dim,
                                       action_dim=action_dim,
                                       hidden_dim=hidden_dim,
                                       hidden_depth=hidden_depth,
                                       log_std_bounds=[-5, 2]).to(self.device)

        self.nu_optimizer = torch.optim.Adam(self.nu.parameters(), lr=nu_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim
        alpha = 0.1
        self.log_alpha = torch.tensor(np.log(alpha)).float().to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                   lr=3e-4,
                                                   betas=[0.9, 0.999])
        self.kwargs = kwargs

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, explore = False):
        if isinstance(state, list):
            state = np.array(state)
        state = state.astype(np.float32)
        assert len(state.shape) == 1
        state = torch.from_numpy(state).to(self.device)
        state = state.unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if explore else dist.mean
        action = action.clamp(torch.tensor(-1, device=self.device),
                              torch.tensor(1, device=self.device))
        assert action.ndim == 2 and action.shape [0] == 1
        return to_np(action[0])

    def imitate(self, expert_dataloader, rb_batch, discount, replay_regularization = 0.05, nu_reg = 10.0):
        """
        pytorch version of ValueDICE,
        Parameters
        ----------
        expert_data
        policy_data
        discount
        replay_regularization
        nu_reg

        Returns
        -------

        """

        expert_data = next(iter(expert_dataloader))

        expert_states, expert_actions, expert_next_states = (expert_data[:, :self.state_dim],
                                                             expert_data[:,  self.state_dim: self.state_dim + self.action_dim],
                                                             expert_data[:,  self.state_dim + self.action_dim:])


        rb_states, rb_actions, rb_next_states, _, _ = unpack_batch(rb_batch)

        expert_next_actions_dist = self.actor(expert_next_states)
        expert_next_actions = expert_next_actions_dist.rsample()

        rb_next_actions_dist = self.actor(expert_next_states)
        rb_next_actions = rb_next_actions_dist.rsample().clamp(min=-1 + 1e-6, max=1 - 1e-6)
        log_prob = rb_next_actions_dist.log_prob(rb_next_actions).sum(-1, keepdim=True)

        expert_initial_states = expert_states.clone()
        exp_a0_dist = self.actor(expert_initial_states)
        exp_a0 = exp_a0_dist.rsample()
        rb_initial_states = rb_states.clone()
        rb_a0_dist = self.actor(rb_initial_states)
        rb_a0 = rb_a0_dist.rsample()

        expert_init_sa = torch.hstack((expert_initial_states, exp_a0))
        expert_inputs_sa = torch.hstack((expert_states, expert_actions))
        expert_next_inputs = torch.hstack((expert_next_states, expert_next_actions))

        rb_inputs_sa = torch.hstack((rb_states, rb_actions))
        rb_next_inputs = torch.hstack((rb_next_states, rb_next_actions))

        expert_nu_0 = self.nu(expert_init_sa)
        expert_nu = self.nu(expert_inputs_sa)
        expert_nu_next = self.nu(expert_next_inputs)

        rb_nu = self.nu(rb_inputs_sa)
        rb_nu_next = self.nu(rb_next_inputs)

        expert_diff = expert_nu - discount * expert_nu_next
        rb_diff = rb_nu - discount * rb_nu_next

        linear_loss_expert = torch.mean(expert_nu_0 * (1 - discount))
        linear_loss_rb = rb_diff.mean()

        rb_expert_diff = torch.vstack((expert_diff, rb_diff))
        rb_expert_weights = torch.vstack([
            torch.ones_like(expert_diff) * (1 - replay_regularization),
            torch.ones_like(rb_diff) * replay_regularization,
        ])
        rb_expert_weights = rb_expert_weights / rb_expert_weights.sum()

        with torch.no_grad():
            w_softmax = weighted_softmax(rb_expert_diff, rb_expert_weights, dim=0)

        nonlinear_loss = (w_softmax * rb_expert_diff).sum()

        linear_loss = linear_loss_expert * (1 - replay_regularization) + linear_loss_rb * replay_regularization

        loss = nonlinear_loss - linear_loss

        '''
        Gradient penalty from
        Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A.C., 2017. Improved training of wasserstein gans. Advances in neural information processing systems, 30.
        In pytorch implementation, we need to specify create_graph=True to create the graph of derivatives.
        '''

        alpha = torch.rand((len(expert_inputs_sa), 1), device=self.device)
        nu_inter = alpha * expert_inputs_sa + (1 - alpha) * rb_inputs_sa
        nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs
        nu_inter = torch.vstack((nu_inter, nu_next_inter))

        nu_grad = torch.autograd.grad(self.nu(nu_inter).sum(), nu_inter,create_graph=True)[0]
        nu_grad_penalty = torch.mean(
            torch.square(torch.norm(nu_grad, dim=-1, keepdim=True) - 1))

        nu_loss = loss + nu_grad_penalty * nu_reg
        pi_loss = -1 * loss + orthogonal_regularization(self.actor.trunk, self.device) # + self.alpha.detach() * log_prob.mean() #

        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss = (self.alpha *
        #               (-log_prob - self.target_entropy).detach()).mean()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()


        self.nu_optimizer.zero_grad()
        nu_loss.backward(retain_graph=True, inputs=list(self.nu.parameters()))

        self.actor_optimizer.zero_grad()
        pi_loss.backward(inputs=list(self.actor.parameters()))
        self.nu_optimizer.step()
        self.actor_optimizer.step()

        return {'loss': loss.item(), 'nu_expert': expert_nu.mean().item(), 'nu_rb': rb_nu.mean().item(),
                'nu_grad_penalty': nu_grad_penalty.item(),
                'actor_loss': pi_loss.item(),
                'policy_entropy': -1 * log_prob.mean().item(),
                # 'alpha_loss': alpha_loss,
                # 'alpha': self.alpha
                }


class ReprValueDICEImitator(ValueDICEImitator):
    def __init__(self, embedding_dim, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        hidden_dim = kwargs.get('hidden_dim', 256)
        hidden_depth = kwargs.get('hidden_depth', 2)
        nu_lr = kwargs.get('nu_lr', 1e-3)
        repr_lr = kwargs.get('repr_lr', 3e-4)
        # policy_lr = kwargs.get('policy_lr', 1e-5)
        # out_mod = torch.nn.Sigmoid() if kwargs.get('sigmoid_output', False) else torch.nn.Softplus()
        self.embedding_dim = embedding_dim
        self.phi = MLP(input_dim=state_dim + action_dim,
                       hidden_dim=hidden_dim,
                       hidden_depth=hidden_depth,
                       output_dim=embedding_dim,
                       output_bias=False).to(device=self.device)
        self.mu = MLP(input_dim=state_dim,
                      hidden_dim=hidden_dim,
                      hidden_depth=hidden_depth,
                      output_dim=embedding_dim).to(device=self.device)
        self.log_zeta = MLP(input_dim=state_dim + action_dim,
                      hidden_dim=hidden_dim,
                      hidden_depth=hidden_depth,
                      output_dim=1,
                      output_bias=False).to(device=self.device)
        self.phi_optimizer = torch.optim.Adam(params=self.phi.parameters(),
                                              lr=repr_lr,
                                              betas=(0.9, 0.999))
        self.mu_optimizer = torch.optim.Adam(params=self.mu.parameters(),
                                             lr=repr_lr,
                                             betas=(0.9, 0.999))
        self.phi_weights = torch.nn.Parameter(torch.randn((embedding_dim), device=self.device))
        self.mu_weights = torch.nn.Parameter(torch.randn((embedding_dim), device=self.device))
        # self.actor = DiagGaussianActor(state_dim,
        #                                action_dim,
        #                                hidden_dim,
        #                                hidden_depth, [-5, 2]).to(device=self.device)

        self.phi_weights_optimizer = torch.optim.Adam([self.phi_weights], lr=nu_lr)
        self.mu_weights_optimizer = torch.optim.Adam([self.mu_weights], lr=nu_lr)
        self.log_zeta_optimizer = torch.optim.Adam(self.log_zeta.parameters(), lr=nu_lr)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
        #                                         lr=policy_lr,
        #                                         betas=[0.9, 0.999])
        # self.log_alpha = torch.tensor(np.log(0.1)).float().to(self.device)
        # self.log_alpha.requires_grad = True
        # self.target_entropy = -action_dim
        # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
        #                                             lr=policy_lr,
        #                                             betas=[0.9, 0.999])

    def repr_regularize(self, log_prob):
        # if self.kwargs.get('integral_normalization', False):
        #     norm_weights = self.kwargs.get('integral_normalization_weights', 1.)
        #     normalization_loss = norm_weights * (torch.mean(torch.exp(log_prob)) - 1) ** 2
        #     return normalization_loss
        #
        if self.kwargs.get('logprob_regularization', False):
            # log prob regularization from Making Linear MDP practical via NCE, section 4.2
            regularization_loss = torch.mean(log_prob ** 2)
            regularization_weights = self.kwargs.get('logprob_regularization_weights', 10.)
            return regularization_weights * regularization_loss
        else:
            return torch.tensor(0., device=self.device)
    def learn_repr(self, expert_dataloader, noise_dataloader):
        info = {}
        expert_data = next(iter(expert_dataloader))
        # expert_states, expert_actions, expert_next_states = (expert_data[:, :self.state_dim],
        #                                                      expert_data[:,
        #                                                      self.state_dim: self.state_dim + self.action_dim],
        #                                                      expert_data[:, self.state_dim + self.action_dim:])
        st_at, s_tp1 = (expert_data[:, :self.state_dim + self.action_dim],
                        expert_data[:, self.state_dim + self.action_dim:])

        noise_states = next(iter(noise_dataloader))
        phi_sa = 1 / (self.embedding_dim ** 0.5) * self.phi(st_at)
        mu_stp1 = 1 / (self.embedding_dim ** 0.5) * self.mu(s_tp1)
        prob = torch.clamp(torch.sum(phi_sa * mu_stp1, dim=-1), min=1e-6)
        # noise = self.noise_dist.sample([len(transition)])  # only numbers of samples in the batch
        # noise = st_at[:, :self.state_dim]
        mu_noise_stp1 = 1 / (self.embedding_dim ** 0.5) * self.mu(noise_states)
        noise_prob = torch.sum(phi_sa * mu_noise_stp1, dim=-1)
        spectral_svd_loss = torch.mean(-2 * prob + noise_prob ** 2)

        log_prob = torch.log(prob)
        reg_norm_loss = self.repr_regularize(log_prob)
        loss = spectral_svd_loss + reg_norm_loss
        info.update({'est_loss': spectral_svd_loss.item(),
                     'reg_norm_loss': reg_norm_loss.item(),
                     'phi_norm': torch.norm(phi_sa, dim=1).mean().item(),
                     'mu_norm': torch.norm(mu_stp1, dim=1).mean().item(),
                     })

        self.phi_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        loss.backward()
        self.phi_optimizer.step()
        self.mu_optimizer.step()

        return info

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, explore = False):
        if isinstance(state, list):
            state = np.array(state)
        state = state.astype(np.float32)
        assert len(state.shape) == 1
        state = torch.from_numpy(state).to(self.device)
        state = state.unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if explore else dist.mean
        action = action.clamp(torch.tensor(-1, device=self.device),
                              torch.tensor(1, device=self.device))
        assert action.ndim == 2 and action.shape [0] == 1
        return to_np(action[0])


    def get_nu(self, sa):
        with torch.no_grad():
            phi = self.phi(sa)
        log_zeta = self.log_zeta(sa)
        q = torch.vmap(torch.inner, in_dims=(0, None))(phi, self.phi_weights).unsqueeze(dim=1) + log_zeta
        return q

    # def get_d_ratio(self, state):
    #     with torch.no_grad():
    #         mu = self.get_mu(state)
    #     d_ratio = torch.vmap(torch.inner, in_dims=(0, None))(mu, self.mu_weights).clamp(min=1e-8)
    #     return d_ratio

    # def get_d_sa_ratio(self, state, action):
    #     sa = torch.hstack((state, action))
    #     return (self.get_d_ratio(state) * self.pi_ratio(sa).squeeze()).clamp(min=1e-8)

    # def get_primal_dual_loss(self, state, action, next_state, next_action, init_state, initial_action, discount):
    #
    #     reward = -torch.log(self.get_d_sa_ratio(state, action))
    #     bellman_residual = reward + discount * self.get_nu(next_state, next_action) - self.get_nu(state, action)
    #     primal_dual_loss = ((1 - discount) * self.get_nu(init_state, initial_action).mean()
    #                         + torch.mean(self.get_d_ratio(state) * bellman_residual))
    #
    #     return primal_dual_loss

    # def imitate(self, expert_dataloader, rb_batch, discount, replay_regularization = 0.05, nu_reg = 10):
    #     """
    #     pytorch version of ValueDICE,
    #     Parameters
    #     ----------
    #     expert_data
    #     policy_data
    #     discount
    #     replay_regularization
    #     nu_reg
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     expert_data = next(iter(expert_dataloader))
    #
    #     expert_states, expert_actions, expert_next_states = (expert_data[:, :self.state_dim],
    #                                                          expert_data[:,  self.state_dim: self.state_dim + self.action_dim],
    #                                                          expert_data[:,  self.state_dim + self.action_dim:])
    #
    #
    #     rb_states, rb_actions, rb_next_states, _, _ = unpack_batch(rb_batch)
    #
    #     expert_next_actions_dist = self.actor(expert_next_states)
    #     expert_next_actions = expert_next_actions_dist.rsample()
    #
    #     rb_next_actions_dist = self.actor(expert_next_states)
    #     rb_next_actions = rb_next_actions_dist.rsample()
    #     log_prob = rb_next_actions_dist.log_prob(rb_next_actions).sum(-1, keepdim=True)
    #
    #
    #     expert_initial_states = expert_states.clone()
    #     exp_a0_dist = self.actor(expert_initial_states)
    #     exp_a0 = exp_a0_dist.rsample()
    #     rb_initial_states = rb_states.clone()
    #     rb_a0_dist = self.actor(rb_initial_states)
    #     rb_a0 = rb_a0_dist.rsample()
    #
    #     def get_loss():
    #
    #         expert_primal_dual_loss = self.get_primal_dual_loss(expert_states,
    #                                                             expert_actions,
    #                                                             expert_next_states,
    #                                                             expert_next_actions,
    #                                                             expert_initial_states,
    #                                                             exp_a0,
    #                                                             discount)
    #
    #         rb_primal_dual_loss = self.get_primal_dual_loss(rb_states,
    #                                                         rb_actions,
    #                                                         rb_next_states,
    #                                                         rb_next_actions,
    #                                                         rb_states,
    #                                                         rb_a0,
    #                                                         discount
    #                                                         )
    #         loss = (1 - replay_regularization) * expert_primal_dual_loss + replay_regularization * rb_primal_dual_loss
    #         return loss
    #
    #     self.mu_weights_optimizer.zero_grad()
    #     self.log_zeta_optimizer.zero_grad()
    #     primal_dual_loss_d = -1 * get_loss()
    #     primal_dual_loss_d.backward(retain_graph=True)
    #     self.mu_weights_optimizer.step()
    #     self.log_zeta_optimizer.step()
    #
    #     self.phi_weights_optimizer.zero_grad()
    #     primal_dual_loss_q = get_loss()
    #     primal_dual_loss_q.backward()
    #     self.phi_weights_optimizer.step()
    #
    #     self.actor_optimizer.zero_grad()
    #     primal_dual_loss_pi = self.alpha.detach() * log_prob.mean() - get_loss()
    #
    #     primal_dual_loss_pi.backward()
    #     self.actor_optimizer.step()
    #
    #     if True:
    #         self.log_alpha_optimizer.zero_grad()
    #         alpha_loss = (self.alpha *
    #                       (-log_prob - self.target_entropy).detach()).mean()
    #         alpha_loss.backward()
    #         self.log_alpha_optimizer.step()
    #
    #
    #     info = {'primal_dual_loss_d': primal_dual_loss_d.item(),
    #             'primal_dual_loss_q': primal_dual_loss_q.item(),
    #             'primal_dual_loss_pi': primal_dual_loss_pi.item(),}
    #
    #     info ['alpha_loss'] = alpha_loss
    #     info ['alpha'] = self.alpha
    #
    #     return info
    def imitate(self, expert_dataloader, rb_batch, discount, replay_regularization = 0.05, nu_reg = 10.0):
        """
        pytorch version of ValueDICE,
        Parameters
        ----------
        expert_data
        policy_data
        discount
        replay_regularization
        nu_reg

        Returns
        -------

        """

        expert_data = next(iter(expert_dataloader))

        expert_states, expert_actions, expert_next_states = (expert_data[:, :self.state_dim],
                                                             expert_data[:,  self.state_dim: self.state_dim + self.action_dim],
                                                             expert_data[:,  self.state_dim + self.action_dim:])


        rb_states, rb_actions, rb_next_states, _, _ = unpack_batch(rb_batch)

        expert_next_actions_dist = self.actor(expert_next_states)
        expert_next_actions = expert_next_actions_dist.rsample()

        rb_next_actions_dist = self.actor(expert_next_states)
        rb_next_actions = rb_next_actions_dist.rsample().clamp(min=-1 + 1e-6, max=1 - 1e-6)
        log_prob = rb_next_actions_dist.log_prob(rb_next_actions).sum(-1, keepdim=True)

        expert_initial_states = expert_states.clone()
        exp_a0_dist = self.actor(expert_initial_states)
        exp_a0 = exp_a0_dist.rsample()
        rb_initial_states = rb_states.clone()
        rb_a0_dist = self.actor(rb_initial_states)
        rb_a0 = rb_a0_dist.rsample()

        expert_init_sa = torch.hstack((expert_initial_states, exp_a0))
        expert_inputs_sa = torch.hstack((expert_states, expert_actions))
        expert_next_inputs = torch.hstack((expert_next_states, expert_next_actions))

        rb_inputs_sa = torch.hstack((rb_states, rb_actions))
        rb_next_inputs = torch.hstack((rb_next_states, rb_next_actions))

        expert_nu_0 = self.get_nu(expert_init_sa)
        expert_nu = self.get_nu(expert_inputs_sa)
        expert_nu_next = self.get_nu(expert_next_inputs)

        rb_nu = self.get_nu(rb_inputs_sa)
        rb_nu_next = self.get_nu(rb_next_inputs)

        expert_diff = expert_nu - discount * expert_nu_next
        rb_diff = rb_nu - discount * rb_nu_next

        linear_loss_expert = torch.mean(expert_nu_0 * (1 - discount))
        linear_loss_rb = rb_diff.mean()

        rb_expert_diff = torch.vstack((expert_diff, rb_diff))
        rb_expert_weights = torch.vstack([
            torch.ones_like(expert_diff) * (1 - replay_regularization),
            torch.ones_like(rb_diff) * replay_regularization,
        ])
        rb_expert_weights = rb_expert_weights / rb_expert_weights.sum()

        with torch.no_grad():
            w_softmax = weighted_softmax(rb_expert_diff, rb_expert_weights, dim=0)

        nonlinear_loss = (w_softmax * rb_expert_diff).sum()

        linear_loss = linear_loss_expert * (1 - replay_regularization) + linear_loss_rb * replay_regularization

        loss = nonlinear_loss - linear_loss

        '''
        Gradient penalty from
        Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. and Courville, A.C., 2017. Improved training of wasserstein gans. Advances in neural information processing systems, 30.
        In pytorch implementation, we need to specify create_graph=True to create the graph of derivatives.
        '''

        alpha = torch.rand((len(expert_inputs_sa), 1), device=self.device)
        nu_inter = alpha * expert_inputs_sa + (1 - alpha) * rb_inputs_sa
        nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs
        nu_inter = torch.vstack((nu_inter, nu_next_inter))

        nu_grad = torch.autograd.grad(self.get_nu(nu_inter).sum(), nu_inter, create_graph=True)[0]
        nu_grad_penalty = torch.mean(
            torch.square(torch.norm(nu_grad, dim=-1, keepdim=True) - 1))

        nu_loss = loss + nu_grad_penalty * nu_reg
        log_zeta_loss = nu_loss + 0.1 * (self.log_zeta(expert_inputs_sa) ** 2).mean()
        pi_loss = -1 * loss + orthogonal_regularization(self.actor.trunk, self.device) # + self.alpha.detach() * log_prob.mean() #

        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss = (self.alpha *
        #               (-log_prob - self.target_entropy).detach()).mean()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()


        self.phi_weights_optimizer.zero_grad()
        nu_loss.backward(retain_graph=True, inputs=[self.phi_weights])
        self.log_zeta_optimizer.zero_grad()
        log_zeta_loss.backward(retain_graph=True, inputs=list(self.log_zeta.parameters()))


        self.actor_optimizer.zero_grad()
        pi_loss.backward(inputs=list(self.actor.parameters()))
        self.phi_weights_optimizer.step()
        self.log_zeta_optimizer.step()
        self.actor_optimizer.step()

        return {'loss': loss.item(),
                'nu_expert': expert_nu.mean().item(),
                'nu_rb': rb_nu.mean().item(),
                'nu_grad_penalty': nu_grad_penalty.item(),
                'actor_loss': pi_loss.item(),
                'policy_entropy': -1 * log_prob.mean().item(),
                'phi_weights_norm':torch.norm(self.phi_weights).item()
                # 'alpha_loss': alpha_loss,
                # 'alpha': self.alpha
                }

    def load(self, exp_dir):
        # normalization_consts = torch.load(os.path.join(exp_dir, 'normalization.pth'))
        # self.shift = normalization_consts['shift']
        # self.scale = normalization_consts['scale']
        self.phi.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_phi.pth')))
        self.mu.load_state_dict(torch.load(os.path.join(exp_dir, 'feature_mu.pth')))

    def save(self, exp_dir):
        # if 'rf' not in args.estimator:
        # torch.save({'shift': self.shift, 'scale': self.scale}, os.path.join(exp_dir, 'normalization.pth'))
        torch.save(self.phi.state_dict(), os.path.join(exp_dir, 'feature_phi.pth'))
        torch.save(self.mu.state_dict(), os.path.join(exp_dir, 'feature_mu.pth'))








