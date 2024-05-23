import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal 
import os

import torch.nn.init as init
import numpy as np
from utils import util 

# from utils.util import unpack_batch, RunningMeanStd
from utils.util import unpack_batch
from networks.policy import GaussianPolicy
from networks.vae import Encoder, Decoder, GaussianFeature
from agent.sac.sac_agent import SACAgent
# from main import DEVICE

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class Critic(nn.Module):
    """
    Critic with random fourier features
    """
    def __init__(
        self,
        feature_dim,
        num_noise=20, 
        hidden_dim=256,
        device=torch.device('cpu')
        ):

        super().__init__()
        self.device = device
        self.num_noise = num_noise
        self.noise = torch.randn(
            [self.num_noise, feature_dim], requires_grad=False, device=self.device)

        # Q1
        self.l1 = nn.Linear(feature_dim, hidden_dim) # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.l4 = nn.Linear(feature_dim, hidden_dim) # random feature
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)


    def forward(self, mean, log_std):
        """
        """
        std = log_std.exp()
        batch_size, d = mean.shape 
    
        x = mean[:, None, :] + std[:, None, :] * self.noise
        x = x.reshape(-1, d)

        q1 = F.elu(self.l1(x)) #F.relu(self.l1(x))
        q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q1 = F.elu(self.l2(q1)) #F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.elu(self.l4(x)) #F.relu(self.l4(x))
        q2 = q2.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        q2 = F.elu(self.l5(q2)) #F.relu(self.l5(q2))
        # q2 = self.l3(q2) #is this wrong?
        q2 = self.l6(q2)

        return q1, q2




class RLNetwork(nn.Module):
    """
    An abstract class for neural networks in reinforcement learning (RL). In deep RL, many algorithms
    use DP algorithms. For example, DQN uses two neural networks: a main neural network and a target neural network.
    Parameters of a main neural network is periodically copied to a target neural network. This RLNetwork has a
    method called soft_update that implements this copying.
    """
    def __init__(self):
        super(RLNetwork, self).__init__()
        self.layers = []

    def forward(self, *x):
        return x

    def soft_update(self, target_nn: nn.Module, update_rate: float):
        """
        Update the parameters of the neural network by
            params1 = self.parameters()
            params2 = target_nn.parameters()

            for p1, p2 in zip(params1, params2):
                new_params = update_rate * p1.data + (1. - update_rate) * p2.data
                p1.data.copy_(new_params)

        :param target_nn:   DDPGActor used as explained above
        :param update_rate: update_rate used as explained above
        """

        params1 = self.parameters()
        params2 = target_nn.parameters()
        
        #bug? 
        for p1, p2 in zip(params1, params2):
            new_params = update_rate * p1.data + (1. - update_rate) * p2.data
            p1.data.copy_(new_params)

    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





#currently hardcoding sa_dim
#this is a  Q function
class RFQCritic(RLNetwork):
    def __init__(self, sa_dim = 4, embedding_dim = 4, n_neurons = 256):
        super().__init__()
        self.n_layers = 1
        self.feature_dim = n_neurons

        self.embed = nn.Linear(sa_dim, embedding_dim)

        # fourier_feats1 = nn.Linear(sa_dim, n_neurons)
        fourier_feats1 = nn.Linear(embedding_dim,n_neurons)
        init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0,2*np.pi)
        # init.zeros_(fourier_feats.bias)
        fourier_feats1.weight.requires_grad = False
        fourier_feats1.bias.requires_grad = False
        self.fourier1 = fourier_feats1 #unnormalized, no cosine/sine yet



        # fourier_feats2 = nn.Linear(sa_dim, n_neurons)
        fourier_feats2 = nn.Linear(embedding_dim,n_neurons)
        init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0,2*np.pi)
        fourier_feats2.weight.requires_grad = False
        fourier_feats2.bias.requires_grad = False
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear( n_neurons, 1) #try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False #weight is the only thing we update
        self.output1 = layer1


        layer2 = nn.Linear( n_neurons, 1) #try default scaling
        # init.uniform_(layer2.weight, -3e-3,3e-3) 
        # init.uniform_(layer2.weight, -3e-4,3e-4)
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False #weight is the only thing we update
        self.output2= layer2


    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([states,actions],axis = -1)
        # print("x initial norm", torch.linalg.norm(x))
        # x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
        # x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
        # x = self.bn(x)
        x = self.embed(x) #use an embedding layer
        # print("x norm after embedding", torch.linalg.norm(x))
        # print("layer1 norm", torch.linalg.norm(self.output1.weight))
        # x = F.relu(x)
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        x1 = torch.cos(x1)
        x2 = torch.cos(x2)
        # x1 = torch.cos(x)
        # x2 = torch.sin(x)
        # x = torch.cat([x1,x2],axis = -1)
        # x = torch.div(x,1./np.sqrt(2 * self.feature_dim))
        x1 = torch.div(x1,1./np.sqrt(self.feature_dim)) #why was I multiplyigng?
        x2 = torch.div(x2,1./np.sqrt(self.feature_dim))
        return self.output1(x1), self.output2(x2)





#currently hardcoding s_dim
#this is a  V function
class RFVCritic(RLNetwork):
    def __init__(self, s_dim = 3, embedding_dim = -1, rf_num = 256,sigma = 0.0, learn_rf = False, **kwargs):
        super().__init__()
        self.n_layers = 1
        self.feature_dim = rf_num

        self.sigma = sigma

        if embedding_dim != -1:
            self.embed = nn.Linear(s_dim, embedding_dim)
        else: #we don't add embed in this case
            embedding_dim = s_dim
            self.embed = nn.Linear(s_dim,s_dim)
            init.eye_(self.embed.weight)
            init.zeros_(self.embed.bias)
            self.embed.weight.requires_grad = False
            self.embed.bias.requires_grad = False

        # fourier_feats1 = nn.Linear(sa_dim, n_neurons)
        fourier_feats1 = nn.Linear(embedding_dim,self.feature_dim)
        # fourier_feats1 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats1.weight, std = 1./self.sigma)
            # pass
        else:
            init.normal_(fourier_feats1.weight)
        init.uniform_(fourier_feats1.bias, 0,2*np.pi)
        # init.zeros_(fourier_feats.bias)
        fourier_feats1.weight.requires_grad = learn_rf
        fourier_feats1.bias.requires_grad = learn_rf
        self.fourier1 = fourier_feats1 #unnormalized, no cosine/sine yet



        fourier_feats2 = nn.Linear(embedding_dim, self.feature_dim)
        # fourier_feats2 = nn.Linear(s_dim,n_neurons)
        if self.sigma > 0:
            init.normal_(fourier_feats2.weight, std = 1./self.sigma)
            # pass
        else:
            init.normal_(fourier_feats2.weight)
        init.uniform_(fourier_feats2.bias, 0,2*np.pi)
        fourier_feats2.weight.requires_grad = learn_rf
        fourier_feats2.bias.requires_grad = learn_rf
        self.fourier2 = fourier_feats2

        layer1 = nn.Linear( self.feature_dim, 1) #try default scaling
        # init.uniform_(layer1.weight, -3e-3,3e-3) #weight is the only thing we update
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False #weight is the only thing we update
        self.output1 = layer1


        layer2 = nn.Linear( self.feature_dim, 1) #try default scaling
        # init.uniform_(layer2.weight, -3e-3,3e-3) 
        # init.uniform_(layer2.weight, -3e-4,3e-4)
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False #weight is the only thing we update
        self.output2 = layer2

        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm1.bias.requires_grad = False

        self.norm = nn.LayerNorm(self.feature_dim)
        self.norm.bias.requires_grad = False

        self.robust_feature = kwargs.get('robust_feature', False)
        if self.robust_feature:
            self.pertubation_phi = nn.Parameter(torch.normal(mean=0, std=1.,size=(self.feature_dim, )))
            self.pertubation_w = copy.deepcopy(layer2)


    def forward(self, states: torch.Tensor):
        x = states
        # print("x initial norm",torch.linalg.norm(x))
        # x = torch.cat([states,actions],axis = -1)
        # x = F.batch_norm(x) #perform batch normalization (or is dbn better?)
        # x = (x - torch.mean(x, dim=0))/torch.std(x, dim=0) #normalization
        # x = self.bn(x)
        x = self.embed(x) #use an embedding layer
        # print("x embedding norm", torch.linalg.norm(x))
        # x = F.relu(x)
        x1 = self.fourier1(x)
        x2 = self.fourier2(x)
        if self.robust_feature:
            x1 = torch.cos(x1) + self.pertubation_phi
            x2 = torch.cos(x2) + self.pertubation_phi
        else:
            x1 = torch.cos(x1)
            x2 = torch.cos(x2)
        # x1 = torch.cos(x)
        # x2 = torch.sin(x)
        # x = torch.cat([x1,x2],axis = -1)
        # x = torch.div(x,1./np.sqrt(2 * self.feature_dim))
        # if self.sigma > 0:
        #   x1 = torch.multiply(x1,1./np.sqrt(2 * np.pi * self.sigma))
        #   x2 = torch.multiply(x2,1./np.sqrt(2 * np.pi * self.sigma)) 
        # x1 = torch.div(x1,np.sqrt(self.feature_dim/2))
        # x2 = torch.div(x2,np.sqrt(self.feature_dim/2))
        # x1 = torch.div(x1,1./self.feature_dim)
        # x2 = torch.div(x2,1./self.feature_dim)
        # change to layer norm
        x1 = 10. * self.norm1(x1)
        x2 = 10. * self.norm(x2)
        # print("x1 norm", torch.linalg.norm(x1,axis = 1))
        # x = torch.relu(x)
        if self.robust_feature:
            return self.output1(x1) + self.pertubation_w(x1), self.output2(x2) + self.pertubation_w(x2)
        else:
            return self.output1(x1), self.output2(x2)

    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)




#currently hardcoding s_dim
#this is a  V function
#buffer: if not None, use samples from buffer to compute nystrom features
class nystromVCritic(RLNetwork):
    def __init__(self,
                 s_dim=3,
                 s_low=np.array([-1, -1, -8]),
                 feat_num=256,
                 sigma=0.0,
                 buffer=None,
                 learn_rf = False,
                 **kwargs):
        super().__init__()
        self.n_layers = 1
        self.feature_dim = feat_num
        self.sigma = sigma
        # s_high = -s_low

        self.s_low = kwargs.get('obs_space_low')
        self.s_high = kwargs.get('obs_space_high')
        self.s_dim = kwargs.get('obs_space_dim')
        self.s_dim = self.s_dim[0] if (not isinstance(self.s_dim, int)) else self.s_dim
        # self.feature_dim = kwargs.get('random_feature_dim')
        self.sample_dim = kwargs.get('nystrom_sample_dim')
        # self.sigma = kwargs.get('sigma')
        self.dynamics_type = kwargs.get('dynamics_type')
        self.sin_input = kwargs.get('dynamics_parameters').get('sin_input')
        self.dynamics_parameters = kwargs.get('dynamics_parameters')

        eval = kwargs.get('eval', False)
        if not eval:
            np.random.seed(kwargs.get('seed'))
            # create nystrom feats
            self.nystrom_samples1 = np.random.uniform(self.s_low, self.s_high, size=(self.sample_dim, self.s_dim))
            # self.nystrom_samples1 = np.random.uniform([-0.3, -0.03, 0.3, -0.03, 0.955, 0., -0.03],
            #                                           [0.3, 0.03, 0.7, 0.03, 1., 0.295, 0.03], size=(self.sample_dim, self.s_dim))
            # self.nystrom_samples2 = np.random.uniform(s_low,s_high,size = (feat_num, s_dim))

            if sigma > 0.0:
                self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2. * sigma ** 2))
            else:
                self.kernel = lambda z: np.exp(-np.linalg.norm(z) ** 2 / (2.))
            K_m1 = self.make_K(self.nystrom_samples1, self.kernel)
            print('start eig')

            [eig_vals1, S1] = np.linalg.eig(K_m1)  # numpy linalg eig doesn't produce negative eigenvalues... (unlike torch)

            # truncate top k eigens
            argsort = np.argsort(eig_vals1)[::-1]
            eig_vals1 = eig_vals1[argsort]
            S1 = S1[:, argsort]
            eig_vals1 = np.clip(eig_vals1, 1e-8, np.inf)[:self.feature_dim]
            self.eig_vals1 = torch.from_numpy(eig_vals1).float().to(self.device)
            self.S1 = torch.from_numpy(S1[:, :self.feature_dim]).float().to(self.device)
            self.nystrom_samples1 = torch.from_numpy(self.nystrom_samples1).to(self.device)
        else:
            self.nystrom_samples1 = torch.zeros((self.sample_dim, self.s_dim))
            self.eig_vals1 = torch.ones([self.feature_dim,])
            self.S1 = torch.zeros([self.s_dim, self.feature_dim])

        layer1 = nn.Linear(self.feature_dim, 1)  # try default scaling
        init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False  # weight is the only thing we update
        self.output1 = layer1

        layer2 = nn.Linear(self.feature_dim, 1)  # try default scaling
        init.zeros_(layer2.bias)
        layer2.bias.requires_grad = False  # weight is the only thing we update
        self.output2 = layer2

        self.norm = nn.LayerNorm(self.feature_dim)
        self.norm.bias.requires_grad = False

    def make_K(self, samples,kernel):
        print('start cal K')
        m,d = samples.shape
        K_m = np.empty((m,m))
        for i in np.arange(m):
            for j in np.arange(m):
                K_m[i,j] = kernel(samples[i,:] - samples[j,:])
        return K_m

    def kernel_matrix_numpy(self, x1, x2):
        print('start cal K')
        dx2 = np.expand_dims(x1, axis=1) - np.expand_dims(x2,
                                                          axis=0)  # will return the kernel matrix of k(x1, x2) with symmetric kernel.
        if self.sigma > 0.0:
            K_x2 = np.exp(-np.linalg.norm(dx2, axis=2) ** 2 / (2. * self.sigma ** 2))
        else:
            K_x2 = np.exp(-np.linalg.norm(dx2, axis=2) ** 2 / (2.))
        return K_x2


    def forward(self, states: torch.Tensor):
        x1 = self.nystrom_samples1.unsqueeze(0) - states.unsqueeze(1)
        K_x1 = torch.exp(-torch.linalg.norm(x1,axis = 2)**2/2).float()
        phi_all1 = (K_x1 @ (self.S1)) @ torch.diag((self.eig_vals1.clone() + 1e-8) ** (-0.5))
        # phi_all1 = self.norm(phi_all1)
        phi_all1 = 50. * phi_all1
        phi_all1 = phi_all1.to(torch.float32)
        return self.output1(phi_all1), self.output2(phi_all1)


    def get_norm(self):
        l1_norm = torch.norm(self.output1)
        l2_norm = torch.norm(self.output2)
        return (l1_norm, l2_norm)


class DoubleQCritic(nn.Module):
  """Critic network, employes double Q-learning."""
  def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
    super().__init__()

    self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
    self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

    self.outputs = dict()
    self.apply(util.weight_init)

  def forward(self, obs, action):
    assert obs.size(0) == action.size(0)

    obs_action = torch.cat([obs, action], dim=-1)
    q1 = self.Q1(obs_action)
    q2 = self.Q2(obs_action)

    self.outputs['q1'] = q1
    self.outputs['q2'] = q2

    return q1, q2


class RFSACAgent(SACAgent):
    """
    SAC with random features
    """
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            action_space, 
            # lr=1e-3,
            # lr = 5e-4,
            # lr = 3e-4,
            lr = 3e-4,
            discount=0.99, 
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            sigma = 0.0,
            rf_num = 256,
            learn_rf = False,
            use_nystrom = False,
            replay_buffer = None,
            # feature_tau=0.001,
            # feature_dim=256, # latent feature dim
            # use_feature_target=True, 
            extra_feature_steps=1,
            **kwargs
            ):

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            **kwargs
        )
        # self.feature_dim = feature_dim
        # self.feature_tau = feature_tau
        # self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps

        # self.encoder = Encoder(state_dim=state_dim, 
        #       action_dim=action_dim, feature_dim=feature_dim).to(device)
        # self.decoder = Decoder(state_dim=state_dim,
        #       feature_dim=feature_dim).to(device)
        # self.f = GaussianFeature(state_dim=state_dim, 
        #       action_dim=action_dim, feature_dim=feature_dim).to(device)
        
        # if use_feature_target:
        #   self.f_target = copy.deepcopy(self.f)
        # self.feature_optimizer = torch.optim.Adam(
        #   list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.f.parameters()),
        #   lr=lr)

        # self.critic = RFCritic().to(device)
        # self.rfQcritic = RFQCritic().to(device)
        # self.rfQcritic_target = copy.deepcopy(self.rfQcritic)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(
        #   self.critic.parameters(), lr=lr, betas=[0.9, 0.999])
        # self.rfQcritic_optimizer = torch.optim.Adam(
        #   self.rfQcritic.parameters(), lr=lr, betas=[0.9, 0.999])

        # self.critic = DoubleQCritic(
        #   obs_dim = state_dim,
        #   action_dim = action_dim,
        #   hidden_dim = hidden_dim,
        #   hidden_depth = 2,
        #   ).to(device)
        # self.critic = RFQCritic().to(device)

        if use_nystrom == False: #use RF
            self.rf_num = rf_num
            self.critic = RFVCritic(s_dim=state_dim, sigma = sigma, rf_num = rf_num, learn_rf = learn_rf, **kwargs).to(self.device)
        else: #use nystrom
            feat_num = rf_num
            self.critic = nystromVCritic(sigma = sigma, feat_num = feat_num, buffer = replay_buffer, learn_rf = learn_rf,  **kwargs).to(self.device)
        # self.critic = Critic().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.robust_feature = kwargs.get('robust_feature', False)

        # separate update targets if we have robust updates
        if self.robust_feature:
            self.pertubation_optimizer = torch.optim.Adam([
                *self.critic.pertubation_w.parameters(),
                self.critic.pertubation_phi
            ], lr=lr, betas=[0.9, 0.999])
            self.critic_optimizer = torch.optim.Adam([
                *self.critic.output1.parameters(),
                *self.critic.output2.parameters()
            ], lr=lr, betas=[0.9, 0.999])
        else:
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr, betas=[0.9, 0.999])
        self.args = kwargs
        self.dynamics_type = kwargs.get('dynamics_type')
        self.sin_input = kwargs.get('dynamics_parameters').get('sin_input')
        if self.dynamics_type == 'Pendulum':
            self.dynamics = self.f_star_3d

        elif self.dynamics_type == 'Pendubot':
            if self.sin_input:
                self.dynamics = self.pendubot_f_6d
            else:
                raise NotImplementedError
        elif self.dynamics_type == 'CartPoleContinuous':
            if self.sin_input:
                self.dynamics = self.cartpole_f_5d
            else:
                self.dynamics = self.cartpole_f_4d
        elif self.dynamics_type == 'Quadrotor2D':
            if self.sin_input:
                self.dynamics = self.quadrotor_f_star_7d
            else:
                self.dynamics = self.quadrotor_f_star_6d
        elif self.dynamics_type == 'CartPendulum':
            if self.sin_input:
                self.dynamics = self.cart_pendulum_5d
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if learn_rf and use_nystrom:
            self.eig_optimizer = torch.optim.Adam([self.critic.eig_vals1, self.critic.S1],
                                                  lr = 1e-4)
        self.learn_rf = learn_rf






    # def feature_step(self, batch):
    #   """
    #   Feature learning step

    #   KL between two gaussian p1 and p2:

    #   log sigma_2 - log sigma_1 + sigma_1^2 (mu_1 - mu_2)^2 / 2 sigma_2^2 - 0.5
    #   """
    #   # ML loss
    #   z = self.encoder.sample(
    #       batch.state, batch.action, batch.next_state)
    #   x, r = self.decoder(z)
    #   s_loss = 0.5 * F.mse_loss(x, batch.next_state)
    #   r_loss = 0.5 * F.mse_loss(r, batch.reward)
    #   ml_loss = r_loss + s_loss

    #   # KL loss
    #   mean1, log_std1 = self.encoder(
    #       batch.state, batch.action, batch.next_state)
    #   mean2, log_std2 = self.f(batch.state, batch.action)
    #   var1 = (2 * log_std1).exp()
    #   var2 = (2 * log_std2).exp()
    #   kl_loss = log_std2 - log_std1 + 0.5 * (var1 + (mean1-mean2)**2) / var2 - 0.5
        
    #   loss = (ml_loss + kl_loss).mean()

    #   self.feature_optimizer.zero_grad()
    #   loss.backward()
    #   self.feature_optimizer.step()

    #   return {
    #       'vae_loss': loss.item(),
    #       'ml_loss': ml_loss.mean().item(),
    #       'kl_loss': kl_loss.mean().item(),
    #       's_loss': s_loss.mean().item(),
    #       'r_loss': r_loss.mean().item()
    #   }

    #inputs are tensors
    # def get_reward(self, states,action):
    #     th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
    #     thdot = states[:,2]
    #     action = torch.reshape(action, (action.shape[0],))
    #     # print("th shape", th.shape)
    #     # print("thdot shape", thdot.shape)
    #     # print('action shape', action.shape)
    #     th = self.angle_normalize(th)
    #     reward = -(th**2 + 0.1* thdot**2 + 0.01*action**2)
    #     return torch.reshape(reward,(reward.shape[0],1))

    def angle_normalize(self,th):
        return((th + np.pi) % (2 * np.pi)) -np.pi
    
    def f_star_2d(self,states,action,g = 10.0,m = 1.,l=1.,max_a = 2.,max_speed = 8.,dt = 0.05):
        th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
        thdot = states[:,2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action,-max_a,max_a)
        newthdot = thdot +(3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed,max_speed)
        newth = th + newthdot * dt
        new_states = torch.empty((states.shape[0],2))
        new_states[:,0] = self.angle_normalize(newth)
        new_states[:,1] = newthdot
        return new_states

    #this returns cos(th), sin(th), thdot
    def f_star_3d(self,states,action,g = 10.0,m = 1.,l = 1.,max_a = 2.,max_speed = 8.,dt = 0.05):
        th = torch.atan2(states[:,1],states[:,0]) #1 is sin, 0 is cosine 
        thdot = states[:,2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action,-max_a,max_a)
        newthdot = thdot +(3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed,max_speed)
        newth = th + newthdot * dt
        return torch.vstack([torch.cos(newth), torch.sin(newth), newthdot]).T

    def __quad_action_preprocess(self, action):
        action = 0.075 * action + 0.075  # map from -1, 1 to 0.0 - 0.15
        # print(action)
        return action


    def quadrotor_f_star_6d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
        # dot_states = torch.empty_like(states)
        # dot_states[:, 0] = states[:, 1]
        # dot_states[:, 1] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4]))
        # dot_states[:, 2] = states[:, 3]
        # dot_states[:, 3] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g
        # dot_states[:, 4] = states[:, 5]
        # dot_states[:, 5] = 1 / 2 / Iyy * (action[:, 1] - action[:, 0])
        action = self.__quad_action_preprocess(action)
        def dot_states(states):
            dot_states = torch.vstack([states[:, 1],
                                 1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4])),
                                 states[:, 3],
                                 1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g,
                                 states[:, 5],
                                 1 / 2 / Iyy * (action[:, 1] - action[:, 0]) * 0.025
                                 ]).T
            return dot_states

        k1 = dot_states(states)
        k2 = dot_states(states + dt / 2 * k1)
        k3 = dot_states(states + dt / 2 * k2)
        k4 = dot_states(states + dt * k3)

        return states + dt / 6 * (k1 + k2 + k3 + k4)

    def quadrotor_f_star_7d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
        action = self.__quad_action_preprocess(action)
        theta = torch.atan2(states[:, -2], states[:, -3])
        state_6d = torch.concat([states[:,:-3], torch.reshape(theta, [-1, 1]), states[:, -1:]], dim=1)
        new_states = self.quadrotor_f_star_6d(state_6d, action)
        new_theta = new_states[:, -2]
        new_cos = torch.reshape(torch.cos(new_theta), [-1, 1])
        new_sin = torch.reshape(torch.sin(new_theta), [-1, 1])
        new_states_7d = torch.concat([new_states[:, :-2], new_cos, new_sin, new_states[:, -1:]], dim=-1)
        return new_states_7d

    def cartpole_f_4d(self, states, action, ):
        """

        :param states: # x, x_dot, theta, theta_dot
        :param action: Force applied to the cart
        :return: new states
        """
        masscart = 1.0
        masspole = 0.1
        length = 0.5
        total_mass = masspole + masscart
        polemass_length = masspole * length
        dt = 0.02
        gravity = 9.81
        new_states = torch.empty_like(states, device=self.device)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        theta = states[:, 2]
        theta_dot = states[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force = torch.squeeze(10. * action)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = 1. / total_mass * (
                force + polemass_length * theta_dot ** 2 * sintheta
        )
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        new_states[:, 1] = states[:, 1] + dt * xacc
        new_states[:, 3] = theta_dot + dt * thetaacc
        return new_states

    def cartpole_f_5d(self, states, action,):
        """

        :param states: # x, x_dot, sin_theta, cos_theta, theta_dot
        :param action: Force applied to the cart
        :return: new states
        """
        masscart = 1.0
        masspole = 0.1
        length = 0.5
        total_mass = masspole + masscart
        polemass_length = masspole * length
        dt = 0.02
        gravity = 9.81
        new_states = torch.empty_like(states, device=self.device)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        costheta = states[:, -3]
        sintheta = states[:, -2]
        theta_dot = states[:, -1]
        theta = torch.atan2(sintheta, costheta)
        new_theta = theta + dt * theta_dot
        new_states[:, -3] = torch.cos(new_theta)
        new_states[:, -2] = torch.sin(new_theta)
        # new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        # theta = states[:, 2]

        force = torch.squeeze(10. * action)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = 1. / total_mass * (
                force + polemass_length * theta_dot ** 2 * sintheta
        )
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        new_states[:, 1] = states[:, 1] + dt * xacc
        new_states[:, 4] = theta_dot + dt * thetaacc
        return new_states

    def pendubot_f_6d(self, states, action):
        dt = 0.05
        new_states = torch.empty_like(states, device=self.device)
        cos_theta1, sin_theta1 = states[:, 0], states[:, 1]
        cos_theta2, sin_theta2 = states[:, 2], states[:, 3]
        theta1_dot, theta2_dot = states[:, 4], states[:, 5]
        theta1 = torch.atan2(sin_theta1, cos_theta1)
        theta2 = torch.atan2(sin_theta2, cos_theta2)
        new_theta1 = theta1 + dt * theta1_dot
        new_theta2 = theta2 + dt * theta2_dot
        new_states[:, 0] = torch.cos(new_theta1)
        new_states[:, 1] = torch.sin(new_theta1)
        new_states[:, 2] = torch.cos(new_theta2)
        new_states[:, 3] = torch.sin(new_theta2)

        d1 = 0.089252
        d2 = 0.027630
        d3 = 0.023502
        d4 = 0.011204
        d5 = 0.002938
        g = 9.81

        self.d4 = d4
        self.d5 = d5

        m11 = d1 + d2 + 2 * d3 * torch.cos(theta2)
        m21 = d2 + d3 * torch.cos(theta2)
        # m12 = d2 + d3 * torch.cos(theta2)
        m22 = d2

        mass_matrix = torch.empty((states.shape[0], 2, 2), device=self.device)
        mass_matrix[:, 0, 0] = m11
        mass_matrix[:, 0, 1] = m21
        mass_matrix[:, 1, 0] = m21
        mass_matrix[:, 1, 1] = m22

        self.mass_matrix = mass_matrix

        # mass_inv = torch.empty_like(mass_matrix, device=self.device)
        # mass_matrix[:, 0, 0] = m22
        # mass_matrix[:, 0, 1] = - m21
        # mass_matrix[:, 1, 0] = - m21
        # mass_matrix[:, 1, 1] = m11


        # mass_matrix = np.array([[m11, m12],
        #                         [m21, m22]])

        c_matrix = torch.empty((states.shape[0], 2, 2), device=self.device)
        c11 = -1. * d3 * torch.sin(theta2) * theta2_dot
        c12 = -d3 * torch.sin(theta2) * (theta2_dot + theta1_dot)
        c21 = d3 * torch.sin(theta2) * theta1_dot
        c22 = torch.zeros_like(theta1)
        c_matrix[:, 0, 0] = c11
        c_matrix[:, 0, 1] = c12
        c_matrix[:, 1, 0] = c21
        c_matrix[:, 1, 1] = c22

        g1 = d4 * torch.cos(theta2) * g + d5 * g * torch.cos(theta1 + theta2)
        g2 = d5 * torch.cos(theta1 + theta2) * g

        g_vec = torch.empty((states.shape[0], 2, 1), device=self.device)
        g_vec[:, 0, 0] = g1
        g_vec[:, 1, 0] = g2

        action = torch.hstack([action, torch.zeros_like(action)])[:, :, np.newaxis]
        # acc = torch.reciprocal(m11 * m22 - m21 ** 2 + 1e-6).unsqueeze(1).unsqueeze(2) * mass_inv @ (action -
        #                                                         torch.matmul(c_matrix, states[:, -2:][:, :, np.newaxis]) - g_vec)
        acc = torch.linalg.solve(mass_matrix,
                                 action - torch.matmul(c_matrix, states[:, -2:][:, :, np.newaxis]) - g_vec)
        new_states[:, 4] = theta1_dot + dt * torch.squeeze(acc[:, 0])
        new_states[:, 5] = theta2_dot + dt * torch.squeeze(acc[:, 1])

        return new_states

    def cart_pendulum_5d(self, states, action):
        dt = 0.02
        g, M, m, b, I, l = 10, 0.5, 0.2, 0.1, 0.006, 0.3
        force_mag = 0.4
        new_states = torch.empty_like(states, device=self.device)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        costheta = states[:, -3]
        sintheta = states[:, -2]
        theta = torch.atan2(sintheta, costheta)
        thdot = states[:, -1]
        xdot = states[:, 1]
        action = force_mag * torch.squeeze(action)

        a11, a22 = M + m, I + m * l ** 2
        a12 = m * l * costheta
        detA = I * (M + m) + m * l ** 2 * M + m ** 2 * l ** 2 * sintheta ** 2
        b1 = m * l * thdot ** 2 * sintheta - b * xdot + action
        b2 = -m * g * l * sintheta
        dxdot = (a22 * b1 - a12 * b2) / detA
        dthdot = (-a12 * b1 + a11 * b2) / detA
        # dx = xdot.unsqueeze(-1)
        # dth = thdot.unsqueeze(-1)
        new_states[:, 1] = xdot + dxdot * dt
        new_theta = theta + dt * thdot
        new_states[:, 2] = torch.sin(new_theta)
        new_states[:, 3] = torch.cos(new_theta)
        new_states[:, 4] = thdot + dt * dthdot
        return new_states


    def _get_energy_error(self, obs, action, ke=1.5):
        assert self.dynamics_type == 'Pendubot'
        dot_theta = obs[:, -2:][:, :, np.newaxis]  # batch, 2, 1
        dot_theta_t = obs[:, -2:][:, np.newaxis]  # batch, 1, 2
        cos_theta1, sin_theta1 = obs[:, 0], obs[:, 1]
        cos_theta2, sin_theta2 = obs[:, 2], obs[:, 3]
        sin_theta1_plus_theta2 = torch.multiply(sin_theta1, cos_theta2) + torch.multiply(cos_theta1, sin_theta2)

        kinetic_energy = torch.squeeze(torch.matmul(torch.matmul(dot_theta_t, self.mass_matrix), dot_theta))
        potential_energy = self.d4 * 9.81 * sin_theta1 + self.d5 * 9.81 * sin_theta1_plus_theta2
        energy_on_top = (self.d4 + self.d5) * 9.81
        energy_error = kinetic_energy + potential_energy - energy_on_top

        return ke * energy_error ** 2

    def angle_normalize(self, th):
        return ((th + np.pi) % (2 * np.pi)) - np.pi

    def get_reward(self, obs, action):
        if self.dynamics_type == 'Pendulum':
            assert obs.shape[1] == 3
            th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
            thdot = obs[:, 2]
            action = torch.reshape(action, (action.shape[0],))
            th = self.angle_normalize(th)
            reward = -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * action ** 2)

        elif self.dynamics_type == 'Quadrotor2D':
            if isinstance(self.args.get('dynamics_parameters').get('stabilizing_target'), list):
                stabilizing_target = torch.tensor(self.args.get('dynamics_parameters').get('stabilizing_target'),
                                                  device=self.device)
            else:
                stabilizing_target = self.args.get('dynamics_parameters').get('stabilizing_target').to(self.device)
            if self.sin_input is False:
                assert obs.shape[1] == 6
                state_error = obs - stabilizing_target
                reward = -(torch.sum( torch.multiply(torch.tensor([1., 0., 1., 0., 0., 0.], device=self.device),
                                                     state_error ** 2), dim=1) ) # + torch.sum(0.1 * action ** 2, dim=1)
                # if self.args.get('dynamics_parameters').get('reward_exponential'):
                #     reward = torch.exp(reward)
            else:
                assert obs.shape[1] == 7
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :4], th, obs[:, -1:]])
                state_error = obs - stabilizing_target
                reward = -(torch.sum(torch.multiply(torch.tensor([1., 0., 1., 0., 0., 0.], device=self.device),
                                                    state_error ** 2), dim=1) ) # + torch.sum(0.1 * action ** 2, dim=1)

        elif self.dynamics_type == 'CartPoleContinuous':
            if self.sin_input is False:
                reward = -(torch.sum(obs ** 2, dim=1) + torch.sum(0.01 * action ** 2, dim=1))
            else:
                assert obs.shape[1] == 5
                th = torch.atan2(obs[:, -2], obs[:, -3]) # torch.unsqueeze(, dim=1)  # -2 is sin, -3 is cos
                th = torch.where(obs[:, -3] >= 0., th, th + torch.pi )
                x = obs[:, 0]
                obs_no_theta = torch.hstack([obs[:, :-3], obs[:, -1:]])
                reward = -(torch.sum(torch.multiply(torch.tensor([0.01, 0.001, 0.001], device=self.device), obs_no_theta), dim=1)
                           # + torch.sin(th) ** 2 + (torch.cos(th) - 1) ** 2
                           + (torch.remainder(th + torch.pi, 2 * torch.pi) - torch.pi) ** 2
                           + torch.sum(0.001 * action ** 2, dim=1)) \
                         # - torch.where(torch.abs(x) > 5., 1000 * torch.ones_like(x), torch.zeros_like(x))

        elif self.dynamics_type == 'CartPendulum':
            if self.sin_input is False:
                raise NotImplementedError
            else:
                assert obs.shape[1] == 5
                th = torch.atan2(obs[:, -2], obs[:, -3]) # torch.unsqueeze(, dim=1)  # -2 is sin, -3 is cos
                # th = torch.where(obs[:, -3] >= 0., th, th + torch.pi)
                ## arctan only return [-pi/2, pi/2].
                reward = - ((torch.remainder(th, 2 * torch.pi) - torch.pi) ** 2)

        elif self.dynamics_type == 'Pendubot':
            if self.sin_input:
                assert obs.shape[1] == 6
                th1dot = obs[:, 4]
                th2dot = obs[:, 5]
                if self.args.get('dynamics_parameters').get('reward_type') == 'lqr':
                    if self.args.get('dynamics_parameters').get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 +
                                        0.01 * th2 ** 2 + 0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    elif self.args.get('dynamics_parameters').get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 +
                                        0.01 * (sin_th2) ** 2 + 0.01 * (cos_th2 - 1.) ** 2 +
                                        0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    else:
                        raise NotImplementedError
                elif self.args.get('dynamics_parameters').get('reward_type') == 'energy':
                    if self.args.get('dynamics_parameters').get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 + self._get_energy_error(obs, action))
                    elif self.args.get('dynamics_parameters').get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 + self._get_energy_error(obs,
                                                                                                                    action))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
        reward_scale = self.args.get('dynamics_parameters').get('reward_scale')
        reward = reward_scale * reward
        # exponent
        if self.args.get('dynamics_parameters').get('reward_exponential'):
            reward = torch.exp(reward)
        return torch.reshape(reward, (reward.shape[0], 1))

    def update_actor_and_alpha(self, batch):
        """
        Actor update step
        """
        # dist = self.actor(batch.state, batch.next_state)
        dist = self.actor(batch.state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # if self.use_feature_target:
        #   mean, log_std = self.f_target(batch.state, action)
        # else:
        #   mean, log_std = self.f(batch.state, action)
        # q1, q2 = self.critic(mean, log_std)
        # q = torch.min(q1, q2)
        # q = self.discount * self.critic(batch.next_state) + batch.reward 
        # q = batch.reward 
        # q1,q2 = self.rfQcritic(batch.state,batch.action)
        # q1,q2 = self.critic(batch.state,action) #not batch.action!!!
        # q = torch.min(q1, q2)
        # q = q1 #try not using q1, q1
        reward = self.get_reward(batch.state,action) #use reward in q-fn
        q1,q2 = self.critic(self.dynamics(batch.state,action))
        q = self.discount * torch.min(q1,q2) + reward

        actor_loss = ((self.alpha) * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        info = {'actor_loss': actor_loss.item()}

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                                        (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            info['alpha_loss'] = alpha_loss 
            info['alpha'] = self.alpha 

        return info

    def pertubation_step(self, batch):
        state, action, next_state, reward, done = unpack_batch(batch)
        q1, q2 = self.critic(self.dynamics(state, action))
        q = torch.min(q1,q2)
        # penalty_loss = 100 * (F.relu(torch.norm(self.critic.pertubation_phi) - np.sqrt(self.rf_num / 5.))
        #                       + F.relu(torch.norm(self.critic.pertubation_w.weight) - np.sqrt(self.rf_num / 5.)))


        pertubation_loss = torch.mean(q) #3 + penalty_loss
        self.pertubation_optimizer.zero_grad()
        pertubation_loss.backward()
        self.pertubation_optimizer.step()
        phi_norm = torch.norm(self.critic.pertubation_phi).item()
        w_norm = torch.norm(self.critic.pertubation_w.weight).item()
        if phi_norm > self.args.get('robust_radius'):
            print('assign phi')
            self.critic.pertubation_phi = nn.Parameter(1 / phi_norm * self.args.get('robust_radius') * self.critic.pertubation_phi )
        if w_norm > self.args.get('robust_radius'):
            print('assign w')
            self.critic.pertubation_w.weight = nn.Parameter(1 / w_norm * self.args.get('robust_radius') * self.critic.pertubation_w.weight)
        info = {'pertubation_loss': pertubation_loss.item(),
                'phi_norm': phi_norm,
                'w_norm': w_norm
                }
        return info

    def critic_step(self, batch):
        """
        Critic update step
        """         
        # state, action, reward, next_state, next_action, next_reward,next_next_state, done = unpack_batch(batch)
        state,action,next_state,reward,done = unpack_batch(batch)
        
        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
            # if self.use_feature_target:
            #   mean, log_std = self.f_target(state, action)
            #   next_mean, next_log_std = self.f_target(next_state, next_action)
            # else:
            #   mean, log_std = self.f(state, action)
            #   next_mean, next_log_std = self.f(next_state, next_action)
            next_q1, next_q2 = self.critic_target(self.dynamics(next_state,next_action))
            next_q = torch.min(next_q1,next_q2)-  self.alpha * next_action_log_pi
            next_reward = self.get_reward(next_state,next_action) #reward for new s,a
            target_q = next_reward + (1. - done) * self.discount * next_q

        q1,q2 = self.critic(self.dynamics(state,action))
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        # if self.args.get('robust_feature', False):
        #     penalty_loss = 100 * (F.relu(torch.norm(self.critic.pertubation_phi) - 3.)
        #                           + F.relu(torch.norm(self.critic.pertubation_w.weight) - 3.))
        #     q_loss = q1_loss + q2_loss + penalty_loss
        # else:
        q_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        if self.learn_rf and isinstance(self.critic, nystromVCritic) and self.steps % 10 == 0:
            q1, q2 = self.critic(self.dynamics(state, action))
            q1_loss = F.mse_loss(target_q, q1)
            q2_loss = F.mse_loss(target_q, q2)
            q_loss = q1_loss + q2_loss
            self.eig_optimizer.zero_grad()
            q_loss.backward()
            self.eig_optimizer.step()

        info = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item(),
            'layer_norm_weights_norm': self.critic.norm.weight.norm(),
            }

        # if self.args.get('robust_feature', False):
        #     info.update({'penalty_loss': penalty_loss.item(),})

        dist = {
            'td_error': (torch.min(q1, q2) - target_q).cpu().detach().clone().numpy(),
            'q': torch.min(q1, q2).cpu().detach().clone().numpy()
        }

        info.update({'critic_dist': dist})

        if self.learn_rf and isinstance(self.critic, nystromVCritic):
            info.update({'largest_eig': self.critic.eig_vals1.max().detach().clone().item(),
                         'smallest_eig': self.critic.eig_vals1.min().detach().clone().item()},
                        )

        return info

    def update_feature_target(self):
        for param, target_param in zip(self.f.parameters(), self.f_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def train(self, buffer, batch_size):
        """
        One train step
        """
        self.steps += 1

        # Feature step
        # for _ in range(self.extra_feature_steps+1):
        #   batch = buffer.sample(batch_size)
        #   feature_info = self.feature_step(batch)

        #   # Update the feature network if needed
        #   if self.use_feature_target:
        #       self.update_feature_target()

        batch = buffer.sample(batch_size)

        if self.robust_feature and self.steps % 5 == 0:
            pertubation_info = self.pertubation_step(batch)

        # Acritic step
        critic_info = self.critic_step(batch)
        # critic_info = self.rfQcritic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Update the frozen target models
        self.update_target()

        infos = {
            # **feature_info,
            **critic_info,
            **actor_info,
        }

        if self.robust_feature and self.steps % 5 == 0:
            infos.update({**pertubation_info})

        return infos


    
