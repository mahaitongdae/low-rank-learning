import torch
import numpy as np
import os
from scipy.stats import norm
class MVN(object):

    def __init__(self, rollout_batch_size=512):
        self.cov = np.array([[1.0, 0.5],[0.5, 1.0]])
        # self.dist =
        self.rollout_batch_size = rollout_batch_size
        # self.conditional_dist =
        self.state_dim = 1
        self.action_dim = 0


    def get_conditional_dist(self, samples):
        x1 = samples[:, 0]
        x2 = samples[:, 1]
        # conditional distribution is
        # \mu_{x2 | x1} = \mu_2 + E_{21} E_{11}^{-1}(x1-\mu_1) = 1 / 2 x1
        # Cov_{x2 | x1} = Cov_{22} - Cov_{21} Cov_{11}^{-1}Cov_{12} = 1 - 1/2*1/2 = 3/4
        pdf = 4 / (3 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x2 - 0.5 * x1 / (3 / 4)) ** 2)
        return pdf


    def sample(self,batches=200, seed = 0, store_path=None):
        ptr = 0
        dataset = np.zeros((batches * self.rollout_batch_size, 2))
        prob_set = np.zeros((batches * self.rollout_batch_size,))
        for i in range(batches):
            np.random.seed(seed)
            samples = np.random.multivariate_normal(mean=np.zeros(2),
                                                    cov=self.cov,
                                                    size=(self.rollout_batch_size,))
            dataset[ptr: ptr+self.rollout_batch_size] = samples

            prob = self.get_conditional_dist(samples=samples)
            prob_set[ptr: ptr+self.rollout_batch_size] = prob
            ptr += self.rollout_batch_size
            seed += 1

        if store_path is not None and isinstance(store_path, str):
            np.save(os.path.join(store_path, 'tran_normal.npy'), dataset)
            np.save(os.path.join(store_path, 'prob_normal.npy'), prob_set)

        return dataset, prob_set

class MVNUniform(MVN):

    def sample(self,batches=200, seed = 0, store_path=None):
        ptr = 0
        dataset = np.zeros((batches * self.rollout_batch_size, 2))
        prob_set = np.zeros((batches * self.rollout_batch_size,))
        for i in range(batches):
            np.random.seed(seed)
            x1 = np.random.uniform(low=-1,
                                   high=1,
                                   size=(self.rollout_batch_size,))

            x2 = np.random.multivariate_normal(mean=0.5 * x1,
                                                cov= 3 / 4 * np.eye(self.rollout_batch_size))
            samples = np.vstack([x1, x2]).T
            dataset[ptr: ptr + self.rollout_batch_size] = samples

            prob = self.get_conditional_dist(samples=samples)
            prob_set[ptr: ptr + self.rollout_batch_size] = prob
            ptr += self.rollout_batch_size
            seed += 1

        if store_path is not None and isinstance(store_path, str):
            np.save(os.path.join(store_path, 'tran_normal.npy'), dataset)
            np.save(os.path.join(store_path, 'prob_normal.npy'), prob_set)

        return dataset, prob_set
