import collections
import numpy as np
import torch
# from main import DEVICE


Batch = collections.namedtuple(
	'Batch',
	# ['state', 'action', 'reward', 'next_state', 'next_action', 'next_reward', 'next_next_state','done']
	['state', 'action', 'reward', 'next_state','done']
	)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size,1))
		self.next_state = np.zeros((max_size, state_dim))
		self.next_action = np.zeros((max_size,action_dim))
		self.next_reward = np.zeros((max_size, 1))
		self.next_next_state = np.zeros((max_size,state_dim))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device(device)


	def addv2(self, state, action, reward, next_state, next_action, next_reward,next_next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.next_action[self.ptr] = next_action
		self.next_reward[self.ptr] = next_reward
		self.next_next_state[self.ptr] = next_next_state
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def add(self, state, action,  next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def samplev2(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			next_action=torch.FloatTensor(self.next_action[ind]).to(self.device),
			next_reward = torch.FloatTensor(self.next_reward[ind]).to(self.device),
			next_next_state = torch.FloatTensor(self.next_next_state[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			# next_action=torch.FloatTensor(self.next_action[ind]).to(self.device),
			# next_reward = torch.FloatTensor(self.next_reward[ind]).to(self.device),
			# next_next_state = torch.FloatTensor(self.next_next_state[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)



