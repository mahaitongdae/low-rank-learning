import gym
import numpy as np
import os
import json
import torch
from agents.imitator import ValuDICEImitator

def eval_policy(policy, eval_env, eval_episodes=100, render=False, seed=0):
	"""
	Eval a policy
	"""
	ep_rets = []
	avg_len = 0.
	for i in range(eval_episodes):
		ep_ret = 0.
		# eval_env.seed(i)
		state, _ = eval_env.reset(seed=i + seed)
		done = False
		# print("eval_policy state", state)
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			ep_ret += reward
			avg_len += 1
			if render:
				eval_env.render()
		ep_rets.append(ep_ret)

	avg_ret = np.mean(ep_rets)
	std_ret = np.std(ep_rets)
	avg_len /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: avg eplen {avg_len}, avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}")
	print("---------------------------------------")
	return avg_len, avg_ret, std_ret, ep_rets

def get_controller(log_path):
	from agents.actor import DiagGaussianActor
	from envs.wrappers import create_il_env
	# try:
	with open(os.path.join(log_path, 'args.json'), 'rb') as f:
		kwargs = json.load(f)

	# except:
	#     exit()
	env = gym.make(kwargs['env_id'][:-1] + '4')
	env = create_il_env(env)

	actor = DiagGaussianActor(obs_dim=env.observation_space.shape,
							  action_dim=env.action_space.shape,
							  hidden_dim=kwargs['hidden_dim'],
							  hidden_depth=kwargs['hidden_depth'],
							  log_std_bounds=[-5., 2.])
	agent = ValuDICEImitator(env.observation_space.shape, env.action_space.shape)
	actor.load_state_dict(torch.load(log_path + "/actor_last.pth"))
	agent.actor = actor
	agent.device = torch.device("cpu")
	return agent, env

if __name__ == '__main__':
	agent, env = get_controller(log_path="/home/haitong/PycharmProjects/low_rank_learning/log/HalfCheetah-v2/repr_value_dice/2024-09-06-01-12-13")
	eval_policy(agent, env, eval_episodes=1)