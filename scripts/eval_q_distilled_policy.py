import os
import re
import yaml
import json
import time
from typing import Optional, Tuple, List, Callable

import gymnasium as gym
import minari
import numpy as np
import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from agents.estimator.random_feature import RandomFeatureQNet
from utilities.dataset_utils import Normalizer
from utilities.dataset_utils import process_dataset_name


def load_saved_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing saved config at {cfg_path}")
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def load_env_from_config(cfg: dict, render=False):
    # Prefer single dataset_name; fallback to first in dataset_names
    dataset_name = cfg.get('dataset_name')
    dataset_names = cfg.get('dataset_names')
    dataset_name = process_dataset_name(dataset_name)
    if not dataset_name and dataset_names:
        dataset_name = dataset_names[0]
    if not dataset_name:
        raise ValueError("No dataset_name found in saved config")
    env_source = minari.load_dataset(dataset_name, download=True)
    env = env_source.recover_environment(render_mode="human" if render else None)
    return env


def find_q_checkpoint(run_dir: str, epoch: Optional[str | int]) -> str:
    if epoch is not None:
        fname = f"qnet_{epoch}.pth"
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    # Find latest numerically indexed qnet_*.pth
    pattern = re.compile(r"^qnet_(\d+)\.pth$")
    candidates = []
    for fn in os.listdir(run_dir):
        m = pattern.match(fn)
        if m:
            candidates.append((int(m.group(1)), fn))
    if not candidates:
        raise FileNotFoundError(f"No qnet_*.pth found in {run_dir}")
    candidates.sort(key=lambda x: x[0])
    return os.path.join(run_dir, candidates[-1][1])


def load_observation_normalizer(run_dir: str) -> Optional[Normalizer]:
    stats_path = os.path.join(run_dir, 'normalizer_stats.pth')
    if not os.path.exists(stats_path):
        return None
    try:
        stats = torch.load(stats_path, map_location='cpu')
        obs_stats = stats.get('observations')
        if obs_stats is None:
            return None
        return Normalizer(mean=obs_stats['mean'], std=obs_stats['std'])
    except Exception:
        return None


class QGreedyPolicy:
    def __init__(self,
                 qnet: RandomFeatureQNet,
                 action_space: gym.Space,
                 observation_normalizer: Optional[Normalizer],
                 device: torch.device,
                 action_samples: int = 512):
        self.qnet = qnet
        self.action_space = action_space
        self.obs_norm = observation_normalizer
        self.device = device
        self.action_samples = action_samples

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, gym.spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float32)
            high = np.asarray(self.action_space.high, dtype=np.float32)
            actions = np.random.uniform(low=low, high=high,
                                        size=(self.action_samples, low.shape[0])).astype(np.float32)
        else:
            raise NotImplementedError("Only Box action spaces are supported")

        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device).unsqueeze(0)
        if self.obs_norm is not None:
            obs_t = self.obs_norm(obs_t)
        obs_t = obs_t.repeat(self.action_samples, 1)
        act_t = torch.from_numpy(actions).to(self.device)
        q_values = self.qnet(obs_t, act_t).squeeze(-1)  # [K]
        best = int(torch.argmax(q_values).item())
        return actions[best]


def distill_policy_from_qnet(
        qnet: RandomFeatureQNet, env: gym.Env,
        observation_normalizer: Optional[Normalizer],
        beta_scale: float = 0.3) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Placeholder for policy distillation from Q-function.
    Return a policy object with an Callable[[obs], action] method.
    """
    from utilities.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
    beta_scale = beta_scale
    betas = get_beta_schedule(beta_schedule="cosine",
                              beta_start=0.0,
                              beta_end=1.0,
                              timesteps=20)
    betas = beta_scale * betas
    gd = GaussianDiffusion(betas,
                           model_mean_type="eps",
                           model_var_type="fixed-large",
                           loss_type="mse")
    print(env.action_space.shape)
    action_dim = env.action_space.shape[0]

    def policy_fn(obs, temperature: float = 1.0, best_of_n: int = 1):
        assert temperature > 1e-6, f"Temperature must be greater than 1e-6, but got {temperature}"
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
            obs = observation_normalizer(obs)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        def energy_func(action):
            if obs.ndim == action.ndim:
                return qnet(obs, action) / temperature
            elif obs.ndim == action.ndim - 1:
                # obs is [B, ...] and action is [N, B, action_dim]
                tilde_obs = obs[None, :, :].repeat(action.shape[0], 1, 1)
                return torch.vmap(qnet, (0, 0))(
                    tilde_obs, action).squeeze(dim=-1) / temperature
            else:
                raise ValueError(
                    f"Invalid obs and action dimensions: {obs.shape} and {action.shape}"
                )

        actions = gd.p_sample_idem_from_energy(energy_func,
                                               shape=(obs.shape[0],
                                                      action_dim))
        return actions.squeeze(dim=0).detach().cpu().numpy()

    return policy_fn


def evaluate_policy(env: gym.Env,
                    policy: Callable[[np.ndarray], np.ndarray],
                    episodes: int = 5,
                    max_steps: Optional[int] = None,
                    render: bool = False,
                    temperature: float = 1.0) -> List[float]:
    returns = []
    for e in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        steps = 0
        pbar = tqdm(range(max_steps), desc=f"Episode {e + 1}")
        for step in pbar:
            if render:
                env.render()
            action = policy(obs, temperature)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            print(f"Step {step + 1}, Reward: {reward}")
            steps += 1
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break
        returns.append(ep_ret)
        print(f"Temperature: {temperature}, Return: {ep_ret}")
    return returns


@hydra.main(config_path='../config', config_name='lrl_eval_q')
def run(args: DictConfig):
    if args.run_dir is None:
        raise ValueError(
            "Please provide run_dir via Hydra, e.g., run_dir=/path/to/training/run"
        )

    train_cfg = load_saved_config(args.run_dir)
    env = load_env_from_config(train_cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # device_str = args.device or train_cfg.get('device', 'cpu')
    device_str = args.device or "cpu"
    device = torch.device(device_str)

    qnet = RandomFeatureQNet(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dim=int(train_cfg['hidden_dim']),
                             hidden_depth=int(train_cfg['hidden_depth']),
                             mc_dim=int(train_cfg['feature_dim']),
                             device=device)
    ckpt_path = find_q_checkpoint(args.run_dir, args.epoch)
    qnet.load_state_dict(torch.load(ckpt_path, map_location=device))
    qnet.eval()

    obs_norm = load_observation_normalizer(args.run_dir)

    policy = distill_policy_from_qnet(qnet,
                                      env,
                                      obs_norm,
                                      beta_scale=args.beta_scale)
    if policy is None:
        policy = QGreedyPolicy(qnet=qnet,
                               action_space=env.action_space,
                               observation_normalizer=obs_norm,
                               device=device,
                               action_samples=args.action_samples)
    returns = []
    if isinstance(args.temperature, list):
        temperatures = args.temperature
    else:
        temperatures = [args.temperature]
    for temperature in temperatures:
        start = time.time()
        returns = evaluate_policy(env=env,
                                  policy=policy,
                                  episodes=args.episodes,
                                  max_steps=args.max_steps,
                                  render=args.render,
                                  temperature=temperature)
        duration = time.time() - start
        print(f"Evaluated {args.episodes} episodes in {duration:.2f}s")
        print(
            f"Temperature: {temperature}, Returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, min={np.min(returns):.2f}, max={np.max(returns):.2f}"
        )
        returns.append({
            'temperature': temperature,
            'returns': returns,
            'duration': duration,
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
        })
    # Persist a simple summary next to the run
    out = {
        'checkpoint': os.path.basename(ckpt_path),
        'episodes': int(args.episodes),
        'run_dir': args.run_dir,
        'returns': returns
    }
    # Save into the Hydra run directory (current working directory)
    with open('eval_q_policy.json', 'w') as f:
        json.dump(out, f, indent=2)

@hydra.main(config_path='../config', config_name='lrl_eval_q')
def test_policy_fn(args: DictConfig):
    if args.run_dir is None:
        raise ValueError(
            "Please provide run_dir via Hydra, e.g., run_dir=/path/to/training/run"
        )

    train_cfg = load_saved_config(args.run_dir)
    env = load_env_from_config(train_cfg, render=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # device_str = args.device or train_cfg.get('device', 'cpu')
    device_str = args.device or "cpu"
    device = torch.device(device_str)

    qnet = RandomFeatureQNet(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dim=int(train_cfg['hidden_dim']),
                             hidden_depth=int(train_cfg['hidden_depth']),
                             mc_dim=int(train_cfg['feature_dim']),
                             device=device)
    ckpt_path = find_q_checkpoint(args.run_dir, args.epoch)
    qnet.load_state_dict(torch.load(ckpt_path, map_location=device))
    qnet.eval()

    obs_norm = load_observation_normalizer(args.run_dir)

    policy = distill_policy_from_qnet(qnet, env, obs_norm, beta_scale=args.beta_scale)
    obs, _ = env.reset()
    if isinstance(args.temperature, list):
        temperatures = args.temperature
    else:
        temperatures = [args.temperature]
    for temperature in temperatures:
        action = policy(obs, temperature=temperature)
        q_value = qnet(
            torch.from_numpy(obs).float().unsqueeze(0),
            torch.from_numpy(action).float().unsqueeze(0)).squeeze(dim=-1)
        print(
            f"Temperature: {temperature}, Action: {action}, Q-value: {q_value.item()}"
        )
    env.close()




if __name__ == '__main__':
    test_policy_fn()
