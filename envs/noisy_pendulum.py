__credits__ = ["Carlos Luis"]

import os.path
from os import path
from typing import Optional

import numpy as np
import time

import gym
from gym import spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from scipy.stats import norm, truncnorm

DEFAULT_X = np.pi
DEFAULT_Y = 1.0
EPS = 1e-6


#adds gaussian noise
class noisyPendulumEnv(gym.Env):
    """
       ### Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |


    ### Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ### Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ### Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ### Episode Truncation

    The episode truncates at 200 time steps.

    ### Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```
    gym.make('Pendulum-v1', g=9.81)
    ```

    ### Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, sigma = 0.0,
        max_episode_steps = 200,euler = False):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.2
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.sigma = sigma
        self.max_episode_steps = max_episode_steps
        self.euler = euler

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u + np.random.normal(scale = self.sigma)) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        if self.euler == True:
            newth = th + thdot * dt
        else:
            newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])
        # self.state += np.random.normal(size = (2,), scale = self.sigma)

        self.counter += 1

        if self.counter == self.max_episode_steps:
            done = True
        else:
            done = False

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, done, {}

    def reset(self, *, init_state = None, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.

        if init_state is None:
            self.state = self.np_random.uniform(low=low, high=high)
        else:
            self.state = init_state
        self.last_u = None

        self.counter = 0

        if self.render_mode == "human":
            self.render()
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def visualize(self, init_state, cmd, seed = None, dt: float =None):
        """
        Visualize the movement associated to a sequence of control variables
        :param cmd: sequence of controls to be applied on the system given as an numpy array
        :param dt: time step to visualize the movement (default is to use the time step defined in the environment)
        seed: random seed for noise in env (if any)
        """
        if dt is None:
            dt = self.dt
        self.render_mode = "human"
        self.reset(init_state = init_state)
        print("self.state", self.state)
        t = 0
        np.random.seed(seed)
        for ctrl in cmd:
            ctrl = np.array([ctrl])
            self.render()
            time.sleep(dt)
            self.step(ctrl)
            print("self.action (time %d)"%t, ctrl)
            print("self.state (time %d)"%t, self.state)
            t += 1
        self.render()
        self.close()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class ParallelNoisyPendulum(noisyPendulumEnv):
    """
    Data collection environment for parallel environments
    """

    def __init__(self, sigma=0.0, rollout_batch_size=512, sin_cos_obs=False, prob='conditional'):
        super().__init__(sigma=sigma)
        self.rollout_batch_size = rollout_batch_size
        self.sin_cos_obs = sin_cos_obs
        self.state_dim = 3 if sin_cos_obs else 2
        self.action_dim = 1
        self.truncnorm_action = truncnorm(-0.5 * self.max_torque, 0.5 * self.max_torque)
        self.truncnorm_th = truncnorm(-np.pi +EPS, np.pi - EPS)
        self.truncnorm_thdot = truncnorm(-0.5 * self.max_speed, 0.5 * self.max_speed)
        self.prob_label_type = prob

    def sample(self,
               batches=200,
               seed = 0,
               store_path=None,
               non_zero_initial=False,
               dist = 'uniform_theta'):
        ptr = 0
        dataset = np.zeros((batches * self.rollout_batch_size, 2 * self.state_dim + self.action_dim))
        prob_set = np.zeros((batches * self.rollout_batch_size,))
        for i in range(batches):
            np.random.seed(seed)
            if dist == 'uniform_theta':
                th, thdot = self.uniform_theta_sample(non_zero_initial)
            elif dist == 'uniform_sin_theta':
                th, thdot = self.unifrom_sin_theta_sample()
            elif dist == 'gaussian':
                th, thdot, prob_st = self.truncated_gaussian_sample()
            else:
                raise NotImplementedError
            action, prob_at = self.truncated_gaussian_action_sample()
            new_th, new_thdot = self.batch_step(th, thdot, action)

            # add noise
            new_state = np.vstack((new_th, new_thdot)).T
            assert self.sigma != 0.0
            noise = self.get_noise()
            noisy_new_state = new_state + noise

            noisy_new_state = self.clip_states(noisy_new_state)
            corrected_noise = noisy_new_state - new_state
            obs_t = self.get_obs(th, thdot)
            obs_tp1 = self.get_obs(noisy_new_state[:, 0], noisy_new_state[:, 1])
            condi_prob = self.get_prob(corrected_noise)

            batch = np.hstack([obs_t, action[:, np.newaxis], obs_tp1])
            dataset[ptr:ptr + self.rollout_batch_size] = batch
            if self.prob_label_type == 'conditional':
                prob_set[ptr:ptr + self.rollout_batch_size] = condi_prob
            elif self.prob_label_type == 'joint':
                prob_set[ptr:ptr + self.rollout_batch_size] = prob_st * prob_at * condi_prob
            ptr += self.rollout_batch_size
            seed += 1

        if store_path is not None and isinstance(store_path, str):
            np.save(os.path.join(store_path, 'tran_pendulum.npy'), dataset)
            np.save(os.path.join(store_path, 'prob_pendulum.npy'), prob_set)

        return dataset, prob_set


    def unifrom_sin_theta_sample(self,):
        initial_sin_theta = np.random.uniform(low=-1, high=1, size=(self.rollout_batch_size,))
        rademacher = np.random.choice([-1, 1], size=(self.rollout_batch_size,))
        initial_cos_theta = np.sqrt(1 - initial_sin_theta ** 2) * rademacher
        theta_dot = np.random.uniform(-0.5 * self.max_speed, 0.5 * self.max_speed, size=(self.rollout_batch_size,))
        th = np.arctan2(initial_sin_theta, initial_cos_theta)
        return th, theta_dot

    def uniform_action_sample(self):
        return np.random.uniform(low=-self.max_torque,
                                       high=self.max_torque,
                                       size=self.rollout_batch_size)

    def truncated_gaussian_sample(self):
        th = self.truncnorm_th.rvs(size=self.rollout_batch_size)
        prob_th = self.truncnorm_th.pdf(th)
        thdot = self.truncnorm_thdot.rvs(size=self.rollout_batch_size)
        prob_thdot = self.truncnorm_thdot.pdf(thdot)
        return th, thdot, prob_th * prob_thdot

    def truncated_gaussian_action_sample(self):
        actions = self.truncnorm_action.rvs(size=self.rollout_batch_size)
        prob = self.truncnorm_thdot.pdf(actions)
        return actions, prob

    def get_noise(self):
        # if self.sigma != 0.0:
        noise = np.random.normal(scale=self.sigma * self.dt, size=(self.rollout_batch_size, 2))
        # new_state = new_state + noise
        return noise

    def get_prob(self, noise):
        prob = np.prod(norm.pdf(noise, loc=np.zeros([2, ]), scale=self.sigma * self.dt * np.ones([2, ])), axis=1)
        return prob

    def batch_step(self, th, thdot, action):
        theta_ddot = 3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * action

        new_th = th + thdot * self.dt
        new_thdot = thdot + theta_ddot * self.dt

        return new_th, new_thdot

    def get_obs(self, th, thdot):

        if self.sin_cos_obs:
            return np.vstack([np.cos(th), np.sin(th), thdot]).T
        else:
            # th = angle_normalize(th)
            return np.vstack([th, thdot]).T

    def clip_states(self, states):
        th, thdot = states[:, 0], states[:, 1]
        thdot = np.clip(thdot, -self.max_speed, self.max_speed)  # for numerical stability when calculating log prob
        return np.vstack([th, thdot]).T

    def uniform_theta_sample(self, non_zero_initial=False):
        if non_zero_initial:
            # for numerical stability
            initial_state = np.random.uniform(low=[np.pi / 2 + EPS, -0.5 * self.max_speed],
                                              high=[np.pi * 3 / 2 - EPS, 0.5 * self.max_speed],
                                              size=(self.rollout_batch_size, 2))
        else:
            initial_state = np.random.uniform(low=[0, -0.5 * self.max_speed],
                                              high=[2 * np.pi, 0.5 * self.max_speed],
                                              size=(self.rollout_batch_size, 2))
        th, thdot = initial_state[:, 0], initial_state[:, 1]
        return th, thdot

    def get_true_marginal(self, st_at, dist='unifrom_theta'):
        if dist == 'uniform_theta':
            sin_theta = st_at[:, 1]
            true_marginal = (1 / np.pi / 16) * np.reciprocal(np.sqrt(1 - sin_theta ** 2) + 1e-8)  # arcsine distribution
        elif dist == 'uniform_sin_theta':
            true_marginal = 1 / (2 * np.pi * 16) * np.ones((len(st_at)))
        else:
            raise NotImplementedError
        return true_marginal


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi