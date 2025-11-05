import d3rlpy
import minari
import os
import numpy as np
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.constants import LoggingStrategy
import datetime

reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()
observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()

# prepare dataset
# dataset = minari.load_dataset('mujoco/halfcheetah/simple-v0')
dataset, env = d3rlpy.datasets.get_minari('mujoco/halfcheetah/expert-v0')
# dataset, env = d3rlpy.datasets.get_pendulum()

reward_scaler.fit_with_transition_picker(dataset.episodes,
                                         dataset.transition_picker)
observation_scaler.fit_with_transition_picker(dataset.episodes,
                                              dataset.transition_picker)
print(reward_scaler.mean, reward_scaler.std)
print(observation_scaler.mean, observation_scaler.std)

# prepare algorithm
# alg = d3rlpy.algos.CQLConfig().create(device='cuda')
alg = d3rlpy.algos.IQLConfig(
    reward_scaler=reward_scaler,
    observation_scaler=observation_scaler).create(device='cuda')
# start training

wandb_adapter = d3rlpy.logging.WanDBAdapterFactory(project='d3rlpy_lrl')

alg.fit(dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        experiment_name=f'with_reward_scaler_obs_scaler_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
        evaluators={'eval/env': EnvironmentEvaluator(env)},
        logging_strategy=LoggingStrategy.STEPS,
        logger_adapter=wandb_adapter)

# cql.build_with_env(env)
# cql = d3rlpy.load_learnable(
#     '/home/haitong/PycharmProjects/low_rank_learning/d3rlpy_logs/CQL_20251103164210/model_10000.d3'
# )

# obs, _ =env.reset()
# done = False
# i = 0
# while not done and i < 1000:
#     obs, reward, terminated, truncated, _ = env.step(
#         cql.predict(np.expand_dims(obs, 0)).squeeze(0))
#     env.render()
#     i += 1
#     done = terminated or truncated
#     print(f"Reward: {reward}")
