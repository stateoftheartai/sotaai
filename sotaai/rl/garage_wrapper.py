# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Garage's library wrapper.'''
import torch
from torch.nn import functional as F

import numpy as np
from garage.experiment import MetaEvaluator  # For MAMLVPG.
from garage.experiment.task_sampler import SetTaskSampler  # For MAMLVPG.
from garage.envs import normalize, GarageEnv
from garage.torch import algos
from garage.torch.algos import BC, DDPG, MAMLPPO, MAMLTRPO, MAMLVPG, MTSAC, PEARL, SAC, TRPO, VPG
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.q_functions import ContinuousMLPQFunction  # For DDPG.
from garage.replay_buffer import PathBuffer  # For DDPG.
import akro
import gym

SOURCE_METADATA = {
    'name': 'garage',
    'original_name': 'garage',
    'url': 'https://garage.readthedocs.io/en/latest/index.html'
}

# MODELS = {
#     'unknown': [
#         'CEM', 'CMA-ES', 'REINFORCE', 'VPG', 'DDPG', 'DQN', 'ERWR', 'NPO',
#         'PPO', 'REPS', 'TD3', 'TNPG', 'TRPO', 'MAML', 'RL2', 'PEARL', 'SAC',
#         'MTSAC', 'MTPPO', 'MTTRPO', 'TaskEmbedding', 'BehavioralCloning'
#     ],
# }

MODELS = {
    'torch': [
        'BC', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG', 'MTSAC', 'PPO', 'SAC', 'TRPO',
        'VPG'
    ],
}

hyper_parameters = {
    'policy_lr': 1e-4,
    'qf_lr': 1e-3,
    'policy_hidden_sizes': [64, 64],
    'qf_hidden_sizes': [64, 64],
    'n_epochs': 500,
    'steps_per_epoch': 20,
    'n_exploration_steps': 100,
    'n_train_steps': 50,
    'discount': 0.9,
    'tau': 1e-2,
    'replay_buffer_size': int(1e6),
    'sigma': 0.2
}


def load_model(
    name: str,
    name_env='CartPole-v1',
):
  '''
  Load a model with specific configuration.
    Args:
      name (string): name of the model/algorithm.
      gym_env: ray gym environment
      name_env (strig): name of the gym environment
      meta_batch_size (int): Number of tasks sampled per batch.
      discount (float): Discount.
      gae_lambda (float): Lambda used for generalized advantage
              estimation.
      inner_lr (float): Adaptation learning rate.
  '''
  num_train_tasks = 100
  num_test_tasks = 30
  latent_size = 5
  net_size = 300
  max_path_length = 100
  encoder_hidden_size = 200
  encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                          encoder_hidden_size)

  # env = GarageEnv(env_name=name_env)
  env = GarageEnv(normalize(gym.make(name_env)))
  policy = None

  spec = env.spec

  if isinstance(spec.action_space, akro.Discrete):
    # policy = GaussianMLPPolicy(name='policy',
    #                            env_spec=env.spec,
    #                            hidden_sizes=[32, 32])
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[1, 1],
        hidden_nonlinearity=torch.nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )
    # policy = GaussianMLPPolicy(env_spec=spec, hidden_sizes=[32, 32])
  else:
    print('no entro al discrete')
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

  if name == 'MAMLTRPO':

    vfunc = GaussianMLPValueFunction(env_spec=env.spec, hidden_sizes=(32, 32))

    task_sampler = SetTaskSampler(
        lambda: GarageEnv(normalize(env, expected_action_scale=10.)))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)

    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=vfunc,
                    max_path_length=max_path_length,
                    meta_batch_size=5,
                    discount=0.99,
                    meta_evaluator=meta_evaluator,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1)
    return algo
  elif name == 'MAMLPPO':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(
        lambda: GarageEnv(normalize(env, expected_action_scale=10.)))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)
    algo = MAMLPPO(env=env,
                   policy=policy,
                   value_function=vfunc,
                   max_path_length=max_path_length,
                   meta_batch_size=20,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)
    return algo
  elif name == 'MAMLVPG':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(
        lambda: GarageEnv(normalize(env, expected_action_scale=10.)))

    max_path_length = 100
    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)
    algo = MAMLVPG(env=env,
                   policy=policy,
                   value_function=vfunc,
                   max_path_length=max_path_length,
                   meta_batch_size=20,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)
    return algo
  elif name == 'SAC':

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)
    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = SAC(env_spec=env.spec,
               policy=policy,
               qf1=qf1,
               qf2=qf2,
               gradient_steps_per_itr=1000,
               max_path_length=500,
               replay_buffer=replay_buffer)
    return algo
  elif name == 'DDPG':

    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = DDPG(env_spec=env.spec,
                policy=policy,
                max_path_length=max_path_length,
                qf=qf,
                replay_buffer=replay_buffer)
    return algo
  if name == 'TRPO':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = TRPO(env_spec=env.spec, policy=policy, value_function=vfunc)

    return algo

  if name == 'VPG':

    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = VPG(env_spec=env.spec, policy=policy, value_function=vfunc)

    return algo

  elif name == 'PPO':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = getattr(algos, name)(env_spec=env.spec,
                                policy=policy,
                                value_function=vfunc)
    return algo
  elif name == 'BC':

    algo = BC(env_spec=env.spec,
              batch_size=20,
              learner=policy,
              source=policy,
              max_path_length=max_path_length)
  elif name == 'MTSAC':

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)
    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = MTSAC(env_spec=env.spec,
                 policy=policy,
                 qf1=qf1,
                 qf2=qf2,
                 num_tasks=100,
                 max_path_length=max_path_length,
                 eval_env=env,
                 gradient_steps_per_itr=200,
                 replay_buffer=replay_buffer)
  elif name == 'PEARL':
    env_start = GarageEnv(env_name=name_env)
    env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(env_start)))
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(env_start)))

    # Instantiate networks.
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(env=env,
                  inner_policy=inner_policy,
                  qf=qf,
                  vf=vf,
                  num_train_tasks=num_train_tasks,
                  num_test_tasks=num_test_tasks,
                  latent_dim=latent_size,
                  encoder_hidden_sizes=encoder_hidden_sizes,
                  test_env_sampler=test_env_sampler)

    return pearl
  else:
    return {'name': name, 'source': 'garage'}
