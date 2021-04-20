# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Garage's library wrapper.'''
import torch
from torch.nn import functional as F

import tensorflow as tf
from garage.experiment import MetaEvaluator  # For MAMLVPG.
from garage.experiment.task_sampler import SetTaskSampler  # For MAMLVPG.
from garage.envs import normalize, GymEnv
from garage import Wrapper
from garage.sampler import LocalSampler
from garage.torch import algos
from garage.torch.algos import BC, DDPG, MAMLPPO, MAMLTRPO, MAMLVPG, MTSAC, PEARL, SAC, TRPO, VPG
from garage.torch.policies import DeterministicMLPPolicy, TanhGaussianMLPPolicy, GaussianMLPPolicy
from garage.tf.policies import CategoricalMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.q_functions import ContinuousMLPQFunction  # For DDPG.
from garage.replay_buffer import PathBuffer  # For DDPG.

SOURCE_METADATA = {
    'name': 'garage',
    'original_name': 'garage',
    'url': 'https://garage.readthedocs.io/en/latest/index.html'
}

MODELS = {
    'unknown': [
        'CEM', 'CMA-ES', 'REINFORCE', 'VPG', 'DDPG', 'DQN', 'ERWR', 'NPO',
        'PPO', 'REPS', 'TD3', 'TNPG', 'TRPO', 'MAML', 'RL2', 'PEARL', 'SAC',
        'MTSAC', 'MTPPO', 'MTTRPO', 'TaskEmbedding', 'BehavioralCloning'
    ],
}

hyper_parameters = {
    'hidden_sizes': [32, 32],
    'max_kl': 0.01,
    'gae_lambda': 0.97,
    'discount': 0.99,
    'n_epochs': 999,
    'batch_size': 1024,
}


def load_model(
    name: str,
    gym_env=None,
    name_env='CartPole-v1',
    meta_batch_size=20,
    discount=0.99,
    gae_lambda=1.,
    inner_lr=0.1,
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
  gym_env = normalize(GymEnv(name_env))
  env = Wrapper(gym_env)
  # if gym_env is not None:
  #   env = GarageEnv(env=gym_env)
  # else:
  #   env = GarageEnv(env_name=name_env)

  policy = DeterministicMLPPolicy(name='policy',
                                  env_spec=env.spec,
                                  hidden_sizes=[64, 64])

  if name == 'MAMLTRPO':

    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(
        lambda: normalize(GymEnv(name_env), expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=1)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=vfunc,
                    meta_batch_size=meta_batch_size,
                    discount=discount,
                    gae_lambda=gae_lambda,
                    task_sampler=task_sampler,
                    sampler=sampler,
                    inner_lr=inner_lr,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator)
    return algo
  elif name == 'MAMLPPO':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(
        lambda: normalize(GymEnv(name_env), expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=1)
    algo = MAMLPPO(env=env,
                   policy=policy,
                   value_function=vfunc,
                   meta_batch_size=meta_batch_size,
                   discount=discount,
                   gae_lambda=gae_lambda,
                   task_sampler=task_sampler,
                   sampler=sampler,
                   inner_lr=inner_lr,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)
    return algo
  elif name == 'MAMLVPG':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(
        lambda: normalize(GymEnv(name_env), expected_action_scale=10.))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=1)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)
    algo = MAMLVPG(env=env,
                   policy=policy,
                   value_function=vfunc,
                   meta_batch_size=meta_batch_size,
                   discount=discount,
                   gae_lambda=gae_lambda,
                   task_sampler=task_sampler,
                   sampler=sampler,
                   inner_lr=inner_lr,
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

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = SAC(env_spec=env.spec,
               policy=policy,
               qf1=qf1,
               qf2=qf2,
               sampler=sampler,
               gradient_steps_per_itr=1000,
               replay_buffer=replay_buffer)
    return algo
  elif name == 'DDPG':
    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    algo = DDPG(env_spec=env.spec,
                policy=policy,
                sampler=sampler,
                qf=qf,
                replay_buffer=replay_buffer)
    return algo
  if name == 'TRPO':
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=hyper_parameters['hidden_sizes'],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                sampler=sampler,
                value_function=value_function)

    return algo

  if name == 'VPG':

    tf.compat.v1.enable_eager_execution()

    policy = CategoricalMLPPolicy(name='policy',
                                  env_spec=env.spec,
                                  hidden_sizes=(32, 32))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    algo = VPG(env_spec=env.spec,
               policy=policy,
               value_function=vfunc,
               sampler=sampler,
               discount=0.99)

    return algo

  elif name == 'PPO':
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = getattr(algos, name)(env_spec=env.spec,
                                policy=policy,
                                value_function=vfunc)
    return algo
  elif name == 'BC':

    algo = BC(env_spec=env.spec, batch_size=20, learner=policy, source=policy)
  elif name == 'MTSAC':

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)
    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = MTSAC(env_spec=env.spec,
                 policy=policy,
                 qf1=qf1,
                 qf2=qf2,
                 num_tasks=100,
                 sampler=sampler,
                 eval_env=env,
                 gradient_steps_per_itr=200,
                 replay_buffer=replay_buffer)
  elif name == 'PEARL':
    num_train_tasks = 100
    num_test_tasks = 30
    latent_size = 5
    net_size = 300
    encoder_hidden_size = 200
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)

    # Create multi-task environment and sample tasks.
    env_start = GymEnv(name_env)
    env_sampler = SetTaskSampler(lambda: normalize(GymEnv(name_env)))
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(lambda: normalize(GymEnv(env_start)))

    # Instantiate networks.
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    pearl = PEARL(env=env,
                  inner_policy=inner_policy,
                  qf=qf,
                  vf=vf,
                  sampler=sampler,
                  num_train_tasks=num_train_tasks,
                  num_test_tasks=num_test_tasks,
                  latent_dim=latent_size,
                  encoder_hidden_sizes=encoder_hidden_sizes,
                  test_env_sampler=test_env_sampler)

    return pearl
  else:
    return {'name': name, 'source': 'garage'}
