# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the gym wrapper.'''

import unittest
from sotaai.rl import gym_wrapper, garage_wrapper, load_model
from sotaai.rl.abstractions import RlModel
# from garage.trainer import Trainer
# from garage.experiment import SnapshotConfig, LocalRunner, deterministic
from garage.envs import GarageEnv, normalize
import gym
import akro
env_error = [
    'Defender-v0', 'Defender-v4', 'DefenderDeterministic-v0',
    'DefenderDeterministic-v4', 'DefenderNoFrameskip-v0',
    'DefenderNoFrameskip-v4', 'Defender-ram-v0', 'Defender-ram-v4',
    'Defender-ramDeterministic-v0', 'Defender-ramDeterministic-v4',
    'Defender-ramNoFrameskip-v0', 'Defender-ramNoFrameskip-v4'
]

MODELS = [
    'BC', 'DDPG', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG', 'MTSAC', 'PEARL', 'PPO',
    'SAC', 'TRPO', 'VPG'
]

EROR_MODELS = ['DDPG', 'PEARL', 'BC', 'MAMLPPO', 'MAMLTRPO', 'MAMLVPG', 'MTSAC']


class TestGarageWrapper(unittest.TestCase):
  '''Test the wrapped garage module.'''

  @unittest.SkipTest
  def test_load_model(self):
    '''
      Make sure garage envs are returned.
    '''

    # hyper_parameters = {
    #     'policy_lr': 1e-4,
    #     'qf_lr': 1e-3,
    #     'policy_hidden_sizes': [64, 64],
    #     'qf_hidden_sizes': [64, 64],
    #     'n_epochs': 500,
    #     'steps_per_epoch': 20,
    #     'n_exploration_steps': 100,
    #     'n_train_steps': 50,
    #     'discount': 0.9,
    #     'tau': 1e-2,
    #     'replay_buffer_size': int(1e6),
    #     'sigma': 0.2
    # }
    for model_name in MODELS:
      if not model_name in EROR_MODELS:
        for task in gym_wrapper.LIST_ENVIRONMENTS:
          for env_name in gym_wrapper.LIST_ENVIRONMENTS[task]:
            env = GarageEnv(normalize(gym.make(env_name)))
            if not isinstance(env.spec.action_space, akro.Discrete):
              print(f'Model name: {model_name}\t---\t Env name: {env_name}')
              algo = garage_wrapper.load_model(name=model_name,
                                               name_env=env_name)
              print(algo)

  @unittest.SkipTest
  def test_abstraction_load_model(self):
    '''
      Make sure garage envs are returned.
    '''
    for model_name in MODELS:
      if not model_name in EROR_MODELS:
        for task in gym_wrapper.LIST_ENVIRONMENTS:
          for env_name in gym_wrapper.LIST_ENVIRONMENTS[task]:
            print(f'Model name: {model_name}\t---\t Env name: {env_name}')
            rl_model = load_model(name=model_name, name_env=env_name)
            self.assertEqual(RlModel, type(rl_model))
            self.assertEqual(rl_model.source, 'garage')
