# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the gym wrapper.'''

import unittest
from sotaai.rl import gym_wrapper, garage_wrapper, load_model
from sotaai.rl.abstractions import RlModel

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

EROR_MODELS = ['DDPG', 'PEARL']


class TestGarageWrapper(unittest.TestCase):
  '''Test the wrapped garage module.'''

  @unittest.SkipTest
  def test_load_model(self):
    '''
      Make sure garage envs are returned.
    '''
    for model_name in MODELS:
      if not model_name in EROR_MODELS:
        for task in gym_wrapper.LIST_ENVIRONMENTS:
          for env_name in gym_wrapper.LIST_ENVIRONMENTS[task]:
            print(f'Model name: {model_name}\t---\t Env name: {env_name}')
            garage_wrapper.load_model(name=model_name, name_env=env_name)

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
