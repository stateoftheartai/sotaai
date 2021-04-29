# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the gym wrapper.'''

import unittest
import gym  # pylint: disable=unused-import
from sotaai.rl import gym_wrapper, load_environment
from gym_minigrid.wrappers import *  # pylint: disable=unused-wildcard-import, wildcard-import
import pybulletgym  # pylint: disable=unused-import
import procgen  # pylint: disable=unused-import

# It take too long to load
env_error = [
    'Defender-v0', 'Defender-v4', 'DefenderDeterministic-v0',
    'DefenderDeterministic-v4', 'DefenderNoFrameskip-v0',
    'DefenderNoFrameskip-v4', 'Defender-ram-v0', 'Defender-ram-v4',
    'Defender-ramDeterministic-v0', 'Defender-ramDeterministic-v4',
    'Defender-ramNoFrameskip-v0', 'Defender-ramNoFrameskip-v4'
]


class TestGymWrapper(unittest.TestCase):
  '''Test the wrapped fastai module.'''

  @unittest.SkipTest
  def test_load_environment(self):
    '''
      Make sure envs are returned.
    '''

    for name_env in gym_wrapper.ENVIRONMENTS:
      if name_env not in env_error:
        print(name_env)
        env = gym_wrapper.load_environment(name_env)
        self.assertEqual('gym' in str(type(env)), True)
      else:
        print(f'{name_env} skipped...')

  def test_abstraction_to_dict(self):
    '''
      Make sure dicts are returned.
    '''
    # with open('out.txt', 'w') as f:

    for name_env in gym_wrapper.ENVIRONMENTS:
      if name_env not in env_error:
        env = load_environment(name_env)
        print(f'Name: {name_env} - Source: {env.source}')
        print(env.type)
        # print(f'Name: {name_env} - Source: {env.source}', file=f)
        # print(env.raw.action_space, file=f)
        self.assertEqual(isinstance(env.to_dict(), dict), True)
      else:
        print(f'{name_env} skipped...')
