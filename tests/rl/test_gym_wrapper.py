# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the gym wrapper.'''

import unittest
import gym  # pylint: disable=unused-import
from sotaai.rl import gym_wrapper
from gym_minigrid.wrappers import *  # pylint: disable=unused-wildcard-import, wildcard-import
import pybulletgym  # pylint: disable=unused-import
import procgen  # pylint: disable=unused-import


class TestGymWrapper(unittest.TestCase):
  '''Test the wrapped fastai module.'''

  def test_load_environment(self):
    '''
      Make sure envs are returned.
    '''

    # It take too long to load
    env_error = [
        'Defender-v0', 'Defender-v4', 'DefenderDeterministic-v0',
        'DefenderDeterministic-v4', 'DefenderNoFrameskip-v0',
        'DefenderNoFrameskip-v4', 'Defender-ram-v0', 'Defender-ram-v4',
        'Defender-ramDeterministic-v0', 'Defender-ramDeterministic-v4',
        'Defender-ramNoFrameskip-v0', 'Defender-ramNoFrameskip-v4'
    ]

    for name_env in gym_wrapper.ENVIRONMENTS:
      if name_env not in env_error:
        print(name_env)
        env = gym_wrapper.load_environment(name_env)
        self.assertEqual('gym' in str(type(env)), True)
      else:
        print(f'{name_env} skipped...')