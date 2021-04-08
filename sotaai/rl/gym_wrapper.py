# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''OpenAI's Gym library wrapper.'''
import gym
from gym import envs
from gym_minigrid.wrappers import *  # pylint: disable=wildcard-import, unused-wildcard-import
import pybulletgym  # pylint: disable=unused-import
import procgen  # pylint: disable=unused-import

SOURCE_METADATA = {
    'name': 'gym',
    'original_name': 'Gym',
    'url': 'https://gym.openai.com'
}

ENVIRONMENTS = envs.registry.env_specs


def load_environment(name: str) -> dict:
  '''Return a gym environment

  Args:
    name: the environment name in string
  '''
  if name in envs.registry.env_specs:
    env = gym.make(name)
    return env
  else:
    raise NotImplementedError(f'Not implemented {name} environment')
