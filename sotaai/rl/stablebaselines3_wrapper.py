# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Stable Baseline 3's library wrapper.'''

SOURCE_METADATA = {
    'name': 'stablebaselines3',
    'original_name': 'Stable Baselines3',
    'url': 'https://stable-baselines3.readthedocs.io/en/master/index.html'
}

MODELS = {
    'Box': [
        'A2C',
        # 'DDPG',
        'HER',
        'PPO',
        'SAC',
        'TD3'
    ],
    'Discrete': ['A2C', 'DQN', 'HER', 'PPO'],
    'MultiDiscrete': ['A2C', 'PPO'],
    'MultiBinary': ['A2C', 'PPO'],
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'garage'}
