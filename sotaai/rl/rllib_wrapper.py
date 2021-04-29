# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''RLlib's library wrapper.'''

SOURCE_METADATA = {
    'name': 'rllib',
    'original_name': 'RLlib',
    'url': 'https://docs.ray.io/en/master/rllib.html'
}

MODELS = {
    'discrete': [
        'A2C', 'A3C', 'ARS', 'BC', 'ES', 'DQN', 'Rainbow', 'APEX-DQN', 'IMPALA',
        'MARWIL', 'PG', 'PPO', 'APPO', 'R2D2', 'SAC', 'SlateQ', 'LinUCB',
        'LinTS', 'AlphaZero', 'QMIX', 'MADDPG', 'Curiosity'
    ],
    'continuous': [
        'A2C',
        'A3C',
        'ARS',
        'BC',
        'CQL',
        'ES',
        # 'DDPG',
        'TD3',
        'APEX-DDPG',
        'Dreamer',
        'IMPALA',
        'MAML',
        'MARWIL',
        'MBMPO',
        'PG',
        'PPO',
        'APPO',
        'SAC',
        'MADDPG'
    ],
    'multi-agent': [
        'A2C',
        'A3C',
        'BC',
        # 'DDPG',
        'TD3',
        'APEX-DDPG',
        'DQN',
        'Rainbow',
        'APEX-DQN',
        'IMPALA',
        'MARWIL',
        'PG',
        'PPO',
        'APPO',
        'R2D2',
        'SAC',
        'LinUCB',
        'LinTS',
        'QMIX',
        'MADDPG',
        'Curiosity'
    ],
    'unknown': [
        'ParameterSharing', 'FullyIndependentLearning', 'SharedCriticMethods'
    ],
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'rllib'}
