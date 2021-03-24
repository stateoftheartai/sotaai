# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Garage's library wrapper.'''

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


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'garage'}
