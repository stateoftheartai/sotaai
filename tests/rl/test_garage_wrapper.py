# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the gym wrapper.'''

import unittest
from sotaai.rl import gym_wrapper, garage_wrapper, load_model
from sotaai.rl.abstractions import RlModel
# from garage.trainer import Trainer
from garage.experiment import SnapshotConfig
from garage.trainer import Trainer
from garage.envs import GymEnv, normalize
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

  def test_load_model(self):
    '''
      Make sure garage envs are returned.
    '''

    hyper_parameters = {
        'hidden_sizes': [32, 32],
        'max_kl': 0.01,
        'gae_lambda': 0.97,
        'discount': 0.99,
        'n_epochs': 999,
        'batch_size': 1024,
    }

    # env_gym = gym_wrapper.load_environment('CartPole-v0')
    env = normalize(GymEnv('CartPole-v0'))
    algo = garage_wrapper.load_model(name='TRPO', name_env='CartPole-v0')

    ctxt = SnapshotConfig(snapshot_dir='~/experiments',
                          snapshot_mode='all',
                          snapshot_gap=1)
    # deterministic.set_seed(10)
    trainer = Trainer(ctxt)

    trainer.setup(algo, env)
    trainer.train(n_epochs=hyper_parameters['n_epochs'],
                  batch_size=hyper_parameters['batch_size'])

    env.close()
    # trainer.train(n_epochs=hyper_parameters['n_epochs'],
    #               batch_size=hyper_parameters['batch_size'])

    # with TFTrainer(ctxt) as trainer:
    #   trainer.setup(algo, env)
    #   trainer.train(n_epochs=100, batch_size=4000)
    # for model_name in MODELS:
    #   if not model_name in EROR_MODELS:
    #     for task in gym_wrapper.LIST_ENVIRONMENTS:
    #       for env_name in gym_wrapper.LIST_ENVIRONMENTS[task]:
    #         print(f'Model name: {model_name}\t---\t Env name: {env_name}')
    #         garage_wrapper.load_model(name=model_name, name_env=env_name)

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
