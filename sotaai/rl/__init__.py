# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main RL module to abstract away library specific API and standardize.'''
from sotaai.rl import utils
from sotaai.rl import abstractions
import importlib


def load_environment(name: str,
                     import_library=True) -> abstractions.RlEnvironment:
  '''Dummy load dataset function. Placeholder for real wrapper.'''
  # dataset_source_map = utils.map_name_sources('datasets')
  if not import_library:
    return abstractions.RlEnvironment(name)
  else:
    source = 'gym'
    wrapper = importlib.import_module('sotaai.rl.' + source + '_wrapper')
    raw_object = wrapper.load_environment(name)
    return abstractions.RlEnvironment(name, raw_object)


def load_model(name: str,
               name_env: str = 'CartPole-v1',
               import_library=True) -> abstractions.RlModel:
  '''Dummy load model function. Placeholder for real wrapper.'''
  # model_source_map = utils.map_name_sources('models')
  source = 'garage'
  if not import_library:
    return abstractions.RlModel(name=name)
  else:
    wrapper = importlib.import_module('sotaai.rl.' + source + '_wrapper')
    raw_object = wrapper.load_model(name, name_env=name_env)
    env = load_environment(name=name_env)
    return abstractions.RlModel(name=name,
                                raw_algo=raw_object,
                                source=source,
                                environment=env)


def create_models_dict(model_names,
                       models_sources_map,
                       import_library=False,
                       log=False):
  '''Given a list of model names, return a list with the JSON representation
    of each model as an standardized dict
    Args:
      model_names (list): list of model names to return the standardized dict
      models_sources_map: a dict map between model names and sources as returned
        by the utils function map_name_sources('models')
    Returns:
      A list of dictionaries with the JSON representation of each CV model
    '''
  models = []

  for i, model_name in enumerate(model_names):

    if log:
      print(' - ({}/{}) {}'.format(i + 1, len(model_names),
                                   'models.' + model_name))
    model = load_model(name=model_name, import_library=import_library)
    model_dict = model.to_dict()

    model_dict['sources'] = models_sources_map[model_dict['name']]
    del model_dict['source']
    model_dict['implemented_sources'] = model_dict['sources']

    model_dict['unified_name'] = model_name  # TODO(tonio) unify...

    models.append(model_dict)

  return models


def create_datasets_dict(dataset_names,
                         dataset_sources_map,
                         import_library=False,
                         log=False):
  '''Given a list of dataset names, return a list with the JSON representation
    of each dataset as an standardized dict

    Args:
      dataset_names (list): list of dataset names to
                            return the standardized dict
        dataset
      dataset_sources_map: a dict map between dataset names and sources as
        returned by the utils function map_name_sources('datasets')

    Returns:
      A list of dictionaries with the JSON representation of each CV model
    '''
  datasets = []

  for i, dataset_name in enumerate(dataset_names):

    if log:
      print(' - ({}/{}) {}'.format(i + 1, len(dataset_names),
                                   'datasets.' + dataset_name))
    dataset = load_environment(name=dataset_name, import_library=import_library)
    dataset_dict = dataset.to_dict()

    dataset_dict['sources'] = dataset_sources_map[dataset_dict['name']]
    del dataset_dict['source']
    dataset_dict['implemented_sources'] = dataset_dict['sources']

    # DELETE all non primitive or array types. Neo4j only
    # accepts primitives and arrays.
    # TODO(jorge): Refactor these properties.
    del dataset_dict['metadata']
    del dataset_dict['observation_space']
    del dataset_dict['action_space']

    dataset_dict['unified_name'] = dataset_name  # TODO(tonio) unify...

    datasets.append(dataset_dict)

  return datasets
