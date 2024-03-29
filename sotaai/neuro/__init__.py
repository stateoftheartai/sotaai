# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main neurosymbolic-reasoning module to standardize.'''
from sotaai.neuro import utils
from sotaai.neuro import abstractions
import importlib


def load_model(name: str) -> abstractions.NeuroModel:
  '''Dummy load model function. Placeholder for real wrapper.'''
  model_source_map = utils.map_name_sources('models')
  source = model_source_map[name][0]
  wrapper = importlib.import_module('sotaai.neuro.' + source + '_wrapper')
  raw_object = wrapper.load_model(name)
  return abstractions.NeuroModel(raw_object['name'], raw_object['source'])


def load_dataset(name: str) -> abstractions.NeuroDataset:
  '''Dummy load dataset function. Placeholder for real wrapper.'''
  dataset_source_map = utils.map_name_sources('datasets')
  source = dataset_source_map[name][0]
  wrapper = importlib.import_module('sotaai.neuro.' + source + '_wrapper')
  raw_object = wrapper.load_dataset(name)
  return abstractions.NeuroDataset(raw_object['name'], raw_object['source'])


def create_models_dict(model_names, models_sources_map, log=False):
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
    model = load_model(model_name)
    model_dict = model.to_dict()

    model_dict['sources'] = models_sources_map[model_dict['name']]
    del model_dict['source']

    model_dict['implemented_sources'] = []

    model_dict['unified_name'] = model_name  # TODO(tonio) unify...

    models.append(model_dict)

  return models


def create_datasets_dict(dataset_names, dataset_sources_map, log=False):
  '''Given a list of dataset names, return a list with the JSON representation
  of each dataset as an standardized dict

  Args:
    dataset_names (list): list of dataset names to return the standardized dict
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
    dataset = load_dataset(dataset_name)
    dataset_dict = dataset.to_dict()

    dataset_dict['sources'] = dataset_sources_map[dataset_dict['name']]
    del dataset_dict['source']

    dataset_dict['implemented_sources'] = []

    dataset_dict['unified_name'] = dataset_name  # TODO(tonio) unify...

    datasets.append(dataset_dict)

  return datasets
