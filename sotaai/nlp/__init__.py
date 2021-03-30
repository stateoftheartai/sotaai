# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main NLP module to abstract away library specific API and standardize.'''
from sotaai.nlp import utils
from sotaai.nlp import abstractions
import importlib


def load_model(name: str) -> abstractions.NlpModel:
  '''Dummy load model function. Placeholder for real wrapper.'''
  model_source_map = utils.map_name_sources('models')
  source = model_source_map[name][0]
  wrapper = importlib.import_module('sotaai.nlp.' + source + '_wrapper')
  raw_object = wrapper.load_model(name)
  return abstractions.NlpModel(raw_object['name'], raw_object['source'])


def load_dataset(name: str) -> abstractions.NlpDataset:
  '''Dummy load dataset function. Placeholder for real wrapper.'''
  dataset_source_map = utils.map_name_sources('datasets')
  source = dataset_source_map[name][0]
  wrapper = importlib.import_module('sotaai.nlp.' + source + '_wrapper')
  raw_object = wrapper.load_dataset(name)
  return abstractions.NlpDataset(raw_object['name'], raw_object['source'])


def model_to_dataset(
    model: abstractions.NlpModel, dataset: abstractions.NlpDataset
) -> (abstractions.NlpModel, abstractions.NlpDataset):
  '''TODO(lalito) describe.'''
  # 1. Check whether to modify the model or the datset.
  # 2. Modify whatever is necessary, e.g.,
  #
  #  compatible_model = utils.modify_layers(model, dataset), o
  #
  #  compatible_dataset = utils.resize_samples(model, dataset)
  #
  raise NotImplementedError('TODO(lalito) implement me')


def create_models_dict(model_names, models_sources_map):
  '''Given a list of model names, return a list with the JSON representation
  of each model as an standardized dict
  Args:
    model_names (list): list of model names to return the standardized dict
    models_sources_map: a dict map between model names and sources as returned
      by the utils function map_name_sources('models')
  Returns:
    A list of dictionaries with the JSON representation of each CV model
  '''

  print('\nCreating model JSONs...')

  models = []

  for i, model_name in enumerate(model_names):
    print(' - ({}/{}) {}'.format(i + 1, len(model_names),
                                 'models.' + model_name))
    model = load_model(model_name)
    model_dict = model.to_dict()

    model_dict['sources'] = models_sources_map[model_dict['name']]
    del model_dict['source']

    model_dict['unified_name'] = model_name  # TODO(tonio) unify...

    models.append(model_dict)

  return models


def create_datasets_dict(dataset_names, dataset_sources_map):
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

  print('\nCreating dataset JSONs...')

  datasets = []

  for i, dataset_name in enumerate(dataset_names):
    print(' - ({}/{}) {}'.format(i + 1, len(dataset_names),
                                 'datasets.' + dataset_name))
    dataset = load_dataset(dataset_name)
    dataset_dict = dataset.to_dict()

    dataset_dict['sources'] = dataset_sources_map[dataset_dict['name']]
    del dataset_dict['source']

    dataset_dict['unified_name'] = dataset_name  # TODO(tonio) unify...

    datasets.append(dataset_dict)

  return datasets
