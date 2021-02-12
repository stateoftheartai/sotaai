# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main CV module to abstract away library specific API and standardize.'''
from sotaai.cv import utils
from sotaai.cv import abstractions
import importlib


def load_model(name: str,
               source: str = '',
               input_tensor=None,
               include_top=None) -> abstractions.CvModel:
  '''Fetch a model from a specific source, and return standardized object.

  Args:
    name (str):
      Name of the model.
    source (str):
      Optional parameter to indicate a specific source library.

  Returns (abstractions.CvModel):
    The standardized model.
  '''
  model_source_map = utils.map_name_sources('models',
                                            return_original_names=False)
  lower_name = name.lower()
  if source:
    valid_sources = model_source_map[lower_name]
    # Make sure the chosen source is available.
    if source not in valid_sources:
      raise NameError(
          'Source {} not available for model {}.'.format(source, name) +
          ' Available sources are: {}'.format(valid_sources))
  else:
    source = model_source_map[lower_name][0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
  raw_object = wrapper.load_model(name,
                                  input_tensor=input_tensor,
                                  include_top=include_top)

  return abstractions.CvModel(raw_object, name)


def load_dataset(name: str, source: str = '') -> abstractions.CvDataset:
  '''Fetch a dataset from a specific source, and return standardized object.

  Args:
    name (str):
      Name of the dataset.
    source (str):
      Optional parameter to indicate a specific source library.

  Returns (abstractions.CvDataset):
    The standardized dataset.

  # TODO(tonioteran) Add input sanitizer checks to make sure we're loading only
  # available models.
  '''
  # TODO(hugo) Switch for new function to get the dataset source.
  ds_source_map = utils.map_name_sources('datasets')
  if source:
    valid_sources = ds_source_map[name]
    # Make sure the chosen source is available.
    if source not in valid_sources:
      raise NameError(
          'Source {} not available for dataset {}.'.format(source, name) +
          ' Available sources are: {}'.format(valid_sources))
  else:
    source = ds_source_map[name][0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
  raw_object = wrapper.load_dataset(name)

  # Build a standardized `CvDataset` object per dataset split:
  std_dataset = dict()
  for split_name in raw_object:
    std_dataset[split_name] = abstractions.CvDataset(raw_object[split_name],
                                                     name, split_name)

  return std_dataset
