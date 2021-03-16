# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
General utilities that support JSON files creation
'''
import importlib
import json


def get_model_and_dataset_names(area: str, utils):
  '''Return the set of model and datasets name available for a given area

  Args:
    area (str): a valid area e.g. cv, nlp, rl, or neuro
    utils (module): the utils module of the given area

  Returns:
    model_names (list): the list of model names available for the given area
    dataset_names (list): the list of dataset names available for the given area
  '''

  sources = list(set(utils.MODEL_SOURCES + utils.DATASET_SOURCES))

  model_names = []
  dataset_names = []

  for source in sources:

    wrapper_file_name = 'sotaai.{}.{}_wrapper'.format(area, source)
    wrapper = importlib.import_module(wrapper_file_name)

    if hasattr(wrapper, 'MODELS'):
      for task in wrapper.MODELS:
        model_names = model_names + wrapper.MODELS[task]

    if hasattr(wrapper, 'DATASETS'):
      for task in wrapper.DATASETS:
        dataset_names = dataset_names + wrapper.DATASETS[task]

  return model_names, dataset_names


def save_json(data, file_path):
  '''Store the given data in a JSON file

  Args:
    data (dict): a valid JSON value (dict or list of dicts)
    file_path (str): the file path where to store the JSON file e.g. cv.json
  '''
  data_file = open(file_path, 'w')
  json.dump(data, data_file, indent=2)
  data_file.close()
