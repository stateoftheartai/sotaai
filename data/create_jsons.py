# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Script used to create JSON files containing sotaai models and datasets metadata
'''
import importlib
import sys
import json


def main(area: str, output_dir='./data/output/'):
  '''Create a JSON file with the all models and datasets of a given area

  Args:
    area: one of cv, nlp, rl, neuro
    output_dir: the output directory where to place the JSON file created
  '''

  print('\nAbout to create JSONs...')

  sotaai_module = importlib.import_module('sotaai.{}'.format(area))

  output_file_path = '{}{}.json'.format(output_dir, area)
  model_names, dataset_names = get_model_and_dataset_names(area)

  print('Area: {}'.format(area.upper()))
  print('Models: {}'.format(len(model_names)))
  print('Datasets: {}'.format(len(dataset_names)))
  print('JSON output: {}'.format(output_file_path))

  models = sotaai_module.create_models_dict(model_names)
  datasets = sotaai_module.create_datasets_dict(dataset_names)

  save_json(models + datasets, output_file_path)

  print('\nJSONs files created successfully.\n')


def get_model_and_dataset_names(area: str):
  '''Return the set of model and datasets name available for a given area

  Args:
    area (str): a valid area e.g. cv, nlp, rl, or neuro

  Returns:
    model_names (list): the list of model names available for the given area
    dataset_names (list): the list of dataset names available for the given area
  '''

  sotaai_utils = importlib.import_module('sotaai.{}.utils'.format(area))
  sources = list(set(sotaai_utils.MODEL_SOURCES + sotaai_utils.DATASET_SOURCES))

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


args = sys.argv[1:]

if len(args) == 0:
  print('An area of cv, nlp, rl, or neuro is required to create JSON files')
  sys.exit(0)

main(args[0])
