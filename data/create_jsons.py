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

  try:
    sotaai_module = importlib.import_module('sotaai.{}'.format(area))
    sotaai_utils_module = importlib.import_module(
        'sotaai.{}.utils'.format(area))

    models_sources_map = sotaai_utils_module.map_name_sources('models')
    datasets_sources_map = sotaai_utils_module.map_name_sources('datasets')

    output_file_path = '{}{}.json'.format(output_dir, area)
    model_names = models_sources_map.keys()
    dataset_names = datasets_sources_map.keys()

    print('Area: {}'.format(area.upper()))
    print('Models: {}'.format(len(model_names)))
    print('Datasets: {}'.format(len(dataset_names)))
    print('JSON output: {}'.format(output_file_path))

    models = sotaai_module.create_models_dict(model_names, models_sources_map)
    datasets = sotaai_module.create_datasets_dict(dataset_names,
                                                  datasets_sources_map)
  except Exception as e:
    raise NotImplementedError(
        'JSON creation for {} is still not implemented'.format(area)) from e

  save_json(models + datasets, output_file_path)

  print('\nJSONs files created successfully.\n')


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
