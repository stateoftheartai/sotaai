# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Script used to create JSON files containing sotaai models and datasets metadata
'''
import importlib
import sys
from data import utils


def main(area: str, output_dir='./data/output/'):

  print('\nAbout to create JSONs...')

  sotaai_utils = importlib.import_module('sotaai.{}.utils'.format(area))
  sotaai_module = importlib.import_module('sotaai.{}'.format(area))
  load_model = getattr(sotaai_module, 'load_model')
  load_dataset = getattr(sotaai_module, 'load_dataset')

  output_file_path = '{}{}.json'.format(output_dir, area)
  model_names, dataset_names = utils.get_model_and_dataset_names(
      area, sotaai_utils)

  print('Area: {}'.format(area.upper()))
  print('Models: {}'.format(len(model_names)))
  print('Datasets: {}'.format(len(dataset_names)))
  print('JSON output: {}'.format(output_file_path))

  module = importlib.import_module('data.{}'.format(area))
  models = module.create_models_dict(model_names, load_model)
  datasets = module.create_datasets_dict(dataset_names, load_dataset)

  utils.save_json(models + datasets, output_file_path)

  print('\nJSONs files created successfully.\n')


args = sys.argv[1:]

if len(args) == 0:
  print('An area of cv, nlp, rl, or neuro is required to create JSON files')
  sys.exit(0)

main(args[0])
