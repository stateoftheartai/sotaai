# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Script used to create JSON files containing sotaai models and datasets metadata
'''
import importlib
import sys
import json

args = sys.argv[1:]

# TODO(Hugo)
# This is a temporal variable to test datasets JSON creation only for this set
# of datasets to save memory. Once tested, the JSON is to be tested and created
# for the whole set of datasets available
test_datasets = [
    'mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'beans',
    'binary_alpha_digits', 'caltech_birds2010', 'caltech_birds2011', 'cars196',
    'cats_vs_dogs', 'omniglot', 'lost_and_found'
]

# Inputs:
data_file_path = './data.json'
area = args[0]

area_utils = importlib.import_module('sotaai.{}.utils'.format(area))
area_module = importlib.import_module('sotaai.{}'.format(area))

sources = list(set(area_utils.MODEL_SOURCES + area_utils.DATASET_SOURCES))

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

print('\nAbout to create JSONs for:')
print('Area: {}'.format(area.upper()))
print('Models: {}'.format(len(model_names)))
print('Datasets: {}'.format(len(dataset_names)))
print('JSON output: {}'.format(data_file_path))

print('\nCreating model JSONs...')
models = []
for i, model_name in enumerate(model_names):
  print(' - ({}/{}) {}'.format(i + 1, len(model_names), 'models.' + model_name))
  model = area_module.load_model(model_name)
  models.append(model.to_dict())

print('\nCreating dataset JSONs...')
datasets = []
for i, dataset_name in enumerate(dataset_names):

  print(' - ({}/{}) {}'.format(i + 1, len(dataset_names),
                               'datasets.' + dataset_name))

  # TODO(Hugo)
  # Remove this if when test_datasets variable is removed
  if not dataset_name in test_datasets:
    continue

  dataset_splits = area_module.load_dataset(dataset_name)
  split_names = dataset_splits.keys()

  dataset_dict = None
  splits_data = []
  total_items = 0
  for split_name in split_names:
    dataset = dataset_splits[split_name]
    dataset_dict = dataset.to_dict()
    splits_data.append({
        'name': split_name,
        'num_items': dataset_dict['num_items']
    })
    total_items += dataset_dict['num_items']
    del dataset_dict['num_items']

  dataset_dict['splits'] = splits_data
  dataset_dict['total_items'] = total_items
  datasets.append(dataset_dict)

data_file = open(data_file_path, 'w')
json.dump(models + datasets, data_file, indent=2)
data_file.close()

print('\nJSONs files created successfully.\n')
