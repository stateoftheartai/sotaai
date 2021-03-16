# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Script used to create JSON files containing sotaai models and datasets metadata
'''
import importlib
import sys

args = sys.argv[1:]
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

print('Area: {}'.format(area.upper()))
print('Models: {}'.format(len(model_names)))
print('Datasets: {}'.format(len(dataset_names)))

for i, model_name in enumerate(model_names):
  print(model_name, i)
  model = area_module.load_model(model_name)
  print(model.to_dict())
