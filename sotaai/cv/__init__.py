# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main CV module to abstract away library specific API and standardize.'''
from sotaai.cv import utils
from sotaai.cv import abstractions
from sotaai.cv import keras_wrapper
from sotaai.cv import torch_wrapper
import importlib

datasets_source_map = utils.map_name_sources('datasets')


def load_model(
    name: str,
    source: str = '',
    pretrained=None,
) -> abstractions.CvModel:
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
  valid_sources = model_source_map[lower_name]

  if source and source not in valid_sources:
    raise NameError(
        'Source {} not available for model {}.'.format(source, name) +
        ' Available sources are: {}'.format(valid_sources))
  else:
    source = valid_sources[0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')

  # TODO(Hugo)
  # Create an abstraction for the input of models that standardizes model inputs
  # across different libraries (configs)
  # As of now, we only have one input: pretrained or not

  if source in ['torch', 'keras']:
    raw_object = wrapper.load_model(name, pretrained=pretrained)
  # Non fully implemented sources fall in this case
  else:
    raw_object = wrapper.load_model(name)

  return abstractions.CvModel(raw_object, name)


def load_dataset(name: str,
                 source: str = '',
                 transform=None,
                 target_transform=None,
                 ann_file=None) -> abstractions.CvDataset:
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
  valid_sources = datasets_source_map[name]

  if source and source not in valid_sources:
    raise NameError(
        'Source {} not available for dataset {}.'.format(source, name) +
        ' Available sources are: {}'.format(valid_sources))
  else:
    source = datasets_source_map[name][0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')

  # TODO(Hugo)
  # Remove this variable or comment it out to create the full JSON data
  # This is a temporal variable to test the JSON creation only for a subset of
  # datasets to save memory
  test_datasets = [
      'mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'beans',
      'binary_alpha_digits', 'caltech_birds2010', 'caltech_birds2011',
      'cars196', 'cats_vs_dogs', 'omniglot', 'lost_and_found', 'wider_face'
  ]

  if name in test_datasets:
    # TODO(Hugo)
    # As more sources are being added (fully-implemented), update the IF
    # statement.
    # The IF was added temporary to make sure only fully implemented sources
    # have the raw object and can actually be used in code
    if source == 'torch':
      raw_object = wrapper.load_dataset(name,
                                        transform=transform,
                                        ann_file=ann_file,
                                        target_transform=target_transform)
    else:
      raw_object = wrapper.load_dataset(name)
  elif source in ['keras', 'tensorflow', 'torch']:
    raw_object = wrapper.load_dataset(name, download=False)
  else:
    raw_object = wrapper.load_dataset(name)

  # Build a standardized `CvDataset` object per dataset split:
  std_dataset = dict()
  for split_name in raw_object:
    raw = raw_object[split_name]

    # TODO(Hugo)
    # As of now, iterator does not exists for those sources not fully
    # implemented or tested, once all sources are implemented this if will be
    # irrelevant since all wrappers will have their iterator class
    iterator = None
    if raw and hasattr(wrapper, 'DatasetIterator'):
      iterator = wrapper.DatasetIterator(raw)

    std_dataset[split_name] = abstractions.CvDataset(raw, iterator, name,
                                                     split_name)

  return std_dataset


def model_to_dataset(cv_model, cv_dataset):
  '''If compatible, adjust model and dataset so that they can be executed
  against each other

  Args:
    cv_model: an abstracted cv model
    cv_dataset: an abstracted cv dataset

  Returns:
    cv_model: the abstracted cv model adjusted to be executed against
      cv_dataset
    cv_dataset: the abstracted cv dataset adjust to be executed against
      cv_model
  '''

  # Uncomment following prints to test model_to_dataset input and outputs...
  print('\nModel ', cv_model.name)
  print(' Input: ', cv_model.original_input_shape)
  print(' Output: ', cv_model.original_output_shape)
  print(' Input Type', cv_model.original_input_type)
  print('Dataset: ', cv_dataset.name)
  print(' Shape:   ', cv_dataset.shape)
  print(' Classes: ', cv_dataset.classes_shape)

  if cv_model.source == 'keras':
    cv_model, cv_dataset = keras_wrapper.model_to_dataset(cv_model, cv_dataset)

  elif cv_model.source == 'torchvision':
    task = cv_dataset.tasks[0]
    if task == 'classification':
      torch_wrapper.model_to_dataset_classification(cv_model, cv_dataset)
    elif task == 'segmentation':
      torch_wrapper.model_to_dataset_segmentation(cv_model, cv_dataset)
    elif task in ('object_detection', 'pose estimation'):
      torch_wrapper.model_to_dataset_object_detection(cv_model, cv_dataset)

  # print('\nModel ', cv_model.name)
  # print(' Input: ', cv_model.original_input_shape)
  # print(' Output: ', cv_model.original_output_shape)
  # print('Dataset: ', cv_dataset.name)
  # print(' Shape:   ', cv_dataset.shape)
  # print(' Classes: ', cv_dataset.classes_shape)

  return cv_model, cv_dataset


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

    # Abstract datasets are created per split, but for the JSON representation
    # we only want one global representation which contains the split metadata
    # as an attribute, that's why we have to iterate over the splits to extract
    # the splits information and then extend the dataset dict with this split
    # data
    dataset_splits = load_dataset(dataset_name)

    dataset_dict = None
    split_names = []
    split_num_items = []
    total_items = 0

    for split_name in dataset_splits:
      dataset = dataset_splits[split_name]
      dataset_dict = dataset.to_dict()
      split_names.append(split_name)

      if dataset_dict['cv_num_items']:
        split_num_items.append(dataset_dict['cv_num_items'])

      if dataset_dict['cv_num_items'] is not None:
        total_items += dataset_dict['cv_num_items']

      del dataset_dict['source']
      del dataset_dict['cv_num_items']

    dataset_dict['sources'] = dataset_sources_map[dataset_dict['name']]

    dataset_dict['cv_split_names'] = split_names
    dataset_dict['cv_split_num_items'] = split_num_items
    dataset_dict['cv_total_items'] = total_items
    datasets.append(dataset_dict)

  return datasets
