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


def load_model(name: str,
               source: str = '',
               input_tensor=None,
               pretrained=None,
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

  if source == 'torch':
    raw_object = wrapper.load_model(name, pretrained=pretrained)
  else:
    raw_object = wrapper.load_model(name,
                                    input_tensor=input_tensor,
                                    include_top=include_top)

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

  if source == 'torch':
    raw_object = wrapper.load_dataset(name,
                                      transform=transform,
                                      ann_file=ann_file,
                                      target_transform=target_transform)
  else:
    raw_object = wrapper.load_dataset(name)

  # Build a standardized `CvDataset` object per dataset split:
  std_dataset = dict()
  for split_name in raw_object:
    raw = raw_object[split_name]
    iterator = wrapper.DatasetIterator(raw)

    # print(iterator)
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
  # print('\nModel ', cv_model.name)
  # print(' Input: ', cv_model.original_input_shape)
  # print(' Output: ', cv_model.original_output_shape)
  # print(' Input Type', cv_model.original_input_type)
  # print('Dataset: ', cv_dataset.name)
  # print(' Shape:   ', cv_dataset.shape)
  # print(' Classes: ', cv_dataset.classes_shape)

  if cv_model.source == 'keras':
    cv_model, cv_dataset = keras_wrapper.model_to_dataset(cv_model, cv_dataset)

  elif cv_model.source == 'torchvision':
    task = cv_dataset.tasks[0]
    if task == 'classification':
      torch_wrapper.model_to_dataset_classification(cv_model, cv_dataset)
    elif task == 'segmentation':
      torch_wrapper.model_to_dataset_segmentation(cv_model, cv_dataset)

  # print('\nModel ', cv_model.name)
  # print(' Input: ', cv_model.original_input_shape)
  # print(' Output: ', cv_model.original_output_shape)
  # print('Dataset: ', cv_dataset.name)
  # print(' Shape:   ', cv_dataset.shape)
  # print(' Classes: ', cv_dataset.classes_shape)

  return cv_model, cv_dataset


def create_models_dict(model_names):
  '''Given a list of model names, return a list with the JSON representation
  of each model as an standardized dict

  Args:
    model_names (list): list of model names to return the standardized dict

  Returns:
    A list of dictionaries with the JSON representation of each CV model
  '''

  print('\nCreating model JSONs...')

  models = []

  for i, model_name in enumerate(model_names):
    print(' - ({}/{}) {}'.format(i + 1, len(model_names),
                                 'models.' + model_name))
    model = load_model(model_name)
    models.append(model.to_dict())

  return models


def create_datasets_dict(dataset_names):
  '''Given a list of dataset names, return a list with the JSON representation
  of each dataset as an standardized dict

  Args:
    dataset_names (list): list of dataset names to return the standardized dict
      dataset

  Returns:
    A list of dictionaries with the JSON representation of each CV model
  '''

  # TODO(Hugo)
  # This is a temporal variable to test datasets JSON creation only for this set
  # of datasets to save memory. Once tested, the JSON is to be tested and
  # created for the whole set of datasets available
  test_datasets = [
      'mnist', 'cifar10', 'cifar100', 'fashion_mnist', 'beans',
      'binary_alpha_digits', 'caltech_birds2010', 'caltech_birds2011',
      'cars196', 'cats_vs_dogs', 'omniglot', 'lost_and_found'
  ]

  print('\nCreating dataset JSONs...')

  datasets = []

  for i, dataset_name in enumerate(dataset_names):

    print(' - ({}/{}) {}'.format(i + 1, len(dataset_names),
                                 'datasets.' + dataset_name))

    # TODO(Hugo)
    # Remove this if when test_datasets variable is removed
    if not dataset_name in test_datasets:
      continue

    # Abstract datasets are created per split, but for the JSON representation
    # we only want one global representation which contains the split metadata
    # as an attribute, that's why we have to iterate over the splits to extract
    # the splits information and then extend the dataset dict with this split
    # data
    dataset_splits = load_dataset(dataset_name)
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

  return datasets
