# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Contains functions that return a list of standardized dictionaries of the
models and datasets available for computer vision
'''


def create_models_dict(model_names, load_model):
  '''Given a list of model names, return a list with the JSON representation
  of each model as an standardized dict

  Args:
    model_names (list): list of model names to return the standardized dict
    load_model: sotaai.cv.load_model function to instantiate the abstract model

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


def create_datasets_dict(dataset_names, load_dataset):
  '''Given a list of dataset names, return a list with the JSON representation
  of each dataset as an standardized dict

  Args:
    dataset_names (list): list of dataset names to return the standardized dict
    load_dataset: sotaai.cv.load_dataset function to instantiate the abstract
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
