# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Torchtext wrapper module.'''

MODELS = {
    'task1': ['model1', 'model2', 'model3'],
    'task2': ['model1', 'model2', 'model3'],
    'task3': ['model1', 'model2', 'model3'],
    'task4': ['model1', 'model2', 'model3'],
}

DATASETS = {
    'task1': ['dataset1', 'dataset2'],
    'task2': ['dataset1', 'dataset2'],
    'task3': ['dataset1', 'dataset2'],
    'task4': ['dataset1', 'dataset2'],
}


def load_model(name: str):
  '''Gets a model directly from Torchtext library.

    Args:
      name: Name of the model to be gotten.

    Returns:
      Torchtext model.
    '''
  raise NotImplementedError("TODO(lalito) Implement me!")


def load_dataset(name: str):
  '''Gets a dataset directly from Torchtext library.

    Args:
      name: Name of the dataset to be gotten.

    Returns:
      Torchtext dataset.
    '''
  raise NotImplementedError("TODO(lalito) Implement me!")
