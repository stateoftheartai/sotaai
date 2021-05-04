# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Abstract classes for standardized models and datasets.'''
from sotaai.neuro import utils

datasets_tasks_map = utils.map_name_tasks('datasets')
models_tasks_map = utils.map_name_tasks('models')


class NeuroDataset(object):
  '''Our attempt at a standardized, task-agnostic dataset wrapper.'''

  def __init__(self, name: str, source: str):
    '''Very preliminary class to encapsulate any neurosymbolic dataset.'''
    self.name = name
    self.source = source
    self.tasks = datasets_tasks_map[name]

  def to_dict(self) -> dict:
    return {
        'name': self.name,
        'is_implemented': False,
        'type': 'dataset',
        'source': self.source,
        'tasks': self.tasks
    }


class NeuroModel(object):
  '''Our attempt at a standardized, task-agnostic model wrapper.'''

  def __init__(self, name: str, source: str):
    '''Very preliminary class to encapsulate any neurosymbolic model.'''
    self.name = name
    self.source = source
    self.tasks = models_tasks_map[name]

  def to_dict(self) -> dict:
    return {
        'name': self.name,
        'type': 'model',
        'is_implemented': False,
        'source': self.source,
        'tasks': self.tasks
    }
