# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Abstract classes for standardized models and datasets.'''
from sotaai.nlp import utils

datasets_tasks_map = utils.map_name_tasks('datasets')
models_tasks_map = utils.map_name_tasks('models')


class NlpDataset(object):
  '''Our attempt at a standardized, task-agnostic NLP dataset wrapper.'''

  def __init__(self, name: str, source: str):
    '''Very preliminary class to encapsulate any NLP dataset.'''
    self.name = name
    self.source = source
    self.tasks = datasets_tasks_map[name]

  def to_dict(self) -> dict:
    return {
        'name': self.name,
        'type': 'dataset',
        'is_implemented': False,
        'source': self.source,
        'tasks': self.tasks
    }


class NlpModel(object):
  '''Our attempt at a standardized, task-agnostic NLP model wrapper.'''

  def __init__(self, name: str, source: str):
    '''Very preliminary class to encapsulate any NLP model.'''
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
