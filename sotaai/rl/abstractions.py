# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Abstract classes for standardized models and datasets.'''
from sotaai.rl import utils

datasets_tasks_map = utils.map_name_tasks('datasets')
models_tasks_map = utils.map_name_tasks('models')


class RlEnvironment(object):
  '''Our attempt at a standardized, task-agnostic NLP dataset wrapper.'''

  def __init__(self, raw_object, name: str):
    '''Very preliminary class to encapsulate any RL environment.
     Args:
      raw_object:
        Environment object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the dataset.
      source (str):
        Source name used when no raw_environment is given
    '''
    self.name = name
    self.source = utils.get_source(name)
    self.raw = raw_object

    self.action_space_size = None
    if hasattr(raw_object.action_space, 'n'):
      self.action_space_size = raw_object.action_space.n
    else:
      self.action_space_size = []
      if 'Box' not in str(type(raw_object.action_space)):
        for size in raw_object.action_space:
          self.action_space_size.append(size.n)
      else:
        self.action_space_size = str(raw_object.action_space)

    self.action_space_dtype = str(raw_object.action_space.dtype)
    self.action_space_shape = raw_object.action_space.shape
    self.observation_space_dtype = str(raw_object.observation_space.dtype)
    self.observation_space_shape = raw_object.observation_space.shape

    self.metadata = None
    self.reward_range = None
    if hasattr(raw_object, 'metadata'):
      self.metadata = raw_object.metadata
    if hasattr(raw_object, 'reward_range'):
      self.reward_range = raw_object.reward_range

    # self.tasks = datasets_tasks_map[name]

  def to_dict(self) -> dict:
    return {
        'name': self.name,
        'type': 'Environment',
        'source': self.source,
        'is_implemented': True,
        'action_space': {
            'size': self.action_space_size,
            'dtype': self.action_space_dtype,
            'shape': self.action_space_shape
        },
        'observation_space': {
            'dtype': self.observation_space_dtype,
            'shape': self.observation_space_shape
        },
        'metadata': self.metadata,
        'reward_range': self.reward_range
    }


class RlModel(object):
  '''Our attempt at a standardized, model wrapper.

  Each abstract `RlModel` represents a model from one of the sources.
  '''

  def __init__(self, name: str, raw_object, source: str):
    '''Constructor using `raw_model` from a source library.

    Args:
      raw_model:
        Model object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the model.
      source (str):
        Source name used when no raw_model is given
    '''
    self.name = name
    self.raw = raw_object
    self.source = source
    # self.tasks = models_tasks_map[name]

  def to_dict(self) -> dict:
    return {
        'name': self.name,
        'type': 'model',
        'is_implemented': True,
        'source': self.source,
        # 'tasks': self.tasks
    }
