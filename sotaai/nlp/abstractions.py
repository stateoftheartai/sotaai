# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Abstract classes for standardized models and datasets.'''


class NlpDataset(object):
  '''Our attempt at a standardized, task-agnostic NLP dataset wrapper.

  TODO(lalo) describe.
  '''

  def __init__(self, raw_dataset):
    '''Ctor for building wrapped dataset from `raw_dataset`.

    TODO(lalo) describe.
    '''
    self.raw = raw_dataset
    # TODO(lalo) implement.


class NlpModel(object):
  '''Our attempt at a standardized, task-agnostic NLP model wrapper.

  TODO(lalo) describe.
  '''

  def __init__(self, raw_model):
    '''Ctor for building wrapped model from `raw_model`.

    TODO(lalo) describe.
    '''
    self.raw = raw_model
    # TODO(lalo) implement.
