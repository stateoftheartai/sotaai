# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Module to store metadata of Neurosym models, datasets, etc.
'''

DATASETS = {}

MODELS = {}


def get(item_type=None, name=None, source=None):
  '''Return datasets/models metadata given source or name

  Args:
    item_type: whether 'models' or 'datasets'
    name: a valid dataset or model name e.g. 'mnist', if given source is ignored
    source: a valid source name e.g. 'keras'. If name given this attribute is
    ignored.

  Returns:
    If name given, return the metadata object for the given dataset or model. If
    source given, return an array of metadatas for the datasets or models that
    matched the given source.
  '''
  if name is not None:
    items = DATASETS if item_type == 'datasets' else MODELS
    return items[name]

  items = DATASETS.values() if item_type == 'datasets' else MODELS.values()
  return filter(lambda item: source in item['sources'], items)
