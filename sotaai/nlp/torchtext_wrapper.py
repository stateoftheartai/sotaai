# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Torchtext wrapper module.'''

DATASETS = {
    'Sentiment Analysis': ['SST', 'IMDb'],
    'Question Classification': ['TREC'],
    'Entailment': ['SNLI', 'MultiNLI'],
    'Language Modeling': ['WikiText-2', 'WikiText103', 'PennTreebank'],
    'Machine Translation': ['Multi30k', 'IWSLT', 'WMT14'],
    'Sequence Tagging': ['UDPOS', 'CoNLL2000Chunking'],
    'Question Answering': ['BABI20']
}


def load_model(name: str):
  '''Gets a model directly from Torchtext library.

    Args:
      name: Name of the model to be gotten.

    Returns:
      Torchtext model.
    '''
  raise NotImplementedError('TODO(lalito) Implement me!')


def load_dataset(name: str):
  '''Gets a dataset directly from Torchtext library.

    Args:
      name: Name of the dataset to be gotten.

    Returns:
      Torchtext dataset.
    '''
  raise NotImplementedError('TODO(lalito) Implement me!')
