# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Torchtext wrapper module.'''

SOURCE_METADATA = {
    'name': 'torchtext',
    'original_name': 'Torchtext',
    'url': 'https://pytorch.org/text/stable/index.html'
}

DATASETS = {
    'Sentiment Analysis': ['SST', 'IMDb'],
    'Entailment': ['SNLI', 'MultiNLI'],
    'Language Modeling': ['WikiText-2', 'WikiText103', 'PennTreebank'],
    'Machine Translation': ['Multi30k', 'IWSLT', 'WMT14'],
    'Sequence Tagging': ['UDPOS', 'CoNLL2000Chunking'],
    'Question Answering': ['BABI20', 'TREC']
}


def load_dataset(name: str):
  '''Gets a dataset directly from Torchtext library.

    Args:
      name: Name of the dataset to be gotten.

    Returns:
      Torchtext dataset.
    '''
  return {'name': name, 'source': 'torchtext'}
