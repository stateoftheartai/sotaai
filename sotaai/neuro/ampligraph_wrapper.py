# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Ampligraph's library wrapper.'''
from sotaai.neuro.abstractions import NeuroDataset, NeuroModel

SOURCE_METADATA = {
    'name': 'ampligraph',
    'original_name': 'AmpliGraph',
    'url': 'ampligraph.org'
}

MODELS = {
    'unknown': [
        'RandomBaseline', 'TransE', 'DistMult', 'ComplEx', 'HolE', 'ConvE',
        'ConvKB'
    ],
}

DATASETS = {
    'unknown': [
        'FB15k-237', 'WN18RR', 'YAGO3-10', 'Freebase15k', 'WordNet18',
        'WordNet11', 'Freebase13'
    ],
}


def load_dataset(name: str) -> NeuroDataset:
  return NeuroDataset(name, 'ampligraph')


def load_model(name: str) -> NeuroModel:
  return NeuroModel(name, 'ampligraph')
