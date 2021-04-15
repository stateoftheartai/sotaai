# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Ampligraph's library wrapper.'''

SOURCE_METADATA = {
    'name': 'ampligraph',
    'original_name': 'AmpliGraph',
    'url': 'https://docs.ampligraph.org'
}

MODELS = {
    'Unknown': [
        'RandomBaseline', 'TransE', 'DistMult', 'ComplEx', 'HolE', 'ConvE',
        'ConvKB'
    ],
}

DATASETS = {
    'Unknown': [
        'FB15k-237', 'WN18RR', 'YAGO3-10', 'Freebase15k', 'WordNet18',
        'WordNet11', 'Freebase13'
    ],
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'ampligraph'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'ampligraph'}
