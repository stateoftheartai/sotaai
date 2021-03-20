# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''DGL-KE's library wrapper.

Model information taken from:
- https://dglke.dgl.ai/doc/eval.html?highlight=models
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'dglke',
    'area': 'neuro',
    'original_name': 'DGL-KE',
    'url': 'https://dglke.dgl.ai/doc/'
}

MODELS = {
    'unknown': [
        'TransE', 'TransE_l1', 'TransE_l2', 'TransR', 'RESCAL', 'DistMult',
        'ComplEx', 'RotatE'
    ]
}

DATASETS = {'unknown': ['FB15k', 'FB15k237', 'WN18', 'WN18RR', 'Freebase']}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'dglke'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'dglke'}
