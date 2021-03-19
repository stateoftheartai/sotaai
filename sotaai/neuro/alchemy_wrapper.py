# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Alchemy's library wrapper.

Dataset information taken from: https://alchemy.cs.washington.edu/data/
'''
from sotaai.neuro.abstractions import NeuroDataset

SOURCE_METADATA = {
    'name': 'alchemy',
    'original_name': 'Alchemy: Open Source AI',
    'url': 'https://alchemy.cs.washington.edu/'
}

DATASETS = {
    'unknown': [
        'Animals', 'Citeseer', 'Cora', 'Epinions', 'IMDB', 'Kinships',
        'Nations', 'ProteinInteraction', 'RadishRobotMapping', 'Tutorial',
        'UMLS', 'UW-CSE', 'WebKB'
    ]
}


def load_dataset(name: str) -> NeuroDataset:
  return NeuroDataset(name, 'alchemy')
