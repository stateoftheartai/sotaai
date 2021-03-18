# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Ampligraph's library wrapper.'''

SOURCE_METADATA = {
    'id': 'ampligraph',
    'name': 'AmpliGraph',
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
