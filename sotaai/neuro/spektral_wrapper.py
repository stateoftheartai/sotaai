# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Spektral's library wrapper.

Model information taken from: https://graphneural.network/models/
Dataset information taken from: https://graphneural.network/datasets/
'''

SOURCE_METADATA = {
    'name': 'spektral',
    'area': 'neuro',
    'original_name': 'Spektral',
    'url': 'https://graphneural.network/'
}

MODELS = {
    'classification': ['GCN'],
    'unknown': ['GeneralGNN'],
}

DATASETS = {
    'unknown': [
        'Cora', 'Citeseer', 'Pubmed', 'ModelNet10', 'ModelNet40', 'OGB'
    ],
    'inductive representation learning': ['GraphSage'],
    'inductive node classification': ['GraphSage', 'PPI'],
    'community detection': ['Reddit'],
    'graph signal classification': ['MNIST'],
    'molecular machine learning': ['QM7', 'QM9'],
    'graph kernels': ['TUDataset']
}

ADDITIONAL_FEATURES = {
    'convolutional layers': [
        'MessagePassing', 'AGNNConv', 'APPNPConv', 'ARMAConv', 'ChebConv',
        'CrystalConv', 'DiffusionConv', 'ECCConv', 'EdgeConv', 'GATConv',
        'GatedGraphConv', 'GCNConv', 'GeneralConv', 'GCSConv', 'GINConv',
        'GraphSageConv', 'TAGConv'
    ],
    'pooling layers': [
        'DiffPool', 'MinCutPool', 'SAGPool', 'TopKPool', 'GlobalAvgPool',
        'GlobalMaxPool', 'GlobalSumPool', 'GlobalAttentionPool',
        'GlobalAttnSumPool', 'SortPool'
    ],
    'base layers': ['InnerProduct', 'Disjoint2Batch', 'MinkowskiProduct']
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'spektral'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'spektral'}
