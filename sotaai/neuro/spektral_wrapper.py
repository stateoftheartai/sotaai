# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Spektral's library wrapper.

Model information taken from: https://graphneural.network/models/
Dataset information taken from: https://graphneural.network/datasets/
'''

SOURCE_METADATA = {
    'name': 'spektral',
    'original_name': 'Spektral',
    'url': 'https://graphneural.network/'
}

MODELS = {
    'Graph Classification': ['GCN'],
    'Unknown': ['GeneralGNN'],
}

DATASETS = {
    'Unknown': [
        'Cora', 'Citeseer', 'Pubmed', 'ModelNet10', 'ModelNet40', 'OGB'
    ],
    'Inductive Representation Learning': ['GraphSage'],
    'Inductive Node Classification': ['GraphSage', 'PPI'],
    'Community Detection': ['Reddit'],
    'Graph Classification': ['MNIST'],
    'Molecular Machine Learning': ['QM7', 'QM9'],
    'Graph Kernels': ['TUDataset']
}

ADDITIONAL_FEATURES = {
    'Convolutional Layers': [
        'MessagePassing', 'AGNNConv', 'APPNPConv', 'ARMAConv', 'ChebConv',
        'CrystalConv', 'DiffusionConv', 'ECCConv', 'EdgeConv', 'GATConv',
        'GatedGraphConv', 'GCNConv', 'GeneralConv', 'GCSConv', 'GINConv',
        'GraphSageConv', 'TAGConv'
    ],
    'Pooling Layers': [
        'DiffPool', 'MinCutPool', 'SAGPool', 'TopKPool', 'GlobalAvgPool',
        'GlobalMaxPool', 'GlobalSumPool', 'GlobalAttentionPool',
        'GlobalAttnSumPool', 'SortPool'
    ],
    'Base Layers': ['InnerProduct', 'Disjoint2Batch', 'MinkowskiProduct']
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'spektral'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'spektral'}
