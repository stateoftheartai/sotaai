# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Deep Graph Library's library wrapper.

Model information taken from:
- https://github.com/dmlc/dgl/tree/master/examples/pytorch
Dataset information taken from:
https://docs.dgl.ai/en/latest/api/python/dgl.data.html
'''

SOURCE_METADATA = {
    'name': 'dgl',
    'original_name': 'Deep Graph Library (DGL)',
    'url': 'https://www.dgl.ai/'
}

MODELS = {
    'Unknown': [
        'GATNE-T', 'GNN-FiLM', 'NGCF', 'APPNP', 'ARMA', 'CapsuleNetwork',
        'VGAE', 'Tree-LSTM', 'BP-Transformer', 'TAGCN', 'STGCN', 'SIGN', 'SGC',
        'SEAL', 'SAGPool', 'RRN', 'Relational-GCN', 'PointNet', 'PointNet++',
        'DynamicEdgeConv', 'PinSAGE', 'OGB', 'Metapath2vec', 'JunctionTreeVAE',
        'DeepWalk', 'ClusterGAT', 'ClusterSAGE', 'LINE', 'DAGNN', 'Cluster-GCN',
        'GC-MC'
    ],
    'Node Classification': [
        'GCN', 'GAT', 'GraphSAGE', 'APPNP', 'GIN', 'TAGCN', 'SGC', 'AGNN',
        'ChebNet', 'MixHop', 'GXN'
    ],
    'Graph Classification': ['ChebNet', 'MoNet', 'HGP-SL', 'GXN', 'GIN'],
    'Community Detection': ['CDGNN'],
    'Representation Learning': [
        'InfoGraph', 'GraphSAGE', 'GIN', 'DiffPool', 'DGI'
    ],
    'Node Embedding': ['HardGAT'],
    'Graph Embedding': ['HardGAT', 'HGT', 'HAN'],
    'Heterogeneous Graph Embedding': ['HGT', 'HAN'],
    'Text Generation': ['GraphWriter'],
    'Reasoning': ['GGNN',],
    'Graph Algorithm Learning': ['GGNN'],
    'Program Verification': ['GGNN'],
    'Molecular Machine Learning': ['DimeNet', 'DimeNet++'],
    'Graph Generation': ['GraphModel'],
    'Scene Graph Parsing': ['SceneGraphParsing']
}

DATASETS = {
    'Node Classification': [
        'Stanford sentiment treebank', 'Karate club', 'Cora', 'Citeseer',
        'Pubmed', 'CoraFull', 'AIFB', 'MUTAG', 'AM', 'BGS', 'Amazon Computer',
        'Amazon Photo', 'Coauthor CS', 'Coauthor Physics', 'PPI', 'Reddit',
        'Symmetric stochastic block model mixture'
    ],
    'Edge/Link Prediction': [
        'FB15k237', 'FB15k', 'WN18', 'BitcoinOTC', 'ICEWS18', 'GDELT'
    ],
    'Graph Prediction': [
        'QM7b', 'QM9', 'QM9Edge', 'MiniGraph', 'TU', 'LegacyTU', 'GIN'
    ]
}

ADDITIONAL_FEATURES = {
    'graph matching routines': ['astar', 'beam', 'bipartite', 'hausdorff'],
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'dgl'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'dgl'}
