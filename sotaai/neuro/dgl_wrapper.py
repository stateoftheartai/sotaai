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
    'unknown': [
        'GATNE-T', 'GNN-FiLM', 'NGCF', 'APPNP', 'ARMA', 'CapsuleNetwork',
        'VGAE', 'Tree-LSTM', 'BP-Transformer', 'TAGCN', 'STGCN', 'SIGN', 'SGC',
        'SEAL', 'SAGPool', 'RRN', 'Relational-GCN', 'PointNet', 'PointNet++',
        'DynamicEdgeConv', 'PinSAGE', 'OGB', 'Metapath2vec', 'JunctionTreeVAE',
        'DeepWalk', 'ClusterGAT', 'ClusterSAGE', 'LINE', 'DAGNN', 'Cluster-GCN'
    ],
    'node classification': [
        'GCN', 'GAT', 'GraphSAGE', 'APPNP', 'GIN', 'TAGCN', 'SGC', 'AGNN',
        'ChebNet', 'MixHop', 'GXN'
    ],
    'graph classification': ['ChebNet', 'MoNet', 'HGP-SL', 'GXN', 'GIN'],
    'community detection': ['CDGNN'],
    'representation learning': [
        'InfoGraph', 'GraphSAGE', 'GIN', 'DiffPool', 'DGI'
    ],
    'hierarchical representation learning': ['DiffPool'],
    'node embedding': ['HardGAT'],
    'graph embedding': ['HardGAT', 'HGT', 'HAN'],
    'heterogeneous graph embedding': [
        'HGT',
        'HAN',
    ],
    'text generation': ['GraphWriter'],
    'bAbI': ['GGNN',],
    'graph algorithm learning': ['GGNN'],
    'program verification': ['GGNN'],
    'recommendation': ['GC-MC'],
    'molecular machine learning': ['DimeNet', 'DimeNet++'],
    'generative models': ['GraphModel'],
    'scene graph parsing': ['SceneGraphParsing']
}

DATASETS = {
    'node classification': [
        'Stanford sentiment treebank', 'Karate club', 'Cora', 'Citeseer',
        'Pubmed', 'CoraFull', 'AIFB', 'MUTAG', 'AM', 'BGS', 'Amazon Computer',
        'Amazon Photo', 'Coauthor CS', 'Coauthor Physics', 'PPI', 'Reddit',
        'Symmetric stochastic block model mixture'
    ],
    'edge prediction': [
        'FB15k237', 'FB15k', 'WN18', 'BitcoinOTC', 'ICEWS18', 'GDELT'
    ],
    'graph prediction': [
        'QM7b', 'QM9', 'QM9Edge', 'MiniGraph', 'TU', 'LegacyTU', 'GIN'
    ]
}

ADDITIONAL_FEATURES = {
    'graph matching routines': ['astar', 'beam', 'bipartite', 'hausdorff'],
}
