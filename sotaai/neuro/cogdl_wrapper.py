# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''CogDL's library wrapper.

Model information taken from:
http://keg.cs.tsinghua.edu.cn/cogdl/methods.html
Dataset information taken from:
http://keg.cs.tsinghua.edu.cn/cogdl/datasets.html
'''

SOURCE_METADATA = {
    'name': 'cogdl',
    'original_name': 'CogDL: Deep Learning on Graphs',
    'url': 'http://keg.cs.tsinghua.edu.cn/cogdl/'
}

MODELS = {
    'unsupervised_node_classification': [
        'ProNE', 'NetMF', 'Node2Vec', 'NetSMF', 'DeepWalk', 'LINE', 'Hope',
        'SDNE', 'GraRep', 'DNGR'
    ],
    'semi-supervised_node_classification': [
        'Graph U-Net', 'MixHop', 'DR-GAT', 'GAT', 'DGI', 'GCN', 'GraphSAGE',
        'Chebyshev'
    ],
    'heterogeneous_node_classification': [
        'GTN', 'HAN', 'PTE', 'Metapath2vec', 'Hin2vec'
    ],
    'edge_prediction': [
        'ProNE', 'NetMF', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope', 'NetSMF',
        'SDNE'
    ],
    'multiplex_link_prediction': [
        'GATNE', 'NetMF', 'ProNE', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope',
        'GraRep'
    ],
    'unsupervised_graph_classification': ['InfoGraph', 'Graph2Vec', 'DGK'],
    'supervised_graph_classification': [
        'GIN', 'DiffPool', 'SortPool', 'PATCH_SAN', 'DGCNN'
    ]
}

DATASETS = {
    'unsupervised_node_classification': [
        'BlogCatalog', 'Wikipedia', 'PPI', 'DBLP', 'Youtube'
    ],
    'semi-supervised_node_classification': ['Cora', 'Citeseer', 'Pubmed'],
    'graph_classification': ['MUTAG', 'IMDB-B', 'IMDB-M', 'PROEINS', 'COLLAB'],
    'multiplex_node_classification': ['DBLP', 'ACM', 'IMDB']
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'cogdl'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'cogdl'}
