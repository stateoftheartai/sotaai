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
    'Unsupervised Node Classification': [
        'ProNE', 'NetMF', 'Node2Vec', 'NetSMF', 'DeepWalk', 'LINE', 'Hope',
        'SDNE', 'GraRep', 'DNGR'
    ],
    'Semi-supervised Node Classification': [
        'Graph U-Net', 'MixHop', 'DR-GAT', 'GAT', 'DGI', 'GCN', 'GraphSAGE',
        'Chebyshev'
    ],
    'Heterogeneous Node Classification': [
        'GTN', 'HAN', 'PTE', 'Metapath2vec', 'Hin2vec'
    ],
    'Edge/Link Prediction': [
        'ProNE', 'NetMF', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope', 'NetSMF',
        'SDNE'
    ],
    'Link Prediction': [
        'GATNE', 'NetMF', 'ProNE', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope',
        'GraRep'
    ],
    'Unsupervised Graph Classification': ['InfoGraph', 'Graph2Vec', 'DGK'],
    'Supervised Graph Classification': [
        'GIN', 'DiffPool', 'SortPool', 'PATCH_SAN', 'DGCNN'
    ]
}

DATASETS = {
    'Unsupervised Node Classification': [
        'BlogCatalog', 'Wikipedia', 'PPI', 'DBLP', 'Youtube'
    ],
    'Semi-supervised Node Classification': ['Cora', 'Citeseer', 'Pubmed'],
    'Graph Classification': ['MUTAG', 'IMDB-B', 'IMDB-M', 'PROEINS', 'COLLAB'],
    'Node Classification': ['DBLP', 'ACM', 'IMDB']
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'cogdl'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'cogdl'}
