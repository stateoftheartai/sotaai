# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''CogDL's library wrapper.

Model information taken from:
http://keg.cs.tsinghua.edu.cn/cogdl/methods.html
Dataset information taken from:
http://keg.cs.tsinghua.edu.cn/cogdl/datasets.html
'''

MODELS = {
    'unsupervised node classification': [
        'ProNE', 'NetMF', 'Node2Vec', 'NetSMF', 'DeepWalk', 'LINE', 'Hope',
        'SDNE', 'GraRep', 'DNGR'
    ],
    'semi-supervised node classification': [
        'Graph U-Net', 'MixHop', 'DR-GAT', 'GAT', 'DGI', 'GCN', 'GraphSAGE',
        'Chebyshev'
    ],
    'heterogeneous node classification': [
        'GTN', 'HAN', 'PTE', 'Metapath2vec', 'Hin2vec'
    ],
    'edge prediction': [
        'ProNE', 'NetMF', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope', 'NetSMF',
        'SDNE'
    ],
    'multiplex link prediction': [
        'GATNE', 'NetMF', 'ProNE', 'Node2Vec', 'DeepWalk', 'LINE', 'Hope',
        'GraRep'
    ],
    'unsupervised graph classification': ['InfoGraph', 'Graph2Vec', 'DGK'],
    'supervised graph classification': [
        'GIN', 'DiffPool', 'SortPool', 'PATCH_SAN', 'DGCNN'
    ]
}

DATASETS = {
    'unsupervised node classification': [
        'BlogCatalog', 'Wikipedia', 'PPI', 'DBLP', 'Youtube'
    ],
    'semi-supervised node classification': ['Cora', 'Citeseer', 'Pubmed'],
    'graph classification': ['MUTAG', 'IMDB-B', 'IMDB-M', 'PROEINS', 'COLLAB'],
    'multiplex node classification': ['DBLP', 'ACM', 'IMDB']
}
