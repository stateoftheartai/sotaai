# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''PyTorch Geometric's library wrapper.

Model information taken from:
https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models
Dataset information taken from:
'''

MODELS = {
    'unknown': [
        'ARGA', 'ARGVA', 'CorrectAndSmooth', 'DeepGCNLayer', 'DeepGraphInfomax',
        'DimeNet', 'GAE', 'GNNExplainer', 'GraphUNet', 'InnerProductDecoder',
        'JumpingKnowledge', 'LabelPropagation', 'Metapath2vec', 'Node2Vec',
        'RENet', 'SchNet', 'SignedGCN', 'TGNMemory', 'VGAE'
    ]
}

DATASETS = {
    'unknown': [
        'KarateClub', 'TU', 'GNNBenchmark', 'Planetoid', 'CitationFull',
        'CoraFull', 'Coauthor', 'Amazon', 'PPI', 'Reddit', 'Reddit2', 'Flickr',
        'Yelp', 'QM7', 'QM9', 'ZINC', 'MoleculeNet', 'Entities', 'GEDDataset',
        'MNISTSuperpixels', 'FAUST', 'DynamicFAUST', 'DynamicFAUST', 'ShapeNet',
        'ModelNet', 'CoMA', 'SHREC2016', 'TOSCA', 'PCPNet', 'S3DIS',
        'GeometricShapes', 'BitcoinOTC', 'ICEWS18', 'GDELT', 'DBP15k',
        'WILLOWObjectClass', 'PascalVOCKeypoints', 'PascalPF', 'SNAPDataset',
        'SuiteSparseMatrixCollection', 'TrackMLParticleTracking', 'AMiner',
        'WordNet18', 'WordNet18RR', 'WikiCS', 'WebKB', 'WikipediaNetwork',
        'Actor', 'JODIE', 'MixHopSyntheticDataset'
    ],
}
