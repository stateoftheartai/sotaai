# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Causal Discovery Toolbox's library wrapper.'''

SOURCE_METADATA = {
    'name':
        'cdt',
    'original_name':
        'Causal Discovery Toolbox',
    'url':
        'https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html'  # pylint: disable=line-too-long
}

MODELS = {
    'Causal Pairwise Inference': [
        'ANM', 'IGCI', 'RCC', 'NCC', 'GNN', 'BivariateFit', 'Jarfo', 'CDS',
        'RECI'
    ],
    'Causal Graph Inference': [
        'CGNN', 'PC', 'GES', 'GIES', 'LiNGAM', 'CAM', 'GS', 'IAMB', 'MMPC',
        'SAM', 'CCDr'
    ],
    'Skeleton Inference': [
        'RandomizedLasso', 'Glasso', 'HSICLasso', 'FSGNN', 'RFECV', 'LinearSVR',
        'RRelief', 'ARD', 'DecisionTree'
    ],
    'Pairwise Dependency': [
        'Pearson', 'Spearman', 'KendallTau', 'NormalizedHSIC', 'MIRegression',
        'Adjusted Mutual Info', 'Normalized Mutual Info'
    ]
}

DATASETS = {'Unknown': ['tuebingen', 'sachs', 'dream4']}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'cdt'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'cdt'}
