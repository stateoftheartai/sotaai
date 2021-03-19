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
    'causal pairwise inference': [
        'ANM', 'IGCI', 'RCC', 'NCC', 'GNN', 'BivariateFit', 'Jarfo', 'CDS',
        'RECI'
    ],
    'causal graph inference': [
        'CGNN', 'PC', 'GES', 'GIES', 'LiNGAM', 'CAM', 'GS', 'IAMB', 'MMPC',
        'SAM', 'CCDr'
    ],
    'skeleton inference': [
        'RandomizedLasso', 'Glasso', 'HSICLasso', 'FSGNN', 'RFECV', 'LinearSVR',
        'RRelief', 'ARD', 'DecisionTree'
    ],
    'pairwise dependency': [
        'Pearson', 'Spearman', 'KendallTau', 'NormalizedHSIC', 'MIRegression',
        'Adjusted Mutual Info', 'Normalized Mutual Info'
    ]
}

DATASETS = {'unknown': ['tuebingen', 'sachs', 'dream4']}
