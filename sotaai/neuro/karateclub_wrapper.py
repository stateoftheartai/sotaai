# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Karate Club's library wrapper.

Model information taken from:
https://karateclub.readthedocs.io/en/latest/modules/root.html
Dataset information taken from:
https://karateclub.readthedocs.io/en/latest/notes/introduction.html
'''

SOURCE_METADATA = {
    'id': 'karateclub',
    'name': 'Karate Club',
    'url': 'https://karateclub.readthedocs.io/en/latest/'
}

MODELS = {
    # ---
    'community detection': [
        'EgoNetSplitter', 'DANMF', 'NNSED', 'MNMF', 'BigClam', 'SymmNMF',
        'GEMSEC', 'EdMot', 'SCD', 'LabelPropagation'
    ],
    # - Community detection subtasks:
    'overlapping community detection': [
        'EgoNetSplitter', 'DANMF', 'NNSED', 'MNMF', 'BigClam', 'SymmNMF'
    ],
    'non-overlapping community detection': [
        'GEMSEC', 'EdMot', 'SCD', 'LabelPropagation'
    ],
    # ---
    'node embedding': [
        'SocioDim', 'RandNE', 'GLEE', 'Diff2Vec', 'NodeSketch', 'NetMF',
        'BoostNE', 'Walklets', 'GraRep', 'DeepWalk', 'Node2Vec', 'NMFADMM',
        'LaplacianEigenmaps', 'GraphWave', 'Role2Vec', 'FeatherNode', 'AE',
        'MUSAE', 'SINE', 'BANE', 'TENE', 'TADW', 'FSCNMF', 'ASNE', 'NEU'
    ],
    # - Node embedding subtasks:
    'neighborhood-based node embedding': [
        'SocioDim', 'RandNE', 'GLEE', 'Diff2Vec', 'NodeSketch', 'NetMF',
        'BoostNE', 'Walklets', 'GraRep', 'DeepWalk', 'Node2Vec', 'NMFADMM',
        'LaplacianEigenmaps'
    ],
    'structural node embedding': ['GraphWave', 'Role2Vec'],
    'attributed node embedding': [
        'FeatherNode', 'AE', 'MUSAE', 'SINE', 'BANE', 'TENE', 'TADW', 'FSCNMF',
        'ASNE'
    ],
    'meta node embedding': ['NEU',],
    # ---
    'graph embedding': [
        'LDP', 'FeatherGraph', 'IGE', 'GeoScattering', 'GL2Vec', 'NetLSD', 'SF',
        'FGSD', 'Graph2Vec'
    ],
    # - Graph embedding subtasks:
    'whole graph embedding': [
        'LDP', 'FeatherGraph', 'IGE', 'GeoScattering', 'GL2Vec', 'NetLSD', 'SF',
        'FGSD', 'Graph2Vec'
    ],
    # ---
}

DATASETS = {
    'community detection': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ],
    'graph embedding': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ],
    'node embedding': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ]
}
