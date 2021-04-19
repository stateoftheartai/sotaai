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
    'name': 'karateclub',
    'original_name': 'Karate Club',
    'url': 'https://karateclub.readthedocs.io/en/latest/'
}

MODELS = {
    # ---
    'Community Detection': [
        'EgoNetSplitter', 'DANMF', 'NNSED', 'MNMF', 'BigClam', 'SymmNMF',
        'GEMSEC', 'EdMot', 'SCD', 'LabelPropagation'
    ],
    # - Community detection subtasks:
    # 'Overlapping Community Detection': [
    #     'EgoNetSplitter', 'DANMF', 'NNSED', 'MNMF', 'BigClam', 'SymmNMF'
    # ],
    # 'Non-overlapping Community Detection': [
    #     'GEMSEC', 'EdMot', 'SCD', 'LabelPropagation'
    # ],
    # ---
    'Node Embedding': [
        'SocioDim', 'RandNE', 'GLEE', 'Diff2Vec', 'NodeSketch', 'NetMF',
        'BoostNE', 'Walklets', 'GraRep', 'DeepWalk', 'Node2Vec', 'NMFADMM',
        'LaplacianEigenmaps', 'GraphWave', 'Role2Vec', 'FeatherNode', 'AE',
        'MUSAE', 'SINE', 'BANE', 'TENE', 'TADW', 'FSCNMF', 'ASNE', 'NEU'
    ],
    # - Node embedding subtasks:
    'Neighborhood-based Node Embedding': [
        'SocioDim', 'RandNE', 'GLEE', 'Diff2Vec', 'NodeSketch', 'NetMF',
        'BoostNE', 'Walklets', 'GraRep', 'DeepWalk', 'Node2Vec', 'NMFADMM',
        'LaplacianEigenmaps'
    ],
    'Structural Node Embedding': ['GraphWave', 'Role2Vec'],
    # 'attributed node embedding': [
    #     'FeatherNode', 'AE', 'MUSAE', 'SINE', 'BANE', 'TENE', 'TADW',
    #     'FSCNMF',
    #     'ASNE'
    # ],
    # ---
    'Graph Embedding': [
        'LDP',
        'FeatherGraph',
        'IGE',
        'GeoScattering',
        'GL2Vec',
        'NetLSD',
        'SF',
        'FGSD',
        'Graph2Vec',
    ],
    # - Graph embedding subtasks:
    # 'Whole Graph Embedding': [
    #     'LDP', 'FeatherGraph', 'IGE', 'GeoScattering', 'GL2Vec', 'NetLSD',
    #     'SF',
    #     'FGSD', 'Graph2Vec'
    # ],
    # ---
}

DATASETS = {
    'Community Detection': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ],
    'Graph Embedding': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ],
    'Node Embedding': [
        'facebook', 'twitch', 'wikipedia', 'github', 'lastfm', 'deezer',
        'reddit10k'
    ]
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'karateclub'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'karateclub'}
