# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''DGL-LifeSci's library wrapper.

Model information taken from: https://lifesci.dgl.ai/api/model.zoo.html
Dataset information taken from: https://lifesci.dgl.ai/api/data.html
'''

SOURCE_METADATA = {
    'name': 'dgllifesci',
    'original_name': 'DGL-LifeSci',
    'url': 'https://lifesci.dgl.ai/'
}

MODELS = {
    'Molecular Machine Learning': [
        'AttentiveFP', 'GAT', 'GCN', 'MGCN', 'MPNN', 'SchNet', 'Weave', 'GIN',
        'GNN OGB'
    ],
    'Graph Generation': ['DGMG', 'JTNN', 'WLN', 'ACNN']
}

DATASETS = {
    'Molecular Machine Learning': [
        'Tox21', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', 'HIV',
        'BACE', 'BBBP', 'ToxCast', 'SIDER', 'ClinTox', 'AstraZenecaChEMBL',
        'TencentQuantum', 'PubChemAromaticity', 'UnlabeledSMILES'
    ],
    'Reaction Prediction': ['USPTO', 'USPTORank'],
    'Graph Generation': ['JTVAE'],
    'Protein-ligand Binding Affinity Prediction': ['PDBBind']
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'dgllifesci'}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'dgllifesci'}
