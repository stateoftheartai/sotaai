# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''DGL-LifeSci's library wrapper.

Model information taken from: https://lifesci.dgl.ai/api/model.zoo.html
Dataset information taken from: https://lifesci.dgl.ai/api/data.html
'''

SOURCE_METADATA = {
    'id': 'dgllifesci',
    'name': 'DGL-LifeSci',
    'url': 'https://lifesci.dgl.ai/'
}

MODELS = {
    'molecular property prediction': [
        'AttentiveFP', 'GAT', 'GCN', 'MGCN', 'MPNN', 'SchNet', 'Weave', 'GIN',
        'GNN OGB'
    ],
    'generative models': ['DGMG', 'JTNN', 'WLN', 'ACNN']
}

DATASETS = {
    'molecular property prediction': [
        'Tox21', 'ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', 'HIV',
        'BACE', 'BBBP', 'ToxCast', 'SIDER', 'ClinTox', 'AstraZenecaChEMBL',
        'TencentQuantum', 'PubChemAromaticity', 'UnlabeledSMILES'
    ],
    'reaction prediction': ['USPTO', 'USPTORank'],
    'generative models': ['JTVAE'],
    'protein-ligand binding affinity prediction': ['PDBBind']
}
