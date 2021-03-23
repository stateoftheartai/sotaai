# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
MMF https://mmf.sh wrapper module
'''

SOURCE_METADATA = {
    'name': 'mmf',
    'original_name': 'MMF',
    'url': 'https://mmf.sh'
}

MODELS = {
    'classification': [
        'ConcatBERT', 'LateFusion', 'MMBT', 'UnimodalModal', 'UnimodalText',
        'ViLBERT', 'VisualBERT'
    ]
}

DATASETS = {
    'visual_question_answering': [
        'coco',
        'hateful_memes',
        'mmimdb',
        'ocrvqa',
        'stvqa',
        'textcaps',
        'textvqa',
        'visual_entailment',
        'vizwiz',
        'vqa2',
    ]
}


def load_model(name: str):
  return {'name': name, 'source': 'mmf'}


def load_dataset(name: str):
  return {'train': {'name': name, 'source': 'mmf'}}
