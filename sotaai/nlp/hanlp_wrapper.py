# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''HanLP's wrapper module.

Model information taken from:
- https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html
Dataset information taken from:
- https://hanlp.hankcs.com/docs/api/hanlp/datasets/index.html
'''

SOURCE_METADATA = {
    'name': 'hanlp',
    'original_name': 'HanLP',
    'url': 'https://hanlp.hankcs.com/docs/index.html'
}

MODELS = {
    'Machine Translation': ['Electra', 'mt5', 'XLM-R'],
    'Sentence Boundary Detection': ['EOS'],
    'Tokenization Tagging': ['ConvModel', 'ALBERT', 'BERT'],
    'Part-of-speech Tagging': ['BiLSTM', 'ALBERT'],
    'Named Entity Recognition': ['BERT', 'ALBERT'],
    'Dependency Parsing': ['BiaffineLSTM', 'BiaffineSDP'],
}

DATASETS = {
    'Sentence Boundary Detection': [
        'EUROPARL_V7_DE_EN_EN_SENTENCES', 'SETIMES2_EN_HR_HR_SENTENCES'
    ],
    'Tokenization Tagging': ['sighan2005', 'CTB6', 'CTB8', 'CTTB9'],
    'Part-of-speech Tagging': ['CTB6', 'CTB8', 'CTTB9'],
    'Named Entity Recognition': [
        'CoNLL 2003', 'MSRA', 'OntoNotes5', 'Resume', 'Weibo'
    ],
    'Dependency Parsing': ['ChineseTreebank', 'EnglishTreebank'],
    'Semantic Role Labeling': ['CoNLL 2012', 'OntoNotes5'],
    'Constituency Parsing': ['CTB8', 'CTB9', 'PTB'],
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'hanlp'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'hanlp'}
