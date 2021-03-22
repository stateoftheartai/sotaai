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
    'mtl': ['Electra', 'mt5', 'XLM-R'],
    'sentence boundary detection': ['EOS'],
    'tokenization tagging': ['ConvModel', 'ALBERT', 'BERT'],
    'part-of-speech tagging': ['BiLSTM', 'ALBERT'],
    'named entity recognition': ['BERT', 'ALBERT'],
    'dependency parsing': ['BiaffineLSTM'],
    'sdp': ['BiaffineSDP'],
}

DATASETS = {
    'sentence boundary detection': [
        'EUROPARL_V7_DE_EN_EN_SENTENCES', 'SETIMES2_EN_HR_HR_SENTENCES'
    ],
    'tokenization tagging': ['sighan2005', 'CTB6', 'CTB8', 'CTTB9'],
    'part-of-speech tagging': ['CTB6', 'CTB8', 'CTTB9'],
    'named entity recognition': [
        'CoNLL 2003', 'MSRA', 'OntoNotes5', 'Resume', 'Weibo'
    ],
    'dependency parsing': ['ChineseTreebank', 'EnglishTreebank'],
    'semantic role labeling': ['CoNLL 2012', 'OntoNotes5'],
    'constituency parsing': ['CTB8', 'CTB9', 'PTB'],
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'hanlp'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'hanlp'}
