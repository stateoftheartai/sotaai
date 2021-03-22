# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Stanza's wrapper module.

Model information taken from:
https://github.com/stanfordnlp/stanza/tree/main/stanza/models
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'stanza',
    'original_name': 'Stanza',
    'url': 'https://stanfordnlp.github.io/stanza/'
}

MODELS = {
    'text classification': ['CNNClassifier'],
    'dependency parsing': ['Parser'],
    'named entity recongnition': ['NERTagger'],
    'part-of-speech tagging': ['Tagger'],
    'tokenization': ['Tokenizer'],
    'clinical syntactic analysis': ['craft', 'genia', 'mimic'],
    'clinical named entity recognition': [
        'anatem', 'bc5cdr', 'bc4chemd', 'bionlp13cg', 'jnlpba', 'linnaeus',
        'ncbi_disease', 's800', 'i2b2', 'radiology'
    ]
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'stanza'}
