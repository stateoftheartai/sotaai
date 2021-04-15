# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''decaNLP's Library wrapper.

Model information taken from:
https://github.com/salesforce/decaNLP/tree/master/models
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'decanlp',
    'original_name': 'decaNLP',
    'url': 'https://decanlp.com/'
}

ALL_MODELS = [
    'CoattentivePointerGenerator', 'MultitaskQuestionAnsweringNetwork',
    'PointerGenerator', 'SelfAttentivePointerGenerator'
]

MODELS = {
    'Question Answering': ALL_MODELS,
    'Machine Translation': ALL_MODELS,
    'Summarization': ALL_MODELS,
    'Natural Language Inference': ALL_MODELS,
    'Sentiment Analysis': ALL_MODELS,
    'Semantic Role Labeling': ALL_MODELS,
    'Zero-shot Relation Extraction': ALL_MODELS,
    'Goal-oriented Dialogue': ALL_MODELS,
    'Semantic Parsing': ALL_MODELS,
    'Commonsense Reasoning': ALL_MODELS
}

DATASETS = {
    'Question Answering': ['SQuAD'],
    'Machine Translation': ['IWSLT'],
    'Summarization': ['CNN/DM'],
    'Natural Language Inference': ['MNLI'],
    'Sentiment Analysis': ['SST'],
    'Semantic Role Labeling': ['QA-SRL'],
    'Zero-shot Relation Extraction': ['QA-ZRE'],
    'Goal-oriented Dialogue': ['WOZ'],
    'Semantic Parsing': ['WikiSQL'],
    'Commonsense Reasoning': ['MWSC']
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'decanlp'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'decanlp'}
