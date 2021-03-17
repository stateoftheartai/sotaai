# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''decaNLP's Library wrapper.

Model information taken from:
https://github.com/salesforce/decaNLP/tree/master/models
Dataset information taken from:
'''

ALL_MODELS = [
    'CoattentivePointerGenerator', 'MultitaskQuestionAnsweringNetwork',
    'PointerGenerator', 'SelfAttentivePointerGenerator'
]

MODELS = {
    'question answering': ALL_MODELS,
    'machine translation': ALL_MODELS,
    'summarization': ALL_MODELS,
    'natural language inference': ALL_MODELS,
    'sentiment analysis': ALL_MODELS,
    'semantic role labeling': ALL_MODELS,
    'zero-shot relation extraction': ALL_MODELS,
    'goal-oriented dialogue': ALL_MODELS,
    'semantic parsing': ALL_MODELS,
    'commonsense reasoning': ALL_MODELS
}

DATASETS = {
    'question answering': ['SQuAD'],
    'machine translation': ['IWSLT'],
    'summarization': ['CNN/DM'],
    'natural language inference': ['MNLI'],
    'sentiment analysis': ['SST'],
    'semantic role labeling': ['QA-SRL'],
    'zero-shot relation extraction': ['QA-ZRE'],
    'goal-oriented dialogue': ['WOZ'],
    'semantic parsing': ['WikiSQL'],
    'commonsense reasoning': ['MWSC']
}
