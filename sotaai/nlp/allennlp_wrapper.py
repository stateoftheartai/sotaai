# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Allen NLP's Library wrapper.

Model information taken from:
https://github.com/allenai/allennlp-models/tree/main/allennlp_models
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'allennlp',
    'original_name': 'AllenNLP',
    'url': 'https://allennlp.org/'
}

MODELS = {
    'Text Classification': ['BiattentiveClassificationNetwork'],
    'Coreference Resolution': ['CoarseToFineCoref'],
    'Text Summarization': [
        'BART', 'Seq2Seq2Encode', 'ComposedSeq2Seq', 'CopyNet', 'SimpleSeq2Seq'
    ],
    'Text Generation': [
        'BART', 'Seq2Seq2Encode', 'ComposedSeq2Seq', 'CopyNet', 'SimpleSeq2Seq'
    ],
    'Language Modeling': [
        'NextTokenLM', 'MaskedLanguageModel', 'LanguageModel',
        'BidirectionalLanguageModel'
    ],
    'Question Answering': ['RoBERTa'],
    'Pair Classification': ['ESIM', 'DecomposableAttention', 'BiPM'],
    'Reading Comprehension': [
        'TransformerQA', 'QANet', 'NumericallyAugmentedQANet', 'DialogQA',
        'BiDAFEnsemble', 'BidirectionalAttentionFlow'
    ],
    'Structured Prediction': [
        'SrlBERT', 'BiaffineDependencyParser', 'SpanConstituencyParser'
    ],
    'Vision': ['VisualEntailmentModel', 'VisionTextModel', 'VQAVilBERT']
}

DATASETS = {
    'Text Classification': ['StanfordSentimentTreebank'],
    'Coreference Resolution': ['CoNLL', 'PreCo', 'Winobias'],
    'Text Summarization': ['CNN/DailyMail'],
    'Language Modeling': [],
    'Question Answering': ['SWAG', 'PIQA', 'CommonsenseQA'],
    'Pair Classification': ['QuoraParaphrase', 'SNLI'],
    'Reading Comprehension': ['TriviaQA', 'SQuAD', 'QuAC', 'Qangaroo', 'DROP'],
    'Structured Prediction': [
        'CoNLL', 'EnglishOntoNotes', 'SemEvalSDP', 'PennTreeBank'
    ],
    'Tagging': ['EnglishOntoNotes', 'CCGbank', 'CoNLL'],
    'Vision': ['VQAv2', 'GQA']
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'allennlp'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'allennlp'}
