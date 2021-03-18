# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Allen NLP's Library wrapper.

Model information taken from:
https://github.com/allenai/allennlp-models/tree/main/allennlp_models
Dataset information taken from:
'''

MODELS = {
    'classification': ['BiattentiveClassificationNetwork'],
    'coreference resolution': ['CoarseToFineCoref'],
    'text summarization': [
        'BART', 'Seq2Seq2Encode', 'ComposedSeq2Seq', 'CopyNet', 'SimpleSeq2Seq'
    ],
    'text generation': [
        'BART', 'Seq2Seq2Encode', 'ComposedSeq2Seq', 'CopyNet', 'SimpleSeq2Seq'
    ],
    'language modeling': [
        'NextTokenLM', 'MaskedLanguageModel', 'LanguageModel',
        'BidirectionalLanguageModel'
    ],
    'multiple choice': ['RoBERTa'],
    'pair classification': ['ESIM', 'DecomposableAttention', 'BiPM'],
    'reading comprehension': [
        'TransformerQA', 'QANet', 'NumericallyAugmentedQANet', 'DialogQA',
        'BiDAFEnsemble', 'BidirectionalAttentionFlow'
    ],
    'structured prediction': [
        'SrlBERT', 'BiaffineDependencyParser', 'SpanConstituencyParser'
    ],
    'vision': ['VisualEntailmentModel', 'VisionTextModel', 'VQAVilBERT']
}

DATASETS = {
    'classification': ['StanfordSentimentTreebank'],
    'coreference resolution': ['CoNLL', 'PreCo', 'Winobias'],
    'text summarization': ['CNN/DailyMail'],
    'language modeling': [],
    'multiple choice': ['SWAG', 'PIQA', 'CommonsenseQA'],
    'pair classification': ['QuoraParaphrase', 'SNLI'],
    'reading comprehension': ['TriviaQA', 'SQuAD', 'QuAC', 'Qangaroo', 'DROP'],
    'structured prediction': [
        'CoNLL', 'EnglishOntoNotes', 'SemEvalSDP', 'PennTreeBank'
    ],
    'tagging': ['EnglishOntoNotes', 'CCGbank', 'CoNLL'],
    'vision': ['VQAv2', 'GQA']
}
