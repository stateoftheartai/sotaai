# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Hugging Face's Library wrapper.

Model information taken from:
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'huggingface',
    'original_name': 'Hugging Face',
    'url': 'https://huggingface.co/'
}

MODELS = {
    'Conditional Generation': [
        'BART', 'Blenderbot', 'FSMT', 'LED', 'M2M100', 'MBart', 'MBart-50',
        'MT5', 'Pegasus', 'ProphetNet', 'Speech2Text', 'T5'
    ],
    'Next Sentence Prediction': ['BERT', 'BORT', 'MobileBERT'],
    'Language Modeling': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'Funnel Transformer',
        'LayoutLM', 'Longformer', 'MobileBERT', 'MPNet', 'OpenAI GPT',
        'OpenAI GPT2', 'Reformer', 'RetriBERT', 'RoBERTa', 'SqueezeBERT',
        'TAPAS', 'Transformer XL', 'XLM', 'XLM-RoBERTa', 'XLNet'
    ],
    'Causal Language Modeling': [
        'BART', 'BigBird', 'Blenderbot', 'CamemBERT', 'MarianMT', 'GPT Neo',
        'Pegasus', 'ProphetNet', 'RoBERTa', 'XLM-RoBERTa'
    ],
    'Sequence Classification': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'CTRL', 'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT',
        'Funnel Transformer', 'I-BERT', 'LayoutLM', 'LED', 'Longformer',
        'MBart', 'MBart-50', 'MobileBERT', 'MPNet', 'OpenAI GPT', 'OpenAI GPT2',
        'Reformer', 'RoBERTa', 'SqueezeBERT', 'TAPAS', 'Transformer XL', 'XLM',
        'XLM-RoBERTa', 'XLNet'
    ],
    'Token Classification': [
        'ALBERT', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT', 'DeBERTa',
        'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT', 'Funnel Transformer',
        'I-BERT', 'LayoutLM', 'Longformer', 'MobileBERT', 'MPNet', 'RoBERTa',
        'SqueezeBERT', 'XLM', 'XLM-RoBERTa', 'XLNet'
    ],
    'Question Answering': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT',
        'Funnel Transformer', 'I-BERT', 'LED', 'Longformer', 'LXMERT',
        'MobileBERT', 'MBart', 'MBart-50', 'MobileBERT', 'MPNet', 'Reformer',
        'RoBERTa', 'SqueezeBERT', 'TAPAS', 'XLM', 'XLM-ProphetNet',
        'XLM-RoBERTa', 'XLNet'
    ],
    'Text Classification': ['Bertweet',],
    'Part-of-speech Tagging': ['Bertweet',],
    'Named Entity Recognition': ['Bertweet',],
    'Summarization': ['BARThez', 'BertGeneration'],
    'Text Generation': ['DialoGPT', 'RAG', 'XLM-ProphetNet'],
    'Token Generation': ['RAG',],
    'Machine Translation': ['BertGeneration',],
    'Sentence Splitting': ['BertGeneration',],
    'Sentence Fusion': ['BertGeneration',],
    'Multilingual': ['herBERT', 'PhoBERT'],
    'Image Classification': ['Vision Transformer (ViT)',],
    'Speech Recognition': ['Wav2Vec2', 'XLSR-Wav2Vec2'],
}

DATASETS = {}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'huggingface'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'huggingface'}
