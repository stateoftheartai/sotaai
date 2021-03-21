# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Fairseq wrapper module.

Model information taken from:
http://github.com/pytorch/fairseq/blob/master/examples/language_model/README.md
http://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'fairseq',
    'original_name': 'decaNLP',
    'url': 'https://github.com/pytorch/fairseq'
}

MODELS = {
    'neural machine translation': [
        'conv.wmt14.en-de', 'conv.wmt14.en-fr', 'conv.wmt17.en-de',
        'transformer.wmt14.en-fr', 'transformer.wmt16.en-de',
        'transformer.wmt18.en-de', 'transformer.wmt19.de-en',
        'transformer.wmt19.de-en.single_model', 'transformer.wmt19.en-de',
        'transformer.wmt19.en-de.single_model', 'transformer.wmt19.en-ru',
        'transformer.wmt19.en-ru.single_model', 'transformer.wmt19.ru-en',
        'transformer.wmt19.ru-en.single_model'
    ],
    'language modeling': [
        'transformer_lm.gbw.adaptive_huge', 'transformer_lm.wiki103.adaptive',
        'transformer_lm.wmt19.de', 'transformer_lm.wmt19.en',
        'transformer_lm.wmt19.ru', 'conv.stories', 'conv.stories.pretrained'
    ],
}


def load_model(name: str) -> dict:
  '''Gets a model directly from Fairseq library.

  Args:
    name: Name of the model to be gotten.

  Returns:
    Fairseq model.
  '''
  return {'name': name, 'source': 'fairseq'}
