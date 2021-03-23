# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''NEAR AI's library wrapper.

Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'nearai',
    'original_name': 'NEAR Program Synthesis',
    'url': 'https://github.com/nearai/program_synthesis'
}

DATASETS = {'program synthesis': ['AlgoLisp', 'Karel', 'NAPS']}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'nearai'}
