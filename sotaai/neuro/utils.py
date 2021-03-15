# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Useful utility functions to navigate the library's available resources.'''
import importlib

MODEL_SOURCES = [
    'ampligraph', 'cdt', 'cogdl', 'dgl', 'dglke', 'dgllifesci', 'karateclub',
    'pytorchgeo', 'spektral'
]

DATASET_SOURCES = [
    'alchemy', 'ampligraph', 'cdt', 'cogdl', 'dgl', 'dglke', 'dgllifesci',
    'karateclub', 'nearai', 'pytorchgeo', 'spektral'
]


def map_name_source_tasks(nametype: str, return_original_names=True) -> dict:
  '''Gathers all models/datasets and their respective sources and tasks.

  Crawls through all modules to arrange entries of the form:

    <item-name>: {
        <name-source-1>: [<supported-task-11>, <supported-task-12>, ...],
        <name-source-2>: [<supported-task-21>, <supported-task-22>, ...],
        ...
        <name-source-n>: [<supported-task-n1>, <supported-task-n2>, ...],
    }

  Ensures duplicate removals by transforming all strings to lower case, and
  preserving the original names in an additional `original_names` dictionary.

  Args:
    nametype (str):
      Types of names to be used, i.e., either 'models' or 'datasets'.
    return_original_names: if true return source original names, if false return
      unified (lower case) names

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  items_breakdown = dict()
  original_names = dict()

  sources = DATASET_SOURCES if nametype == 'datasets' else MODEL_SOURCES

  for source in sources:
    # wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
    wrapper = importlib.import_module(source + '_wrapper')
    items = wrapper.DATASETS if nametype == 'datasets' else wrapper.MODELS
    for task in items:
      for item in items[task]:
        original_names[item.lower()] = item
        item = item.lower()
        if item in items_breakdown.keys():
          if source in items_breakdown[item].keys():
            items_breakdown[item][source].append(task)
          else:
            items_breakdown[item][source] = [task]
        else:
          items_breakdown[item] = {source: [task]}

  # TODO When original_names are replaced, the original name replaced
  # is the last one added to the original_names dict e.g. If vgg exists as
  # VGG and vgg in different sources, the original_names dict will only keep
  # one of those two. We need to fix this evenutally.

  if not return_original_names:
    return items_breakdown

  # Uses the entries of `original_names` as keys to store the entries from
  # the `items_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for itemname in items_breakdown:
    output_dict[original_names[itemname]] = items_breakdown[itemname]

  return output_dict
