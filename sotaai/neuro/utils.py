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


def map_name_source_tasks(nametype: str, return_original_names=True) -> dict:  # pylint: disable=unused-argument
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

  sources = DATASET_SOURCES if nametype == 'datasets' else MODEL_SOURCES

  for source in sources:
    # wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
    wrapper = importlib.import_module(source + '_wrapper')
    items = wrapper.DATASETS if nametype == 'datasets' else wrapper.MODELS
    for task in items:
      for item in items[task]:
        if item in items_breakdown.keys():
          if source in items_breakdown[item].keys():
            items_breakdown[item][source].append(task)
          else:
            items_breakdown[item][source] = [task]
        else:
          items_breakdown[item] = {source: [task]}

  return items_breakdown


def map_name_sources(nametype: str, return_original_names=True) -> dict:
  '''Gathers all models/datasets and their source libraries.

  Builds a dictionary where each entry is of the form:

    <item-name>: [<source-library-1>, <source-library-2>, ...]

  Args:
    nametype (str):
      Types of names to be used, i.e., either 'models' or 'datasets'.

    return_original_names: if true return source original names, if false return
      unified (lower case) names

  Returns (dict):
    Dictionary with an entry for all available items of the above form.
  '''
  item_sources_tasks = map_name_source_tasks(nametype, return_original_names)
  item_sources = dict()

  for item in item_sources_tasks:
    item_sources[item] = list(item_sources_tasks[item].keys())

  return item_sources


def map_source_metadata() -> dict:
  '''Return a map between the source name and its original name

  Crawls through all modules to arrange entries of the form:

    <source-name>: <source-original-name>

  Returns (dict):
    Dictionary with an entry for all available items of the above form.
  '''
  items_breakdown = dict()

  sources = set(DATASET_SOURCES + MODEL_SOURCES)

  for source in sources:
    wrapper = importlib.import_module('sotaai.neuro.' + source + '_wrapper')
    items_breakdown[source] = wrapper.SOURCE_METADATA

  return items_breakdown
