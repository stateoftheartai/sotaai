# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Useful utility functions to navigate the library's available resources."""
import importlib

MODEL_SOURCES = ["fastai", "keras", "mxnet", "torch", "pretrainedmodels"]
# TODO(tonioteran) Currently removed "mxnet" from DATASET_SOURCES. Need to
# restore as soon as the wrapper is done and unit test.
DATASET_SOURCES = ["tensorflow", "fastai", "keras", "torch"]  # "mxnet"


def map_dataset_source_tasks() -> dict:
  """Gathers all datasets and their respective sources and available tasks.

  Crawls through all modules to arrange entries of the form:

    <dataset-name>: {
        <name-source-1>: [<supported-task-11>, <supported-task-12>, ...],
        <name-source-2>: [<supported-task-21>, <supported-task-22>, ...],
        ...
        <name-source-n>: [<supported-task-n1>, <supported-task-n2>, ...],
    }

  Ensures duplicate removals by transforming all strings to lower case, and
  preserving the original names in an additional `original_names` dictionary.

  Returns (dict):
    Dictionary with an entry for all available datasets of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  """
  datasets_breakdown = dict()
  original_names = dict()

  for source in DATASET_SOURCES:
    wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
    for task in wrapper.DATASETS:
      for ds in wrapper.DATASETS[task]:
        original_names[ds.lower()] = ds
        ds = ds.lower()
        if ds in datasets_breakdown.keys():
          if source in datasets_breakdown[ds].keys():
            datasets_breakdown[ds][source].append(task)
          else:
            datasets_breakdown[ds][source] = [task]
        else:
          datasets_breakdown[ds] = {source: [task]}
  # Uses the entries of `original_names` as keys to store the entries from
  # the `datasets_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for dsname in datasets_breakdown:
    output_dict[original_names[dsname]] = datasets_breakdown[dsname]

  return output_dict
