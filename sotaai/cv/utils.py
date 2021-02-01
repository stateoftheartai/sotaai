# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Useful utility functions to navigate the library"s available resources."""
# TODO(tonioteran) Deprecate specific dataset/model functions for the
# generalized version.
import importlib
import mxnet as mx

# TODO(tonioteran) Currently removed "mxnet" and "pretrainedmodels" from
# MODEL_SOURCES. Need to restore as soon as the wrapper is done and unit test.
MODEL_SOURCES = ["fastai", "keras", "torch"]  # "mxnet", "pretrainedmodels"

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


def map_dataset_tasks() -> dict:
  """Gathers all datasets and their supported tasks.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: [<supported-task-1>, <supported-task-2>, ...]

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  """
  dataset_sources_tasks = map_dataset_source_tasks()
  dataset_tasks = dict()

  for ds in dataset_sources_tasks:
    ds_tasks = []

    for source in dataset_sources_tasks[ds].keys():
      for t in dataset_sources_tasks[ds][source]:
        ds_tasks.append(t)
    ds_tasks = list(set(ds_tasks))
    dataset_tasks[ds] = ds_tasks

  return dataset_tasks


def map_dataset_sources() -> dict:
  """Gathers all datasets and their source libraries.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: [<source-library-1>, <source-library-2>, ...]

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.
  """
  dataset_sources_tasks = map_dataset_source_tasks()
  dataset_sources = dict()

  for ds in dataset_sources_tasks:
    dataset_sources[ds] = list(dataset_sources_tasks[ds].keys())

  return dataset_sources


def map_dataset_info() -> dict:
  """Gathers all datasets, listing supported tasks and source libraries.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: {
          "tasks": [<supported-task-1>, <supported-task-2>, ...],
          "sources": [<supported-task-1>, <supported-task-2>, ...]
      }

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.
  """
  dataset_tasks = map_dataset_tasks()
  dataset_sources = map_dataset_sources()
  dataset_info = dict()

  for ds in dataset_tasks:
    dataset_info[ds] = {
        "sources": dataset_sources[ds],
        "tasks": dataset_tasks[ds]
    }

  return dataset_info


def map_name_source_tasks(nametype: str) -> dict:
  """Gathers all models/datasets and their respective sources and tasks.

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
      Types of names to be used, i.e., either "models" or "datasets".

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  """
  items_breakdown = dict()
  original_names = dict()

  sources = DATASET_SOURCES if nametype == "datasets" else MODEL_SOURCES

  for source in sources:
    wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
    items = wrapper.DATASETS if nametype == "datasets" else wrapper.MODELS
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
  # Uses the entries of `original_names` as keys to store the entries from
  # the `items_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for itemname in items_breakdown:
    output_dict[original_names[itemname]] = items_breakdown[itemname]

  return output_dict


def map_name_tasks(nametype: str) -> dict:
  """Gathers all models/datasets and their supported tasks.

  Builds a dictionary where each entry is of the form:

    <item-name>: [<supported-task-1>, <supported-task-2>, ...]

  Args:
    nametype (str):
      Types of names to be used, i.e., either "models" or "datasets".

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  """
  item_sources_tasks = map_name_source_tasks(nametype)
  item_tasks = dict()

  for item in item_sources_tasks:
    it_tasks = []

    for source in item_sources_tasks[item].keys():
      for t in item_sources_tasks[item][source]:
        it_tasks.append(t)
    it_tasks = list(set(it_tasks))
    item_tasks[item] = it_tasks

  return item_tasks


def map_name_sources(nametype: str) -> dict:
  """Gathers all models/datasets and their source libraries.

  Builds a dictionary where each entry is of the form:

    <item-name>: [<source-library-1>, <source-library-2>, ...]

  Args:
    nametype (str):
      Types of names to be used, i.e., either "models" or "datasets".

  Returns (dict):
    Dictionary with an entry for all available items of the above form.
  """
  item_sources_tasks = map_name_source_tasks(nametype)
  item_sources = dict()

  for item in item_sources_tasks:
    item_sources[item] = list(item_sources_tasks[item].keys())

  return item_sources


def map_name_info(nametype: str) -> dict:
  """Gathers all items, listing supported tasks and source libraries.

  Builds a dictionary where each entry is of the form:

      <item-name>: {
          "tasks": [<supported-task-1>, <supported-task-2>, ...],
          "sources": [<supported-task-1>, <supported-task-2>, ...]
      }

  Returns (dict):
      Dictionary with an entry for all available items of the above form.
  """
  item_tasks = map_name_tasks(nametype)
  item_sources = map_name_sources(nametype)
  item_info = dict()

  for item in item_tasks:
    item_info[item] = {"sources": item_sources[item], "tasks": item_tasks[item]}

  return item_info


def map_datasets_by_source() -> dict:
  """Print the list of datasets per source (mini-aa). The list printed was added
     to JIRA for future reference.
  Returns (void)
  """

  ds_to_sources = map_dataset_sources()
  ds_by_source = {}

  for ds in ds_to_sources:
    sources = ds_to_sources[ds]

    # @author HO
    # By manual inspection we saw that all datasets that exists in multiple
    # mini-aa have "tensorflow" in common, also Tensorflow has a lot of datasets
    # we need to wrap, thus Tensorflow was selected as the default source.

    source = "tensorflow"
    if len(sources) == 1:
      source = sources[0]
    if source not in ds_by_source:
      ds_by_source[source] = []
    ds_by_source[source].append(ds)

  for miniaa in ds_by_source:
    print(miniaa)
    for i, dataset in enumerate(ds_by_source[miniaa]):
      print("   " + str(i) + " " + dataset)


def extract_source_from_model(model) -> str:
  """Returns the source library"s name from a model object.

  Args:
    model:
      Model object directly instantiated from a source library. Type is
      dependent on the source library.

  Returns:
    String with the name of the source library.
  """
  if "torchvision" in str(type(model)):
    return "torchvision"
  if "mxnet" in str(type(model)):
    return "mxnet"
  if "keras" in str(type(model)):
    return "keras"
  raise NotImplementedError(
      "Need source extraction implementation for this type of model!")


def flatten_model(model) -> list:
  """Returns a list with the model"s layers.

  Some models are built with blocks of layers. This function flattens the
  blocks and returns a list of all layers of model. One of its uses is to find
  the number of layers and parameters for a model in a programatic way.

  Args:
    model:
      Model object directly instantiated from a source library. Type is
      dependent on the source library.

  Returns:
    A list of layers, which depend on the model"s source library.
  """
  source = extract_source_from_model(model)
  if source in ["keras"]:
    return list(model.submodules)

  layers = []
  flatten_model_recursively(model, source, layers)
  return layers


def flatten_model_recursively(block, source: str, layers: list):
  """Recursive helper function to flatten a model"s layers onto a list.

  Args:
    block:
      Model object directly instantiated from a source library, or a block of
      that model. Type is dependent on the source library.
    source: (string)
      The name of the model"s source library.
    layers: (list)
      The list of layers to be recursively filled.

  TODO(tonioteran,hugoochoa) Clean this up and unit test! This code seems
  pretty messy...
  """
  if source == "mxnet":
    bottleneck_layer = mx.gluon.model_zoo.vision.BottleneckV1
    list1 = dir(bottleneck_layer)
    if "features" in dir(block):
      flatten_model_recursively(block.features, source, layers)

    elif "HybridSequential" in str(type(block)):
      for j in block:
        flatten_model_recursively(j, source, layers)

    elif "Bottleneck" in str(type(block)):
      list2 = dir(block)
      for ll in list1:
        list2.remove(ll)
      subblocks = [x for x in list2 if not x.startswith("_")]
      for element in subblocks:
        attr = getattr(block, element)
        flatten_model_recursively(attr, source, layers)
    else:
      layers.append(block)

  else:
    for child in block.children():
      obj = str(type(child))
      if "container" in obj or "torch.nn" not in obj:
        flatten_model_recursively(child, source, layers)
      else:
        layers.append(child)
