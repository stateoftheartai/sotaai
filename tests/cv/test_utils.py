# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Unit testing the utility functions."""
import unittest
import importlib
from sotaai.cv import utils


class TestCvUtils(unittest.TestCase):
  """The the utils sub-module for CV."""

  def test_map_dataset_source_tasks(self):
    """Make sure the dataset map correctly encapsulates all info."""
    ds_to_sourcetasks = utils.map_dataset_source_tasks()

    # Make sure that every dataset is accounted for by iterating over all
    # sources and ensuring they have a corresponding key.
    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sourcetasks.keys():
            self.assertTrue(source in ds_to_sourcetasks[ds].keys())

  def test_map_dataset_tasks(self):
    """Make sure the map from dataset to tasks correctly encapsulates info."""
    ds_to_tasks = utils.map_dataset_tasks()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_tasks.keys():
            self.assertTrue(task in ds_to_tasks[ds])

  def test_map_dataset_sources(self):
    """Ensure map from dataset name to available sources is correct."""
    ds_to_sources = utils.map_dataset_sources()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sources.keys():
            self.assertTrue(source in ds_to_sources[ds])

  def test_map_dataset_info(self):
    """Ensure tasks and sources are adequately parsed."""
    ds_to_info = utils.map_dataset_info()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_info.keys():
            self.assertTrue(task in ds_to_info[ds]["tasks"])
            self.assertTrue(source in ds_to_info[ds]["sources"])

  def test_map_name_source_tasks(self):
    """Test for both datasets and models."""
    # Try first for datasets.
    ds_to_sourcetasks = utils.map_name_source_tasks("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sourcetasks.keys():
            self.assertTrue(source in ds_to_sourcetasks[ds].keys())

    # Now for models.
    model_to_sourcetasks = utils.map_name_source_tasks("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sourcetasks.keys():
            self.assertTrue(source in model_to_sourcetasks[model].keys())

  def test_map_name_tasks(self):
    """Make sure the map from model/dataset to tasks encapsulates info."""
    # First for datasets.
    ds_to_tasks = utils.map_name_tasks("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_tasks.keys():
            self.assertTrue(task in ds_to_tasks[ds])

    # Then for models.
    model_to_tasks = utils.map_name_tasks("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_tasks.keys():
            self.assertTrue(task in model_to_tasks[model])

  def test_map_name_sources(self):
    """Ensure map from model/dataset name to available sources is correct."""
    # First for datasets.
    ds_to_sources = utils.map_name_sources("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sources.keys():
            self.assertTrue(source in ds_to_sources[ds])

    # Then for models.
    model_to_sources = utils.map_name_sources("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sources.keys():
            self.assertTrue(source in model_to_sources[model])

  def test_map_name_info(self):
    """Ensure tasks and sources are adequately parsed."""
    # First for datasets.
    ds_to_info = utils.map_name_info("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_info.keys():
            self.assertTrue(task in ds_to_info[ds]["tasks"])
            self.assertTrue(source in ds_to_info[ds]["sources"])

    # Then for models.
    models_to_info = utils.map_name_info("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in models_to_info.keys():
            self.assertTrue(task in models_to_info[model]["tasks"])
            self.assertTrue(source in models_to_info[model]["sources"])


if __name__ == "__main__":
  unittest.main()
