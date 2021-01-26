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


if __name__ == "__main__":
  unittest.main()
