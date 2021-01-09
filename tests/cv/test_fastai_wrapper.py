# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Unit testing the fastai wrapper."""

import unittest
from sotaai.cv import fastai_wrapper
import inspect
import torch.nn as nn


class TestFastaiWrapper(unittest.TestCase):
  """Test the wrapped fastai module."""

  no_check_ds = [
      "CALTECH_101",  # Not a valid directory.
      "DOGS",  # Not a valid directory
      "MNIST_SAMPLE",  # Not a valid directory.
  ]

  # def test_load_dataset_return_type(self):
  #   """Make sure fastai objects are returned."""
  #   for task in fastai_wrapper.DATASETS:
  #     print("Checking task: {}".format(task))
  #     for ds in fastai_wrapper.DATASETS[task]:
  #       if ds in self.no_check_ds:
  #         continue
  #       print("Testing {}".format(ds))
  #       dso = fastai_wrapper.load_dataset(ds)
  #       # Make sure we are getting a dictionary.
  #       self.assertEqual(dict, type(dso))
  #       # Make sure the objects in the dict are fastai.
  #       for split in dso:
  #         print("Split name: {}".format(split))
  #         print(type(dso[split]))
  #         self.assertTrue("fastai" in str(type(dso[split])))

  def test_load_model(self):
    """Make sure that we can load every model from the fastai module."""

    for task in fastai_wrapper.MODELS:
      for model_name in fastai_wrapper.MODELS[task]:

        model = fastai_wrapper.load_model(model_name)

        #
        # @author HO
        # Test the returned model against the final parent nn.Module class
        # as documented in
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=nn%20module#torch.nn.Module
        #
        self.assertIsInstance(model, nn.Module)

        self.assertEqual(inspect.ismethod(model.forward), True)
        self.assertEqual(inspect.ismethod(model.eval), True)
        self.assertEqual(inspect.ismethod(model.load_state_dict), True)
        self.assertEqual(inspect.ismethod(model.parameters), True)
        self.assertEqual(inspect.ismethod(model.apply), True)
        self.assertEqual(inspect.ismethod(model.zero_grad), True)


if __name__ == "__main__":
  unittest.main()
