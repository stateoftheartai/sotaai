# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Unit testing the fastai wrapper."""

# import unittest
from sotaai.cv import fastai_wrapper
# import inspect
# import torch.nn as nn

# class TestFastaiWrapper(unittest.TestCase):
#   """Test the wrapped fastai module."""

#   #
#   # @author Hugo Ochoa
#   # Function temporary commented to avoid testexecution as Github Action
#   # Since these tests require dataset to be downloaded
#   # @todo check how to better do this in the CI server
#   #
#   # def test_load_dataset(self):
#   #   """
#   #     Make sure `dict`s are returned, with correct keywords for splits.
#   #   """
#   #   for task in fastai_wrapper.DATASETS:
#   #     for dataset_name in fastai_wrapper.DATASETS[task]:

#   #       dataset = fastai_wrapper.load_dataset(dataset_name)

#   #       self.assertEqual(type(dataset), dict)

#   #       # for key in dataset:
#   #       #
#   #       # @author HO
#   #       # @todo Validate dataset dict key/value in here...

#   def test_load_model(self):
#     """Make sure that we can load every model from the fastai module."""

#     for task in fastai_wrapper.MODELS:
#       for model_name in fastai_wrapper.MODELS[task]:

#         model = fastai_wrapper.load_model(model_name)

#         #
#         # @author HO
#         # Test the returned model against the final parent nn.Module class
#         # as documented in
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=nn%20module#torch.nn.Module
#         #
#         # Since most data comes from Torch, this is very similar to
#         # torch_wrapper.py
#         #
#         self.assertIsInstance(model, nn.Module)

#         self.assertEqual(inspect.ismethod(model.forward), True)
#         self.assertEqual(inspect.ismethod(model.eval), True)
#         self.assertEqual(inspect.ismethod(model.load_state_dict), True)
#         self.assertEqual(inspect.ismethod(model.parameters), True)
#         self.assertEqual(inspect.ismethod(model.apply), True)
#         self.assertEqual(inspect.ismethod(model.zero_grad), True)

# if __name__ == "__main__":
#   unittest.main()

for task in fastai_wrapper.DATASETS:
  for dataset_name in fastai_wrapper.DATASETS[task]:

    dataset = fastai_wrapper.load_dataset(dataset_name)
