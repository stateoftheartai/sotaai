# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""
fastai https://pytorch.org/ wrapper module
"""

# import unittest
from sotaai.cv import torch_wrapper
# import inspect
# import torch.nn as nn
# from torch.utils.data.dataloader import DataLoader
import logging
logging.getLogger("lightning").setLevel(0)

# class TestTorchWrapper(unittest.TestCase):
#   """Test the wrapped torch module."""

#   # @author Hugo Ochoa
#   # Function temporary commented to avoid testexecution as Github Action
#   # Since these tests require dataset to be downloaded
#   # @todo check how to better do this in the CI server
#   def test_load_dataset(self):
#     """
#       Make sure `dict`s are returned, with correct keywords for splits.
#     """
#     for task in torch_wrapper.DATASETS:
#       for dataset_name in torch_wrapper.DATASETS[task]:

#         dataset = torch_wrapper.load_dataset(dataset_name)

#         self.assertEqual(type(dataset), dict)

#         for key in dataset:
#           self.assertEqual(DataLoader, type(dataset[key]))

#   def test_load_model(self):
#     """
#       Make sure that we can load every model from the Torch module.
#     """

#     for task in torch_wrapper.MODELS:
#       for model_name in torch_wrapper.MODELS[task]:

#         model = torch_wrapper.load_model(model_name)

#         #
#         # @author HO
#         # Test the returned model against the final parent nn.Module class
#         # as documented in
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=nn%20module#torch.nn.Module
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

for task in torch_wrapper.DATASETS:
  for dataset_name in torch_wrapper.DATASETS[task]:

    try:
      dataset = torch_wrapper.load_dataset(dataset_name)
    except Exception as e:  # pylint: disable=W0703
      print(e)
