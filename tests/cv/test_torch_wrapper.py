# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''fastai https://pytorch.org/ wrapper module'''

import unittest
from sotaai.cv import torch_wrapper, load_dataset
from sotaai.cv.abstractions import CvDataset
import inspect
import torch.nn as nn
# import torchvision.transforms as transforms
import logging
logging.getLogger('lightning').setLevel(0)


class TestTorchWrapper(unittest.TestCase):
  '''Test the wrapped torch module.'''

  test_datasets = ['QMNIST', 'SEMEION', 'Flickr30k', 'VOCSegmentation/2007']
  # @author HO (legacy comment from sotaai-dev)
  # Some datasets need to be downloaded to disk beforehand:
  # - VOC datasets: wrong checksum, LSUN, ImageNet, CocoDetection,
  #   CocoCaptions, Flickr30k, Flickr8k, HMDB51, Kinetics400, UCF101,
  #   VOCDetection/2009, VOCSegmentation/2009,
  #   Cityscapes, SBU

  # @author Hugo Ochoa
  # Function temporary commented to avoid test execution as a
  # Github Action. Since these tests require dataset to be downloaded
  # @todo check how to better do this in the CI server
  # def test_load_dataset(self):
  #   '''
  #     Make sure `dict`s are returned, with correct keywords for splits.
  #   '''
  #   for task in torch_wrapper.DATASETS:
  #     for dataset_name in torch_wrapper.DATASETS[task]:

  #       dataset = torch_wrapper.load_dataset(dataset_name)

  #       self.assertEqual(type(dataset), dict)

  #       for key in dataset:
  #         self.assertEqual(DataLoader, type(dataset[key]))
  @unittest.SkipTest
  def test_load_model(self):
    '''Make sure that we can load every model from the Torch module.'''

    for task in torch_wrapper.MODELS:
      for model_name in torch_wrapper.MODELS[task]:

        model = torch_wrapper.load_model(model_name)
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

  @unittest.SkipTest
  def test_load_dataset(self):
    '''
      Make sure `dict`s are returned, with correct keywords for splits.
    '''

    for dataset_name in self.test_datasets:
      print(dataset_name)
      dataset = torch_wrapper.load_dataset(
          dataset_name, ann_file='~/.torch/annotation_file.json')

      self.assertEqual(type(dataset), dict)

  def test_abstract_dataset(self):
    '''
      Make sure we can create an abstract dataset using torch datasets.
    '''

    # transform = transforms.Compose([transforms.ToTensor()])

    for dataset in self.test_datasets:
      dso = load_dataset(dataset, ann_file='~/.torch/annotation_file.json')
      for split_name in dso:
        cv_dataset = dso[split_name]
        self.assertEqual(CvDataset, type(cv_dataset))
        iterable_dataset = iter(cv_dataset)
        datapoint = next(iterable_dataset)
        print(datapoint)


if __name__ == '__main__':
  unittest.main()
