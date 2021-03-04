# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''fastai https://pytorch.org/ wrapper module'''

import unittest
from sotaai.cv import torch_wrapper, load_dataset, load_model, model_to_dataset
from sotaai.cv.abstractions import CvDataset, CvModel
# from sotaai.cv.model_to_dataset.torch_to_torch import qmnist_to_alexnet
import inspect
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import logging
logging.getLogger('lightning').setLevel(0)


class TestTorchWrapper(unittest.TestCase):
  '''Test the wrapped torch module.'''

  # test_datasets = [
  #     'QMNIST', 'SEMEION', 'Flickr30k', 'VOCSegmentation/2007', 'SBU'
  # ]

  test_datasets = ['QMNIST', 'SEMEION', 'SVHN', 'USPS']
  test_models = [
      'alexnet', 'densenet161', 'googlenet', 'mnasnet0_5', 'mnasnet0_75',
      'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet18', 'resnet34',
      'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5',
      'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
      'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
      'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2'
  ]

  # @author HO (legacy comment from sotaai-dev)
  # Some datasets need to be downloaded to disk beforehand:
  # - VOC datasets: wrong checksum, LSUN, ImageNet, CocoDetection,
  #   CocoCaptions, Flickr30k, Flickr8k, HMDB51, Kinetics400, UCF101,
  #   VOCDetection/2009, VOCSegmentation/2009,
  #   Cityscapes, SBU

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

  @unittest.SkipTest
  def test_abstract_model(self):
    '''
      Make sure we can create an abstract model using torch models.
    '''

    for task in torch_wrapper.MODELS:
      for model_name in torch_wrapper.MODELS[task]:

        cv_model = load_model(model_name, 'torch')

        self.assertEqual(CvModel, type(cv_model))
        self.assertEqual(cv_model.source, 'torchvision')
        self.assertEqual(cv_model.original_input_type, 'torch.Tensor')

  @unittest.SkipTest
  def test_abstract_dataset(self):
    '''
      Make sure we can create an abstract dataset using torch datasets.
    '''

    # transform = transforms.Compose([transforms.ToTensor()])

    for dataset in self.test_datasets:
      print(dataset)
      dso = load_dataset(dataset, ann_file='~/.torch/annotation_file.json')
      for split_name in dso:
        cv_dataset = dso[split_name]
        self.assertEqual(CvDataset, type(cv_dataset))
        iterable_dataset = iter(cv_dataset)
        datapoint = next(iterable_dataset)
        print(datapoint)

  @unittest.SkipTest
  def test_model_call(self):

    #torch model
    cv_model = load_model('alexnet', source='torch', pretrained=True)

    #Updating the second classifier
    cv_model.raw.classifier[4] = nn.Linear(4096, 1024)

    # Updating the third and the last classifier
    # that is the output layer of the network.
    # Make sure to have 10 output nodes if we
    # are going to get 10 class labels through our model.
    cv_model.raw.classifier[6] = nn.Linear(1024, 10)

    print(cv_model.raw.eval())

    #keras dataset
    cv_dataset = load_dataset('cifar10')
    split_test = cv_dataset['test']

    iterable_dataset = iter(split_test)

    datapoint = next(iterable_dataset)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.fromarray(np.uint8(datapoint['image'])).convert('RGB')

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      cv_model.to('cuda')

    with torch.no_grad():
      output = cv_model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    print(probabilities)

  def test_model_to_dataset(self):

    # cv_model = load_model('alexnet', source='torch')

    for dataset in self.test_datasets:
      cv_dataset = load_dataset(dataset)

      for model in self.test_models:
        cv_model = load_model(model, source='torch')
        if 'test' in cv_dataset:
          model_to_dataset(cv_dataset['test'], cv_model)

    # standarized_dataset, standarized_model = qmnist_to_alexnet(
    #     cv_dataset['test'], cv_model)

    # datapoint = next(standarized_dataset)
    # output = standarized_model(datapoint['image'])

    # print(output)


if __name__ == '__main__':
  unittest.main()
