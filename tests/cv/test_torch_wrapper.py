# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''fastai https://pytorch.org/ wrapper module'''

import unittest
from sotaai.cv import torch_wrapper, load_dataset, load_model, model_to_dataset, utils, keras_wrapper, tensorflow_wrapper
from sotaai.cv.abstractions import CvDataset, CvModel
import inspect
import torch
import torch.nn as nn
import logging
logging.getLogger('lightning').setLevel(0)


class TestTorchWrapper(unittest.TestCase):
  '''Test the wrapped torch module.'''

  # test_datasets = [
  #     'QMNIST', 'SEMEION', 'Flickr30k', 'VOCSegmentation/2007', 'SBU'
  # ]

  test_datasets = ['QMNIST', 'SEMEION', 'SVHN']
  test_models = [
      'alexnet', 'densenet161', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
      'mnasnet1_3', 'mobilenet_v2', 'resnet18', 'resnet34', 'resnext101_32x8d',
      'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
      'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0',
      'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16_bn',
      'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2'
  ]

  test_datasets_tensorflow = [
      'beans',
      'binary_alpha_digits',
      'caltech_birds2010',
      'caltech_birds2011',
      'cars196',
      'cats_vs_dogs',
      # 'celeb_a',
      'cifar10_1',
      'cifar10_corrupted',
      'cmaterdb',
      'colorectal_histology',
      'colorectal_histology_large',
      'cycle_gan',
      'diabetic_retinopathy_detection',
      'downsampled_imagenet',
      'dtd',
      'emnist',
      'eurosat',
      'food101',
  ]

  #'googlenet' is not working

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
      dso = load_dataset(dataset,
                         torch_ann_file='~/.torch/annotation_file.json')
      for split_name in dso:
        cv_dataset = dso[split_name]
        self.assertEqual(CvDataset, type(cv_dataset))
        iterable_dataset = iter(cv_dataset)
        datapoint = next(iterable_dataset)
        print(datapoint)

  # @unittest.SkipTest
  def test_model_to_dataset_classification(self):

    def single_test(model_name, dataset_name):
      '''This is an inner function that test model_to_dataset for a single case
      i.e. a single model against a single dataset
      '''

      print('\n---')

      cv_model = load_model(model_name, source='torch')

      dataset_splits = load_dataset(dataset_name)
      split_name = next(iter(dataset_splits.keys()))
      cv_dataset = dataset_splits[split_name]

      cv_model, cv_dataset = model_to_dataset(cv_model, cv_dataset)

      self.assertEqual(
          utils.compare_shapes(cv_model.original_input_shape, cv_dataset.shape),
          True)
      ds_classes = cv_dataset.classes_shape[0]
      model_classes = cv_model.original_output_shape[0]
      self.assertEqual(model_classes, ds_classes)

      n = 3
      for i, item in enumerate(cv_dataset):
        if i == n:
          break

        image = item['image']
        print(image.shape)
        output = cv_model(image)
        output_shape = output.shape[1]
        self.assertEqual(output_shape, ds_classes)

    #test models torch with datasets tensorflow
    for model in self.test_models:
      for dataset in self.test_datasets_tensorflow:
        single_test(model, dataset)

    # #test models torch with datasets torch
    for model in self.test_models:
      for dataset in self.test_datasets:
        single_test(model, dataset)

    # #test models torch with dataset keras
    for model in self.test_models:
      for dataset in keras_wrapper.DATASETS['classification']:
        single_test(model, dataset)

  @unittest.SkipTest
  def test_model_to_dataset_segmentation(self):

    # cv_model = load_model('deeplabv3_resnet101', source='torch')
    # cv_dataset = load_dataset('VOCSegmentation/2007')

    def single_test(model_name, dataset_name):
      '''This is an inner function that test model_to_dataset for a single case
      i.e. a single model against a single dataset
      '''

      print('\n---')

      cv_model = load_model(model_name, source='torch')

      dataset_splits = load_dataset(dataset_name)
      split_name = next(iter(dataset_splits.keys()))
      cv_dataset = dataset_splits[split_name]

      cv_model, cv_dataset = model_to_dataset(cv_model, cv_dataset)

      n = 3
      images = []
      for i, item in enumerate(cv_dataset):
        images.append(item['image'])

        if i == n - 1:
          break

      batch = torch.stack(images, dim=0)
      self.assertEqual(tuple(batch.shape), (n, 3, 224, 224))
      self.assertEqual(cv_model.original_output_shape,
                       (len(cv_dataset.pixel_classes), 224, 224))

      output = cv_model(batch)['out']

      print('\nTesting predictions...')

      for i, prediction in enumerate(output):

        mask = torch.argmax(prediction.squeeze(), dim=0).detach().numpy()

        self.assertEqual(tuple(prediction.shape),
                         (len(cv_dataset.pixel_classes), 224, 224))
        self.assertEqual(mask.shape, (224, 224))

    for model in torch_wrapper.MODELS['segmentation']:
      for dataset in torch_wrapper.DATASETS['segmentation']:
        single_test(model, dataset)

    for model in torch_wrapper.MODELS['segmentation']:
      for dataset in tensorflow_wrapper.DATASETS['segmentation']:
        single_test(model, dataset)

  @unittest.SkipTest
  def test_model_to_dataset_object_detection(self):

    def single_test(model_name, dataset_name):
      '''This is an inner function that test model_to_dataset for a single case
      i.e. a single model against a single dataset
      '''

      print('\n---')

      cv_model = load_model(model_name, source='torch')

      dataset_splits = load_dataset(dataset_name)
      split_name = next(iter(dataset_splits.keys()))
      cv_dataset = dataset_splits[split_name]

      cv_model, cv_dataset = model_to_dataset(cv_model, cv_dataset)

      device = torch.device(
          'cuda') if torch.cuda.is_available() else torch.device('cpu')

      image, target = next(iter(cv_dataset))
      targets = [target]
      images = [image]

      images = list(img for img in images)
      tar = [{k: v.to(device) for k, v in t.items()} for t in targets]

      predctions = cv_model.raw(images, tar)
      print(predctions)

    single_test('fasterrcnn_resnet50_fpn', 'VOCDetection/2008')


if __name__ == '__main__':
  unittest.main()
