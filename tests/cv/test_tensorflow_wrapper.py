# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the Tensorflow wrapper.'''
import unittest

from sotaai.cv import load_dataset, tensorflow_wrapper, metadata
from sotaai.cv.abstractions import CvDataset
from tensorflow_datasets.core.dataset_utils import _IterableDataset
import numpy as np


class TestTensorflowWrapper(unittest.TestCase):
  '''Test `load_dataset` method.'''

  # Some datasets seem to be too large for unit testing purposes.
  no_check_ds = [
      'open_images_challenge2019_detection',  # Too large, 565 GB.
      'open_images_v4',  # Too large, 565 GB.
      'curated_breast_imaging_ddsm',  # Manual download.
      'deep_weeds',  # Manual download.
      'diabetic_retinopathy_detection',  # Manual download.
      'imagenet2012',  # Manual download.
      'imagenet2012_corrupted',  # Manual download.
      'imagenet2012_subset',  # Manual download.
      'resisc45',  # Manual download.
      'vgg_face2',  # Manual download.
      'celeb_a_hq',  # Manual download.
      'chexpert',  # Manual download.
      'cityscapes',  # Manual download.
      'quickdraw_bitmap',  # Gigantic dataset.
      'lsun',  # Req tensorflow_io.
      'scene_parse150',  # Timeout error.
      'caltech_birds2011',  # Error, GoogDrive.
      'i_naturalist2017',  # Error, GoogDrive.
      'sun397',  # TODO(tonioteran) Error.
  ]

  test_datasets = ['beans', 'omniglot']

  # @unittest.SkipTest
  def test_load_dataset(self):
    '''Make sure `dict`s are returned, with correct keywords for splits.

    As of now, only testing with a limited set of datasets (test_datasets) to
    save memory while running these tests.
    '''
    for dataset_name in self.test_datasets:
      dataset = tensorflow_wrapper.load_dataset(dataset_name)

      self.assertEqual(type(dataset), dict)

      for split in dataset:
        self.assertEqual(_IterableDataset, type(dataset[split]))

  # @unittest.SkipTest
  def test_abstract_dataset(self):
    '''Make sure we can create an abstract dataset using Tensorflow datasets.
    '''

    for dataset_name in self.test_datasets:
      dso = load_dataset(dataset_name)

      for split_name in dso:
        cv_dataset = dso[split_name]
        self.assertEqual(CvDataset, type(cv_dataset))
        self.assertEqual(cv_dataset.source, 'tensorflow')

        iterable_dataset = iter(cv_dataset)

        datapoint = next(iterable_dataset)
        self.assertEqual(np.ndarray, type(datapoint['image']))
        self.assertEqual('label' in datapoint, True)

        dataset_metadata = metadata.DATASETS[dataset_name]
        self.assertEqual(datapoint['label'].shape, dataset_metadata['label'])
        self.assertEqual(datapoint['image'].shape, dataset_metadata['image'])


if __name__ == '__main__':
  unittest.main()
