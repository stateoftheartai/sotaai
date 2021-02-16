# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the Tensorflow wrapper.'''
import unittest
from sotaai.cv import tensorflow_wrapper
from tensorflow_datasets.core.dataset_utils import _IterableDataset


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

  # @unittest.SkipTest
  def test_load_dataset(self):
    '''Make sure `dict`s are returned, with correct keywords for splits.
    '''
    # for task in tensorflow_wrapper.DATASETS:
    # datasets = tensorflow_wrapper.DATASETS[task]
    # for dataset_name in datasets:
    dataset_name = 'beans'
    dataset = tensorflow_wrapper.load_dataset(dataset_name)

    self.assertEqual(type(dataset), dict)

    for split in dataset:
      self.assertEqual(_IterableDataset, type(dataset[split]))
      # self.assertEqual(len(dataset[split]), 2)

      # self.assertEqual(np.ndarray, type(dataset[split][0]))
      # self.assertEqual(np.ndarray, type(dataset[split][1]))


if __name__ == '__main__':
  unittest.main()
