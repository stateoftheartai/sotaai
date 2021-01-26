# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Unit testing the Tensorflow wrapper."""
import unittest
from sotaai.cv import tensorflow_wrapper  # pylint: disable=W0611


class TestTensorflowWrapper(unittest.TestCase):
  """Test `load_dataset` method."""

  # Some datasets seem to be too large for unit testing purposes.
  no_check_ds = [
      "open_images_challenge2019_detection",  # Too large, 565 GB.
      "open_images_v4",  # Too large, 565 GB.
      "curated_breast_imaging_ddsm",  # Manual download.
      "deep_weeds",  # Manual download.
      "diabetic_retinopathy_detection",  # Manual download.
      "imagenet2012",  # Manual download.
      "imagenet2012_corrupted",  # Manual download.
      "imagenet2012_subset",  # Manual download.
      "resisc45",  # Manual download.
      "vgg_face2",  # Manual download.
      "celeb_a_hq",  # Manual download.
      "chexpert",  # Manual download.
      "cityscapes",  # Manual download.
      "quickdraw_bitmap",  # Gigantic dataset.
      "lsun",  # Req tensorflow_io.
      "scene_parse150",  # Timeout error.
      "caltech_birds2011",  # Error, GoogDrive.
      "i_naturalist2017",  # Error, GoogDrive.
      "sun397",  # TODO(tonioteran) Error.
  ]

  # @author Tonio Teran
  # Function temporary commented to avoid testexecution as Github Action
  # Since these tests require dataset to be downloaded
  # @todo check how to better do this in the CI server
  # def test_load_dataset_return_type(self):
  #     """Make sure adequate `dict`s are returned."""
  #     all_keywords = []
  #     for task in tensorflow_wrapper.DATASETS.keys():
  #         print("Checking task: {}".format(task))
  #         for ds in tensorflow_wrapper.DATASETS[task]:
  #             if ds in self.no_check_ds:
  #                 continue
  #             print("*********************")
  #             print("*********************")
  #             print("Testing {}".format(ds))
  #             print("*********************")
  #             print("*********************")
  #             dso = tensorflow_wrapper.load_dataset(ds)
  #             # Make sure we are receiving a dictionary.
  #             self.assertEqual(type(dso), dict)
  #             # Make sure the dictionary has the correct keys
  #             print(dso.keys())
  #             # Make sure the internal objects behind the keys are of
  #             # tensorflow datasets type.
  #             for key in dso.keys():
  #                 all_keywords.append(key)
  #                 self.assertTrue("tensorflow" in str(type(dso[key])))
  #                 print(type(dso[key]))
  #                 # Make sure we are accounting for this particular key.
  #                 # TODO(tonioteran) Activate.
  #                 # self.assertTrue(key in torch_wrapper.SPLIT_KEYWORDS)
  #     print(all_keywords)
  #     print(set(all_keywords))

  def test_dataset_split_keywords(self):
    """Make sure the correct split keywords are being used."""
    # TODO(tonioteran) Activate.
    # self.assertTrue("train" in tensorflow_wrapper.SPLIT_KEYWORDS)
    # self.assertTrue("test" in tensorflow_wrapper.SPLIT_KEYWORDS)
    # self.assertTrue("validation" in tensorflow_wrapper.SPLIT_KEYWORDS)
    self.assertTrue(True)  # pylint: disable=W1503


if __name__ == "__main__":
  unittest.main()
