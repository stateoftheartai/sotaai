# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Unit testing the Torch wrapper."""

import unittest
from sotaai.cv import torch_wrapper


class TestTorchWrapper(unittest.TestCase):
  """Test `load_dataset` method."""

  # Some datasets need to be downloaded to disk beforehand, so we"ll ignore.
  #   - VOC datasets: wrong checksum.
  no_check_ds = [
      "LSUN",
      "ImageNet",
      "CocoDetection",
      "CocoCaptions",
      "Flickr30k",
      "Flickr8k",
      "HMDB51",
      "Kinetics400",
      "UCF101",
      "VOCDetection/2009",
      "VOCSegmentation/2009",
      "Cityscapes",
      "SBU",  # Only because it"s gigantic and long to download.
  ]

  def test_load_dataset(self):
    """Make sure `dict`s are returned, with correct keywords for splits."""
    all_keywords = []
    for task in torch_wrapper.DATASETS:
      print("Checking task: {}".format(task))
      for ds in torch_wrapper.DATASETS[task]:
        if ds in self.no_check_ds:
          continue
        print("Testing {}".format(ds))
        dso = torch_wrapper.load_dataset(ds)
        # Make sure we are receiving a dictionary.
        self.assertEqual(type(dso), dict)
        # Make sure the dictionary has the correct keys
        # Make sure the internal objects behind the keys are of
        # torchvision datasets type.
        for key in dso:
          all_keywords.append(key)
          self.assertTrue("torch" in str(type(dso[key])))
          print(type(dso[key]))
          # Make sure we are accounting for this particular key.
          # TODO(tonioteran) Activate.
          # self.assertTrue(key in torch_wrapper.SPLIT_KEYWORDS)
    print(all_keywords)
    print(set(all_keywords))

  def test_load_model(self):
    """Ensures we can load every model from the torch module."""
    for model_name in torch_wrapper.MODELS:
      print("Checking model {}".format(model_name))
      model = torch_wrapper.load_model(model_name)
      print(type(model))
      print("\n")


if __name__ == "__main__":
  unittest.main()
