# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Unit testing the Keras wrapper."""

import os
import unittest
import numpy as np
import inspect
from tensorflow.python.keras.engine.functional import Functional  # pylint: disable=no-name-in-module
from sotaai.cv import load_dataset, keras_wrapper

#
# @author HO
# Just to prevent Keras library to print warnings and extra logging data...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestKerasWrapper(unittest.TestCase):
  """Test the wrapped Keras module."""

  def test_load_dataset(self):
    """
      Make sure `dict`s are returned, with correct keywords for splits.
    """
    for dataset_name in keras_wrapper.UNITTEST_DATASETS:

      dataset = keras_wrapper.load_dataset(dataset_name)

      self.assertEqual(type(dataset), dict)

      for key in dataset:
        self.assertEqual(tuple, type(dataset[key]))
        self.assertEqual(len(dataset[key]), 2)

        self.assertEqual(np.ndarray, type(dataset[key][0]))
        self.assertEqual(np.ndarray, type(dataset[key][1]))

  def test_load_model(self):
    """Make sure that we can load every model from the Keras module."""

    for task in keras_wrapper.MODELS:
      for model_name in keras_wrapper.MODELS[task]:

        model = keras_wrapper.load_model(model_name)

        #
        # @author HO
        # Test the returned model against tf.Keras.Model functional as
        # documented in
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#top_of_page
        #
        self.assertIsInstance(model, Functional)

        self.assertEqual(inspect.ismethod(model.compile), True)
        self.assertEqual(inspect.ismethod(model.fit), True)
        self.assertEqual(inspect.ismethod(model.predict), True)
        self.assertEqual(inspect.ismethod(model.summary), True)
        self.assertEqual(inspect.ismethod(model.save), True)

  def test_abstract_dataset(self):
    """
      Make sure we can create an abstract dataset using
      Keras datasets.
    """
    for ds in keras_wrapper.UNITTEST_DATASETS:
      dso = load_dataset(ds)

      for split_name in dso:
        cv_dataset = dso[split_name]
        datapoint = cv_dataset[0]

        self.assertEqual(np.ndarray, type(datapoint['image']))
        self.assertEqual('label' in datapoint, True)


if __name__ == '__main__':
  unittest.main()
