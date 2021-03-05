# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the Keras wrapper.'''

import os
import unittest
import numpy as np
import inspect
from tensorflow.python.keras.engine.functional import Functional
from sotaai.cv import load_dataset, load_model, keras_wrapper, utils, model_to_dataset
from sotaai.cv.abstractions import CvDataset, CvModel
from sotaai.cv import metadata

#
# @author HO
# Just to prevent Keras library to print warnings and extra logging data...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestKerasWrapper(unittest.TestCase):
  '''Test the wrapped Keras module.

  For Keras, we test against all datasets and modules since they are a few and
  can fit in memory (CI server)
  '''

  # @unittest.SkipTest
  def test_load_dataset(self):
    '''Make sure `dict`s are returned, with correct keywords for splits.
    '''
    for task in keras_wrapper.DATASETS:
      datasets = keras_wrapper.DATASETS[task]
      for dataset_name in datasets:
        dataset = keras_wrapper.load_dataset(dataset_name)

        self.assertEqual(type(dataset), dict)

        for split in dataset:
          self.assertEqual(tuple, type(dataset[split]))
          self.assertEqual(len(dataset[split]), 2)

          self.assertEqual(np.ndarray, type(dataset[split][0]))
          self.assertEqual(np.ndarray, type(dataset[split][1]))

  # @unittest.SkipTest
  def test_load_model(self):
    '''Make sure that we can load every model from the Keras module.'''

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

  # @unittest.SkipTest
  def test_abstract_dataset(self):
    '''Make sure we can create an abstract dataset using
      Keras datasets.
    '''

    for task in keras_wrapper.DATASETS:
      datasets = keras_wrapper.DATASETS[task]

      for dataset_name in datasets:
        dso = load_dataset(dataset_name)

        for split_name in dso:

          cv_dataset = dso[split_name]

          self.assertEqual(CvDataset, type(cv_dataset))
          self.assertEqual(cv_dataset.source, 'keras')

          iterable_dataset = iter(cv_dataset)

          datapoint = next(iterable_dataset)
          dataset_metadata = metadata.get('datasets', name=dataset_name)

          self.assertEqual(np.ndarray, type(datapoint['image']))
          self.assertEqual('label' in datapoint, True)

          self.assertEqual(
              utils.compare_shapes(dataset_metadata['metadata']['image'],
                                   datapoint['image'].shape), True)
          self.assertEqual(
              utils.compare_shapes(dataset_metadata['metadata']['label'],
                                   datapoint['label'].shape), True)

  # @unittest.SkipTest
  def test_abstract_model(self):
    '''Make sure we can create an abstract model using
      Keras datasets.
    '''

    for task in keras_wrapper.MODELS:
      for model_name in keras_wrapper.MODELS[task]:

        cv_model = load_model(model_name, 'keras')

        self.assertEqual(CvModel, type(cv_model))
        self.assertEqual(cv_model.source, 'keras')
        self.assertEqual(cv_model.original_input_type, 'numpy.ndarray')

  def test_model_to_dataset(self):
    '''Make sure model_to_dataset is working properly for those models whose
    source is Keras.
    '''

    def single_test(model_name, dataset_name, split_name):
      '''This is an inner function that test model_to_dataset for a single case
      i.e. a single model against a single dataset
      '''

      print('\n--- \nModel: {}\nDataset: {}'.format(model_name, dataset_name))

      cv_model = load_model(model_name, 'keras')

      dataset_splits = load_dataset(dataset_name, 'keras')
      cv_dataset = dataset_splits[split_name]

      cv_model, cv_dataset = model_to_dataset(cv_model, cv_dataset)

      # Assert model channels (for all Keras models are to be 3), assert model
      # input shape and dataset shape are now compatible, and assert model
      # output shape is now compatible with the dataset classes

      self.assertEqual(cv_dataset.shape[-1], 3)
      self.assertEqual(
          utils.compare_shapes(cv_model.original_input_shape, cv_dataset.shape),
          True)
      self.assertEqual(cv_model.original_output_shape, cv_dataset.classes_shape)

      # For some image samples, assert dataset sample shapes matched the
      # cv_dataset.shape, and then assert predictions shape (model output)
      # matches the expected classes

      n = 3
      for i, item in enumerate(cv_dataset):

        self.assertEqual(
            utils.compare_shapes(cv_dataset.shape, item['image'].shape), True,
            'Dataset shape {} is not equal to item shape {}'.format(
                cv_dataset.shape, item['image'].shape))

        image_sample = item['image']
        image_sample = image_sample.reshape((1,) + image_sample.shape)

        predictions = cv_model(image_sample)

        predictions_shape = (1,) + cv_dataset.classes_shape
        self.assertEqual(
            utils.compare_shapes(predictions_shape, predictions.shape), True,
            'Prediction shape {} is not equal to prediction shape {}'.format(
                predictions_shape, predictions.shape))

        if i == 0:
          print(' => Prediction matched dimension: {}'.format(
              predictions.shape))

        if i == n:
          break

    for task in keras_wrapper.MODELS:
      for model_name in keras_wrapper.MODELS[task]:
        for dataset_name in keras_wrapper.DATASETS[task]:
          single_test(model_name, dataset_name, 'test')

    # single_test('ResNet101V2', 'mnist', 'test')


if __name__ == '__main__':
  unittest.main()
