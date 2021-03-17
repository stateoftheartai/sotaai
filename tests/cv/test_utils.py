# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the utility functions.'''
import unittest
import importlib

import sotaai.cv.keras_wrapper as keras
from sotaai.cv import utils
from sotaai.cv import load_model
from sotaai.cv import load_dataset
from sotaai.cv import metadata


class TestCvUtils(unittest.TestCase):
  '''The the utils sub-module for CV.'''

  # @unittest.SkipTest
  def test_map_name_source_tasks(self):
    '''Test for both datasets and models.'''
    # Try first for datasets.
    ds_to_sourcetasks = utils.map_name_source_tasks('datasets')

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sourcetasks.keys():
            self.assertTrue(source in ds_to_sourcetasks[ds].keys())

    # Now for models.
    model_to_sourcetasks = utils.map_name_source_tasks('models')

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sourcetasks.keys():
            self.assertTrue(source in model_to_sourcetasks[model].keys())

  # @unittest.SkipTest
  def test_map_name_tasks(self):
    '''Make sure the map from model/dataset to tasks encapsulates info.'''
    # First for datasets.
    ds_to_tasks = utils.map_name_tasks('datasets')

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_tasks.keys():
            self.assertTrue(task in ds_to_tasks[ds])

    # Then for models.
    model_to_tasks = utils.map_name_tasks('models')

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_tasks.keys():
            self.assertTrue(task in model_to_tasks[model])

  # @unittest.SkipTest
  def test_map_name_sources(self):
    '''Ensure map from model/dataset name to available sources is correct.'''
    # First for datasets.
    ds_to_sources = utils.map_name_sources('datasets')

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sources.keys():
            self.assertTrue(source in ds_to_sources[ds])

    # Then for models.
    model_to_sources = utils.map_name_sources('models')

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sources.keys():
            self.assertTrue(source in model_to_sources[model])

  # @unittest.SkipTest
  def test_map_name_info(self):
    '''Ensure tasks and sources are adequately parsed.'''
    # First for datasets.
    ds_to_info = utils.map_name_info('datasets')

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_info.keys():
            self.assertTrue(task in ds_to_info[ds]['tasks'])
            self.assertTrue(source in ds_to_info[ds]['sources'])

    # Then for models.
    models_to_info = utils.map_name_info('models')

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in models_to_info.keys():
            self.assertTrue(task in models_to_info[model]['tasks'])
            self.assertTrue(source in models_to_info[model]['sources'])

  # @unittest.SkipTest
  def test_get_source_from_model(self):
    '''Ensure the source library is correctly determined from a model object.'''

    # Keras
    m = load_model('InceptionResNetV2', source='keras')
    self.assertEqual(utils.get_source_from_model(m.raw), 'keras')
    m = load_model('NASNetMobile', source='keras')
    self.assertEqual(utils.get_source_from_model(m.raw), 'keras')

  # @unittest.SkipTest
  def test_get_input_type(self):
    '''Ensure the correct input type is being parsed from the model object.'''

    # Keras
    model_metadatas = metadata.get('models', source='keras')

    for model in model_metadatas:
      m = keras.load_model(model['name'])
      self.assertEqual(utils.get_input_type(m), model['metadata']['input_type'])

  # @unittest.SkipTest
  def test_get_num_channels_from_model(self):
    '''Make sure we correctly determine whether a model is color or
    grayscale.'''
    # Keras
    for task in keras.MODELS:
      for model in keras.MODELS[task]:
        m = keras.load_model(model)
        self.assertEqual(utils.get_num_channels_from_model(m), 3)

  # @unittest.SkipTest
  def test_get_num_layers_from_model(self):
    '''Make sure we correctly determine number of layers in model's network.'''

    # Keras models
    model_metadatas = metadata.get('models', source='keras')

    for model in model_metadatas:
      m = keras.load_model(model['name'])
      self.assertEqual(utils.get_num_layers_from_model(m),
                       model['metadata']['num_layers'])

  # @unittest.SkipTest
  def test_get_num_parameters_from_model(self):
    '''Make sure we correctly determine number of parameters in the model.'''

    # Keras models
    model_metadatas = metadata.get('models', source='keras')

    for model in model_metadatas:
      m = keras.load_model(model['name'])
      self.assertEqual(utils.get_num_parameters_from_model(m),
                       model['metadata']['num_parameters'])

  # @unittest.SkipTest
  def test_get_source_from_dataset(self):
    '''Make sure we correctly determine the source from a dataset object.'''
    # d = load_dataset('mnist')
    # self.assertEqual(utils.get_source_from_dataset(d), 'tensorflow')  # Fix.

    #keras
    for task in keras.DATASETS:
      for ds in keras.DATASETS[task]:
        d = keras.load_dataset(ds)
        self.assertEqual(utils.get_source_from_dataset(d['test']), 'keras')

  # @unittest.SkipTest
  def test_get_size_from_dataset(self):
    '''Make sure we correctly determine the size of a dataset's split.'''

    # keras
    dataset_metadatas = metadata.get('datasets', source='keras')

    for dataset_metadata in dataset_metadatas:
      dataset = keras.load_dataset(dataset_metadata['name'])

      self.assertEqual(utils.get_size_from_dataset(dataset['train'], 'train'),
                       dataset_metadata['metadata']['train_size'])
      self.assertEqual(utils.get_size_from_dataset(dataset['test'], 'test'),
                       dataset_metadata['metadata']['test_size'])

  # @unittest.SkipTest
  def test_get_shape_from_dataset(self):
    '''Make sure we correctly determine the shape of a dataset's sample.

    TODO(george) finish.
    '''
    # d = load_dataset('mnist')
    # self.assertEqual(
    # utils.get_shape_from_dataset(d['split name'], 'mnist', 'split name'),
    # (1, 2, 3))

  # @unittest.SkipTest
  def test_get_classes_from_dataset(self):
    '''Make sure we correctly determine the classes and class names of datasets
    sample
    '''

    # keras
    dataset_metadatas = metadata.get('datasets', source='keras')

    for dataset_metadata in dataset_metadatas:
      dataset = load_dataset(dataset_metadata['name'], source='keras')

      for split_name in dataset:
        cv_dataset = dataset[split_name]

        self.assertEqual(cv_dataset.classes,
                         dataset_metadata['metadata'][split_name + '_classes'])
        self.assertEqual(cv_dataset.classes_names,
                         dataset_metadata['metadata']['classes_names'])

  # @unittest.SkipTest
  def test_extract_pixel_classes(self):
    '''Make sure we correctly determine the pixels of a dataset's sample.

    TODO(george) finish.
    '''

    # d = keras.load_dataset('mnist')

    # new_cv_dataset = abstractions.CvDataset(d, None, 'mnist', 'train')

  def test_compare_shapes(self):
    '''Make sure compare shapes works properly'''

    self.assertEqual(utils.compare_shapes(1, 1), True)
    self.assertEqual(utils.compare_shapes((1,), (1,)), True)
    self.assertEqual(utils.compare_shapes((28, 28), (28, 28)), True)
    self.assertEqual(utils.compare_shapes((512, 512, 3), (512, 512, 3)), True)
    self.assertEqual(utils.compare_shapes((), ()), True)
    self.assertEqual(utils.compare_shapes((None, None, 3), (100, 200, 3)), True)
    self.assertEqual(utils.compare_shapes((None, None, 3), (200, 400, 3)), True)

    self.assertEqual(utils.compare_shapes((28, 28, 3), (28, 28)), False)
    self.assertEqual(utils.compare_shapes((28, 28, 3), (None, None, 3)), False)


if __name__ == '__main__':
  unittest.main()
