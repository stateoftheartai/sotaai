# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Unit testing the utility functions."""
import unittest
import importlib
from sotaai.cv import utils
from sotaai.cv import load_model
from sotaai.cv import load_dataset


class TestCvUtils(unittest.TestCase):
  """The the utils sub-module for CV."""

  def test_map_dataset_source_tasks(self):
    """Make sure the dataset map correctly encapsulates all info."""
    ds_to_sourcetasks = utils.map_dataset_source_tasks()

    # Make sure that every dataset is accounted for by iterating over all
    # sources and ensuring they have a corresponding key.
    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sourcetasks.keys():
            self.assertTrue(source in ds_to_sourcetasks[ds].keys())

  def test_map_dataset_tasks(self):
    """Make sure the map from dataset to tasks correctly encapsulates info."""
    ds_to_tasks = utils.map_dataset_tasks()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_tasks.keys():
            self.assertTrue(task in ds_to_tasks[ds])

  def test_map_dataset_sources(self):
    """Ensure map from dataset name to available sources is correct."""
    ds_to_sources = utils.map_dataset_sources()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sources.keys():
            self.assertTrue(source in ds_to_sources[ds])

  def test_map_dataset_info(self):
    """Ensure tasks and sources are adequately parsed."""
    ds_to_info = utils.map_dataset_info()

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_info.keys():
            self.assertTrue(task in ds_to_info[ds]["tasks"])
            self.assertTrue(source in ds_to_info[ds]["sources"])

  def test_map_name_source_tasks(self):
    """Test for both datasets and models."""
    # Try first for datasets.
    ds_to_sourcetasks = utils.map_name_source_tasks("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sourcetasks.keys():
            self.assertTrue(source in ds_to_sourcetasks[ds].keys())

    # Now for models.
    model_to_sourcetasks = utils.map_name_source_tasks("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sourcetasks.keys():
            self.assertTrue(source in model_to_sourcetasks[model].keys())

  def test_map_name_tasks(self):
    """Make sure the map from model/dataset to tasks encapsulates info."""
    # First for datasets.
    ds_to_tasks = utils.map_name_tasks("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_tasks.keys():
            self.assertTrue(task in ds_to_tasks[ds])

    # Then for models.
    model_to_tasks = utils.map_name_tasks("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_tasks.keys():
            self.assertTrue(task in model_to_tasks[model])

  def test_map_name_sources(self):
    """Ensure map from model/dataset name to available sources is correct."""
    # First for datasets.
    ds_to_sources = utils.map_name_sources("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_sources.keys():
            self.assertTrue(source in ds_to_sources[ds])

    # Then for models.
    model_to_sources = utils.map_name_sources("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in model_to_sources.keys():
            self.assertTrue(source in model_to_sources[model])

  def test_map_name_info(self):
    """Ensure tasks and sources are adequately parsed."""
    # First for datasets.
    ds_to_info = utils.map_name_info("datasets")

    for source in utils.DATASET_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.DATASETS:
        for ds in wrapper.DATASETS[task]:
          # Account for any spelling discrepancies.
          if ds in ds_to_info.keys():
            self.assertTrue(task in ds_to_info[ds]["tasks"])
            self.assertTrue(source in ds_to_info[ds]["sources"])

    # Then for models.
    models_to_info = utils.map_name_info("models")

    for source in utils.MODEL_SOURCES:
      wrapper = importlib.import_module("sotaai.cv." + source + "_wrapper")
      for task in wrapper.MODELS:
        for model in wrapper.MODELS[task]:
          # Account for any spelling discrepancies.
          if model in models_to_info.keys():
            self.assertTrue(task in models_to_info[model]["tasks"])
            self.assertTrue(source in models_to_info[model]["sources"])

  def test_get_source_from_model(self):
    """Ensure the source library is correctly determined from a model object.

    TODO(hugo) Fill out the tests.
    """
    # Load a couple of torchvision examples here, e.g.,
    m = load_model("alexnet", source="torch")
    self.assertEqual(utils.get_source_from_model(m), "torch")
    m = load_model("densenet121", source="torch")
    self.assertEqual(utils.get_source_from_model(m), "torch")

    m = load_model("resnet152_v2", source="mxnet")
    self.assertEqual(utils.get_source_from_model(m), "mxnet")
    m = load_model("squeezenet1.1", source="mxnet")
    self.assertEqual(utils.get_source_from_model(m), "mxnet")

    m = load_model("InceptionResNetV2", source="keras")
    self.assertEqual(utils.get_source_from_model(m), "keras")
    m = load_model("NASNetMobile", source="keras")
    self.assertEqual(utils.get_source_from_model(m), "keras")

    m = load_model("resnet101", source="fastai")
    self.assertEqual(utils.get_source_from_model(m), "fastai")

  def test_flatten_model(self):
    """Ensure the returned array corresponds to the actual model layers.

    TODO(hugo) Fill out the tests.
    """
    # Need to figure out what the checks should look like (e.g., just count the
    # number of layers?).
    m = load_model("alexnet", source="torch")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.
    m = load_model("densenet121", source="torch")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.

    m = load_model("resnet152_v2", source="mxnet")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.
    m = load_model("squeezenet1.1", source="mxnet")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.

    m = load_model("InceptionResNetV2", source="keras")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.
    m = load_model("NASNetMobile", source="keras")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.

    m = load_model("resnet101", source="fastai")
    layers = utils.flatten_model(m)
    self.assertEqual(layers, ["I", "dont", "know"])  # TODO(hugo) Fix.

  def test_get_input_type(self):
    """Ensure the correct input type is being parsed from the model object.

    TODO(hugo) Fill out the tests.
    """
    # Load a couple of torchvision examples here, e.g.,
    m = load_model("alexnet", source="torch")
    self.assertEqual(utils.get_input_type(m), "torch.Tensor")  # Don't know...
    m = load_model("densenet121", source="torch")
    self.assertEqual(utils.get_input_type(m), "torch.Tensor")  # Don't know...

    m = load_model("resnet152_v2", source="mxnet")
    self.assertEqual(utils.get_input_type(m), "numpy.ndarray")  # Don't know...
    m = load_model("squeezenet1.1", source="mxnet")
    self.assertEqual(utils.get_input_type(m), "numpy.ndarray")  # Don't know...

    m = load_model("InceptionResNetV2", source="keras")
    self.assertEqual(utils.get_input_type(m), "numpy.ndarray")  # Don't know...
    m = load_model("NASNetMobile", source="keras")
    self.assertEqual(utils.get_input_type(m), "numpy.ndarray")  # Don't know...

    m = load_model("resnet101", source="fastai")
    self.assertEqual(utils.get_input_type(m), "numpy.ndarray")  # Don't know...

  def test_get_num_channels_from_model(self):
    """Make sure we correctly determine whether a model is color or grayscale.

    TODO(hugo) finish.
    """
    # Load a couple of torchvision examples here, e.g.,
    m = load_model("alexnet", source="torch")
    self.assertEqual(utils.get_num_channels_from_model(m), 3)

  def test_get_num_layers_from_model(self):
    """Make sure we correctly determine number of layers in model's network.

    TODO(hugo) finish.
    """
    # Load a couple of torchvision examples here, e.g.,
    m = load_model("alexnet", source="torch")
    self.assertEqual(utils.get_num_layers_from_model(m), 8)

  def test_get_num_parameters_from_model(self):
    """Make sure we correctly determine number of parameters in the model.

    TODO(hugo) finish.
    """
    # Load a couple of torchvision examples here, e.g.,
    m = load_model("alexnet", source="torch")
    self.assertEqual(utils.get_num_parameters_from_model(m), 1000000)  # Fix.

  def test_get_source_from_dataset(self):
    """Make sure we correctly determine the source from a dataset object.

    TODO(george) finish.
    """
    d = load_dataset("mnist")
    self.assertEqual(utils.get_source_from_dataset(d), "tensorflow")  # Fix.

  def test_get_size_from_dataset(self):
    """Make sure we correctly determine the size of a dataset's split.

    TODO(george) finish.
    """
    d = load_dataset("mnist")
    self.assertEqual(utils.get_size_from_dataset(d["split name"], "split name"),
                     30000)

  def test_get_shape_from_dataset(self):
    """Make sure we correctly determine the shape of a dataset's sample.

    TODO(george) finish.
    """
    d = load_dataset("mnist")
    self.assertEqual(
        utils.get_shape_from_dataset(d["split name"], "mnist", "split name"),
        (1, 2, 3))


if __name__ == "__main__":
  unittest.main()
