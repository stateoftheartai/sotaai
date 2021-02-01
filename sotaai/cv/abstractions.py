# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
"""Abstract classes for standardized models and datasets."""
from sotaai.cv import utils


class CvDataset(object):
  """Our attempt at a standardized, task-agnostic dataset wrapper.

  Each `CvDataset` represents a specific split of a full dataset.
  """

  def __init__(self, raw_object, name: str):
    """Constructor using `raw_object` from a source library.

    Args:
      raw_object:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the dataset.
    """
    self.raw = raw_object
    self.name = name


class CvModel(object):
  """Our attempt at a standardized, model wrapper.

  Each abstract `CvModel` represents a model from one of the sources.
  """

  def __init__(self, raw_object, name: str):
    """Constructor using `raw_object` from a source library.

    Args:
      raw_object:
        Model object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the model.
    """
    self.raw = raw_object
    self.name = name
    self.source = utils.extract_source_from_model(raw_object)
    self.input_type = None  # TODO(tonioteran) Implement me.
    self.min_size = None  # TODO(tonioteran) Implement me.
    self.num_channels = None  # TODO(tonioteran) Implement me.
    self.num_layers = None  # TODO(tonioteran) Implement me.
    self.num_params = None  # TODO(tonioteran) Implement me.
    self.associated_datasets = None  # TODO(tonioteran) Implement me.
    self.paper = None  # TODO(tonioteran) Implement me.

  def __call__(self, input_data):
    """TODO(tonioteran) Define `input_data` type for standardization.

    Here, compatibility between the `input_data` type and the model must have
    been already ensured, e.g., that the image/video is of the correct size,
    that we have the correct number of outputs in the last layer to use the
    dataset, etc.
    """
    raise NotImplementedError
