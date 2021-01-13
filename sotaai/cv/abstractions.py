# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
"""Abstract classes for standardized models and datasets."""


class AbstractCvDataset(object):
  """Our attempt at a standardized, task-agnostic dataset wrapper.

    TODO(tonioteran) Each abstract dataset represents a specific split of a
    full dataset???
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


class AbstractCvModel(object):
  """Our attempt at a standardized, model wrapper.

    Each abstract model represents a model from one of the sources.
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
