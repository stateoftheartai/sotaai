# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
"""Abstract classes for standardized models and datasets."""
from sotaai.cv import utils


class CvDataset(object):
  """Our attempt at a standardized, task-agnostic dataset wrapper.

  Each `CvDataset` represents a specific split of a full dataset.
  """

  def __init__(self, raw_dataset, name: str):
    """Constructor using `raw_dataset` from a source library.

    Args:
      raw_dataset:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the dataset.
    """
    self.raw = raw_dataset
    self.name = name
    self.source = utils.get_source_from_dataset(raw_dataset)
    self.data_type = None  # TODO(tonioteran) Implement me.
    self.split_name = None  # TODO(tonioteran) Implement me.
    self.tasks = None  # TODO(tonioteran) Implement me.
    self.size = utils.get_size_from_dataset(raw_dataset, self.split_name)
    self.shape = None  # TODO(tonioteran) Implement me.

    # Populated for datasets supporting classification or detection tasks.
    self.classes = None
    self.classes_names = None

    # Only populated for datasets that support segmentation tasks.
    self.pixel_types = None
    self.pixel_types_names = None

    # Only populated for datasets that support image captioning tasks.
    self.captions = None

    # For visual question answering tasks.A
    self.annotations = None
    self.vocab = None

  def __getitem__(self, i: int):
    """Draw the `i`-th item from the dataset.
    Args:
      i (int):
        Index for the item to be gotten.

    Returns: The i-th sample as a dict. The first element of dict is a
      numpy.ndarray with shape (N, H, W, C), where N = 1 represents the
      number of images, H the image height, W the image width, and C the
      number of channels. The next are labeled that depend on the task of the
      dataset.

    TODO(hugo) This is the original description we had for this method, but
    we can chat more about it so see if it's still appropriate.
    """
    raise NotImplementedError("TODO: sample and translate to numpy.ndarray")


class CvModel(object):
  """Our attempt at a standardized, model wrapper.

  Each abstract `CvModel` represents a model from one of the sources.
  """

  def __init__(self, raw_model, name: str):
    """Constructor using `raw_model` from a source library.

    Args:
      raw_model:
        Model object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the model.
    """
    self.raw = raw_model
    self.name = name
    self.source = utils.get_source_from_model(raw_model)
    self.original_input_type = utils.get_input_type(raw_model)
    self.data_type = None  # TODO(tonioteran) Implement me.
    self.min_size = None  # TODO(tonioteran) Implement me.
    self.num_channels = utils.get_num_channels_from_model(raw_model)
    self.num_layers = utils.get_num_layers_from_model(raw_model)
    self.num_params = utils.get_num_parameters_from_model(raw_model)
    self.associated_datasets = None  # TODO(tonioteran) Implement me.
    self.paper = None  # TODO(tonioteran) Implement me.

  def __call__(self, input_data):
    """TODO(tonioteran) Define `input_data` type for standardization.

    Here, compatibility between the `input_data` type and the model must have
    been already ensured, e.g., that the image/video is of the correct size,
    that we have the correct number of outputs in the last layer to use the
    dataset, etc.
    """
    # This, I believe, should more or less be the workflow:
    #
    # 1. Transform the `input_data` into something that `self.raw` can
    # understand.
    # 2. The `self.raw` object should already be able to process the
    # transformed `input_data`, so just directly feed it as:
    #
    #  return self.raw(transformed_input_data)
    raise NotImplementedError("Still need to implement the evaluation fxn.")
