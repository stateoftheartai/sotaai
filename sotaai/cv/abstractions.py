# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''Abstract classes for standardized models and datasets.'''
from sotaai.cv import utils
from sotaai.cv.keras_wrapper import get_dataset_item as keras_item


class CvDataset(object):
  '''Our attempt at a standardized, task-agnostic dataset wrapper.

  Each `CvDataset` represents a specific split of a full dataset.
  '''

  def __init__(self, raw_dataset, name: str):
    '''Constructor using `raw_dataset` from a source library.

    Args:
      raw_dataset:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the dataset.
    '''
    self.raw = raw_dataset
    self.name = name
    self.source = utils.get_source_from_dataset(name)
    self.data_type = None  # TODO(tonioteran) Implement me.
    self.split_name = None  # TODO(tonioteran) Implement me.
    self.tasks = None  # TODO(tonioteran) Implement me.
    self.size = None  # TODO(tonioteran) Implement me.
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
    '''Draw the `i`-th item from the dataset.
    Args:
      i (int):
        Index for the item to be gotten.

    Returns: a dict. The dict will contain a 'data' key which will hold the
      datapoint as a numpy array. The dict will also contain a 'label' key which
      will hold the label of the datapoint. The dict might contain other keys
      depending on the nature of the dataset.
    '''
    if self.source == 'keras':
      return keras_item(self.raw, i)
    else:
      raise NotImplementedError('Get item not implemented for the given source')


class CvModel(object):
  '''Our attempt at a standardized, model wrapper.

  Each abstract `CvModel` represents a model from one of the sources.
  '''

  def __init__(self, raw_model, name: str):
    '''Constructor using `raw_model` from a source library.

    Args:
      raw_model:
        Model object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the model.
    '''
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

  def update(self, model) -> None:
    '''Update raw model with a new one modified using Keras API directly.

    Args:
      model: the new Keras model that will replace the original raw model
    '''
    self.raw = model

  def __call__(self, input_data):
    '''Return model predictions for the input_data

    Args:
      input_data: valid input data as required by the model
    Returns:
      As of now, the predicted data as returned by model.raw
    Raises:
      NotImplementedError: an error in case the method is not implemented for
      the current model
    '''
    if self.source == 'keras':
      return self.raw.predict(input_data)
    else:
      raise NotImplementedError('Still not implemented for the current model')
