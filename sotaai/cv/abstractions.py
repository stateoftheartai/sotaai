# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''Abstract classes for standardized models and datasets.'''
from sotaai.cv import utils


class CvDataset(object):
  '''Our attempt at a standardized, task-agnostic dataset wrapper.

  Each `CvDataset` represents a specific split of a full dataset.
  '''

  def __init__(self, raw_dataset, iterator, name: str, split_name: str):
    '''Constructor using `raw_dataset` from a source library.

    Args:
      raw_dataset:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.
      name (str):
        Name of the dataset.
      split_name (str):
        Name of the dataset's split.
    '''
    self.raw = raw_dataset
    self.iterator = iterator
    self.name = name
    self.source = utils.get_source_from_dataset(raw_dataset)
    self.data_type = None  # TODO(tonioteran) Implement me.
    self.split_name = split_name
    self.tasks = utils.map_dataset_tasks()[name]
    self.size = utils.get_size_from_dataset(raw_dataset, self.split_name)
    self.shape = utils.get_shape_from_dataset(raw_dataset, name, split_name)

    # Populated for datasets supporting classification or detection tasks.
    self.classes = None
    self.classes_names = None
    self.classes_shape = None

    if 'classification' in self.tasks or 'object_detection' in self.tasks:
      classes, classes_names, classes_shape = utils.get_classes_from_dataset(
          raw_dataset, self.source, self.name, self.split_name)

      self.classes = classes
      self.classes_names = classes_names
      self.classes_shape = classes_shape

    # Only populated for datasets that support segmentation tasks.
    self.pixel_classes = None
    self.pixel_classes_names = None
    if 'segmentation' in self.tasks:
      self.pixel_classes, self.pixel_classes_names = (
          utils.extract_pixel_classes(raw_dataset, self.name, self.source,
                                      self.split_name))

    # Only populated for datasets that support image captioning tasks.
    self.captions = None

    # For visual question answering tasks.A
    self.annotations = None
    self.vocab = None

  def __iter__(self):
    '''Returns the CvDataset iterator object'''
    return self.iterator

  def set_image_preprocessing(self, image_preprocessing_callback):
    self.iterator.set_image_preprocessing(image_preprocessing_callback)


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
    self.tasks = utils.map_name_tasks('models')[name]
    self.source = utils.get_source_from_model(self.raw)
    self._populate_attributes()

  def _populate_attributes(self):
    self.original_input_type = utils.get_input_type(self.raw)
    self.original_input_shape = utils.get_input_shape(self.raw)

    self.original_output_shape = utils.get_output_shape(self.raw)

    # TODO(Hugo) Implement me.
    # Still pending to use input_min_shape in model_to_dataset logic instead of
    # the IMAGE_MINS dict
    self.input_shape_min = utils.get_input_shape_min(self.name)

    self.num_channels = utils.get_num_channels_from_model(self.raw)
    self.num_layers = utils.get_num_layers_from_model(self.raw)
    self.num_params = utils.get_num_parameters_from_model(self.raw)
    self.paper = None  # TODO(tonioteran) Implement me.

  def to_dict(self) -> dict:
    return {
        '_name': self.name,
        '_type': 'model',
        '_source': self.source,
        '_tasks': self.tasks,
        '_paper': self.paper,
        'input_type': self.original_input_type,
        'input_shape_height': self.original_input_shape[0],
        'input_shape_width': self.original_input_shape[1],
        'input_shape_channels': self.original_input_shape[2],
        'input_shape_min_height': self.input_shape_min[0],
        'input_shape_min_width': self.input_shape_min[1],
        'output_shape': self.original_output_shape,
        'num_layers': self.num_layers,
        'num_params': self.num_params,
    }

  def update_raw_model(self, model) -> None:
    '''Update raw model with a new one modified using Keras API directly.

    Args:
      model: the new Keras model that will replace the original raw model
    '''
    self.raw = model
    self._populate_attributes()

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
    if self.source == 'torchvision':
      return self.raw(input_data)
    else:
      raise NotImplementedError('Still not implemented for the current model')
