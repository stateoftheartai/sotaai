# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''Abstract classes for standardized models and datasets.'''
from sotaai.cv import utils

datasets_tasks_map = utils.map_name_tasks('datasets')
models_tasks_map = utils.map_name_tasks('models')


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
      source (str):
        Source name used when no raw_model is given
    '''

    self.raw = raw_dataset
    self.name = name
    self.tasks = datasets_tasks_map[name]
    self.iterator = iterator
    self.source = utils.get_source_from_dataset(self.raw)

    self.is_implemented = not (isinstance(self.raw, dict) and
                               'source' in self.raw)
    if not self.is_implemented:

      self.split_name = None
      self.size = None
      self.shape = None
      self.classes = None
      self.classes_names = None
      self.classes_shape = None
      self.pixel_classes = None
      self.pixel_classes_names = None
      self.source = None

    else:

      self.split_name = split_name
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

  def to_dict(self) -> dict:
    return {
        'name':
            self.name,
        'type':
            'dataset',
        'source':
            self.source,
        'tasks':
            self.tasks,
        'cv_num_items':
            self.size,
        'cv_item_width':
            self.shape[0] if self.shape else None,
        'cv_item_height':
            self.shape[1] if self.shape else None,
        'cv_item_channels':
            self.shape[2] if self.shape and len(self.shape) == 3 else None,
        'cv_num_classes':
            self.classes_shape[0] if self.classes_shape else None,
        'cv_num_pixel_classes':
            len(self.pixel_classes) if self.pixel_classes else None,
    }

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
      source (str):
        Source name used when no raw_model is given
    '''
    self.raw = raw_model
    self.name = name
    self.tasks = models_tasks_map[name]
    self.source = utils.get_source_from_model(self.raw)
    self.is_implemented = not (isinstance(self.raw, dict) and
                               'source' in self.raw)

    if not self.is_implemented:
      self.original_input_type = None
      self.original_input_shape = None
      self.original_output_shape = None
      self.input_shape_min = None
      self.num_channels = None
      self.num_layers = None
      self.num_params = None
      self.paper = None
    else:
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
        'name':
            self.name,
        'type':
            'model',
        'source':
            self.source,
        'tasks':
            self.tasks,
        'paper':
            self.paper,
        'cv_input_type':
            self.original_input_type,
        'cv_input_shape_height':
            self.original_input_shape[0] if self.original_input_shape else None,
        'cv_input_shape_width':
            self.original_input_shape[1] if self.original_input_shape else None,
        'cv_input_shape_channels':
            self.original_input_shape[2] if self.original_input_shape else None,
        'cv_input_shape_min_height':
            self.input_shape_min[0] if self.input_shape_min else None,
        'cv_input_shape_min_width':
            self.input_shape_min[1] if self.input_shape_min else None,
        'cv_output_shape':
            self.original_output_shape,
        'cv_num_layers':
            self.num_layers,
        'cv_num_params':
            self.num_params,
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
