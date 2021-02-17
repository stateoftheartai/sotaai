# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''Abstract classes for standardized models and datasets.'''
from sotaai.cv import utils
import numpy as np
import time
import tensorflow_datasets as tfds


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
    # TODO Fix shape, utils function is not working since the first parameter
    # passed is the raw dataset instead of the CvDataset that the function
    # expects (which is still being created)
    # self.shape = utils.get_shape_from_dataset(raw_dataset, name, split_name)

    # Populated for datasets supporting classification or detection tasks.
    self.classes = None
    self.classes_names = None
    if 'classification' in self.tasks or 'object detection' in self.tasks:
      self.classes, self.classes_names = self._get_classes_from_dataset(
          raw_dataset)

    # Only populated for datasets that support segmentation tasks.
    self.pixel_types = None
    self.pixel_types_names = None
    if 'segmentation' in self.tasks:
      self.pixel_types, self.pixel_types_names = (
          self._extract_pixel_types(raw_dataset))

    # Only populated for datasets that support image captioning tasks.
    self.captions = None

    # For visual question answering tasks.A
    self.annotations = None
    self.vocab = None

  def __iter__(self):
    '''Returns the CvDataset iterator object'''
    return self.iterator

  def _get_classes_from_dataset(self, raw_object):
    '''Get the IDs and the names (if available) of the classes.

    Args:
        raw_object:
          Dataset object directly instantiated from a source library. Type
          is dependent on the source library.

    Returns:
        A pair of values, `classes` and `classes_names`. If no
        `classes_names` are available, the pair becomes `classes` and
        `None`.
    '''
    classes = None
    classes_names = None
    if self.source == 'mxnet':
      classes = set(raw_object[:][1])
      classes_names = None
    elif self.source == 'keras':
      classes = np.unique(raw_object[1])
      classes_names = None
    elif self.source == 'torchvision':
      if 'VOC' in self.name:
        # If dataset is an Object Detection Dataset
        classes_names = [
            'unlabeled/void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
            'tv/monitor'
        ]
        classes = list(range(21))
      elif 'class_to_idx' in dir(raw_object):
        classes = list(raw_object.class_to_idx.values())
        classes_names = list(raw_object.class_to_idx.keys())
      elif 'dataset' in dir(raw_object):
        if 'class_to_idx' in dir(raw_object):
          classes = list(raw_object.class_to_idx.values())
          classes_names = list(raw_object.class_to_idx.keys())
        else:
          classes = list(set(raw_object.targets))
          classes_names = None
      elif 'labels' in dir(raw_object):
        classes = list(set(raw_object.labels.numpy()))
        classes_names = None
      else:
        classes = []
        finished = True
        # Set limited time to go through dataset and obtain classes
        time_end = time.time() + 20
        for i in range(self.size):
          # Append the label of each example
          classes.append(self.__getitem__(i)['label'])
          if time.time() > time_end:
            # Execute stopping condition
            finished = False
            break
        if finished:
          classes = list(set(classes))
        else:
          classes = None
        classes_names = None

    elif self.source == 'tensorflow':
      _, ds_info = tfds.load(self.name, with_info=True)
      n_classes = ds_info.features['label'].num_classes
      classes = range(n_classes)
      classes_names = None

    elif self.source == 'fastai':
      obj = getattr(raw_object, self.split_name + '_ds')
      classes_names = obj.y.classes
      classes = range(len(classes_names))

    if classes is not None:
      if list(range(len(classes))) == list(classes):
        # Avoid representing classes with a long list by using range
        classes = range(len(classes))

    return classes, classes_names

  def _extract_pixel_types(self, raw_object):
    '''Get the IDs and the names (if available) of the pixel types.

    Args:
      raw_object:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.

    Returns:
      A pair of values, `pixel_types` and `pixel_types_names`. If no
      `pixel_types_names` are available, the pair becomes `pixel_types`
      and `None`.
    '''
    if 'VOC' in self.name or 'SBD' in self.name:
      classes = [
          'unlabeled/void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
          'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
          'tv/monitor'
      ]
      indexes = list(range(21))
    elif self.source == 'fastai':
      obj = getattr(raw_object, self.split_name + '_ds')
      classes = obj.y.classes
      indexes = None
    else:
      indexes, classes = None, None
    return indexes, classes


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
