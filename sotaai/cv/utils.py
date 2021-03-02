# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Useful utility functions to navigate the library's available resources.'''
# TODO(tonioteran) Deprecate specific dataset/model functions for the
# generalized version.
import importlib
import mxnet as mx
import numpy as np
import torch
import tensorflow_datasets as tfds
import time
import os

# Prevent Tensorflow to print warning and meta logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO(tonioteran) Currently removed 'mxnet' and 'pretrainedmodels' from
# MODEL_SOURCES. Need to restore as soon as the wrapper is done and unit test.
MODEL_SOURCES = ['fastai', 'keras', 'torch']  # 'mxnet', 'pretrainedmodels'

# TODO(tonioteran) Currently removed 'mxnet' from DATASET_SOURCES. Need to
# restore as soon as the wrapper is done and unit test.
DATASET_SOURCES = ['tensorflow', 'fastai', 'keras', 'torch']  # 'mxnet'


def map_dataset_source_tasks() -> dict:
  '''Gathers all datasets and their respective sources and available tasks.

  Crawls through all modules to arrange entries of the form:

    <dataset-name>: {
        <name-source-1>: [<supported-task-11>, <supported-task-12>, ...],
        <name-source-2>: [<supported-task-21>, <supported-task-22>, ...],
        ...
        <name-source-n>: [<supported-task-n1>, <supported-task-n2>, ...],
    }

  Ensures duplicate removals by transforming all strings to lower case, and
  preserving the original names in an additional `original_names` dictionary.

  Returns (dict):
    Dictionary with an entry for all available datasets of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  datasets_breakdown = dict()
  original_names = dict()

  for source in DATASET_SOURCES:
    wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
    for task in wrapper.DATASETS:
      for ds in wrapper.DATASETS[task]:
        original_names[ds.lower()] = ds
        ds = ds.lower()
        if ds in datasets_breakdown.keys():
          if source in datasets_breakdown[ds].keys():
            datasets_breakdown[ds][source].append(task)
          else:
            datasets_breakdown[ds][source] = [task]
        else:
          datasets_breakdown[ds] = {source: [task]}
  # Uses the entries of `original_names` as keys to store the entries from
  # the `datasets_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for dsname in datasets_breakdown:
    output_dict[original_names[dsname]] = datasets_breakdown[dsname]

  return output_dict


def map_dataset_tasks() -> dict:
  '''Gathers all datasets and their supported tasks.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: [<supported-task-1>, <supported-task-2>, ...]

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  dataset_sources_tasks = map_dataset_source_tasks()
  dataset_tasks = dict()

  for ds in dataset_sources_tasks:
    ds_tasks = []

    for source in dataset_sources_tasks[ds].keys():
      for t in dataset_sources_tasks[ds][source]:
        ds_tasks.append(t)
    ds_tasks = list(set(ds_tasks))
    dataset_tasks[ds] = ds_tasks

  return dataset_tasks


def map_dataset_sources(count=False) -> dict:
  '''Gathers all datasets and their source libraries.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: [<source-library-1>, <source-library-2>, ...]

  If count is True, return the count (length) of sources instead

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.
  '''
  dataset_sources_tasks = map_dataset_source_tasks()
  dataset_sources = dict()

  for ds in dataset_sources_tasks:
    dataset_sources[ds] = list(
        dataset_sources_tasks[ds].keys()) if not count else len(
            dataset_sources_tasks[ds].keys())

  return dataset_sources


def map_dataset_info() -> dict:
  '''Gathers all datasets, listing supported tasks and source libraries.

  Builds a dictionary where each entry is of the form:

      <dataset-name>: {
          'tasks': [<supported-task-1>, <supported-task-2>, ...],
          'sources': [<supported-task-1>, <supported-task-2>, ...]
      }

  Returns (dict):
      Dictionary with an entry for all available datasets of the above form.
  '''
  dataset_tasks = map_dataset_tasks()
  dataset_sources = map_dataset_sources()
  dataset_info = dict()

  for ds in dataset_tasks:
    dataset_info[ds] = {
        'sources': dataset_sources[ds],
        'tasks': dataset_tasks[ds]
    }

  return dataset_info


def map_name_source_tasks(nametype: str, return_original_names=True) -> dict:
  '''Gathers all models/datasets and their respective sources and tasks.

  Crawls through all modules to arrange entries of the form:

    <item-name>: {
        <name-source-1>: [<supported-task-11>, <supported-task-12>, ...],
        <name-source-2>: [<supported-task-21>, <supported-task-22>, ...],
        ...
        <name-source-n>: [<supported-task-n1>, <supported-task-n2>, ...],
    }

  Ensures duplicate removals by transforming all strings to lower case, and
  preserving the original names in an additional `original_names` dictionary.

  Args:
    nametype (str):
      Types of names to be used, i.e., either 'models' or 'datasets'.
    return_original_names: if true return source original names, if false return
      unified (lower case) names

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  items_breakdown = dict()
  original_names = dict()

  sources = DATASET_SOURCES if nametype == 'datasets' else MODEL_SOURCES

  for source in sources:
    wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
    items = wrapper.DATASETS if nametype == 'datasets' else wrapper.MODELS
    for task in items:
      for item in items[task]:
        original_names[item.lower()] = item
        item = item.lower()
        if item in items_breakdown.keys():
          if source in items_breakdown[item].keys():
            items_breakdown[item][source].append(task)
          else:
            items_breakdown[item][source] = [task]
        else:
          items_breakdown[item] = {source: [task]}

  # TODO When original_names are replaced, the original name replaced
  # is the last one added to the original_names dict e.g. If vgg exists as
  # VGG and vgg in different sources, the original_names dict will only keep
  # one of those two. We need to fix this evenutally.

  if not return_original_names:
    return items_breakdown

  # Uses the entries of `original_names` as keys to store the entries from
  # the `items_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for itemname in items_breakdown:
    output_dict[original_names[itemname]] = items_breakdown[itemname]

  return output_dict


def map_name_tasks(nametype: str) -> dict:
  '''Gathers all models/datasets and their supported tasks.

  Builds a dictionary where each entry is of the form:

    <item-name>: [<supported-task-1>, <supported-task-2>, ...]

  Args:
    nametype (str):
      Types of names to be used, i.e., either 'models' or 'datasets'.

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  item_sources_tasks = map_name_source_tasks(nametype)
  item_tasks = dict()

  for item in item_sources_tasks:
    it_tasks = []

    for source in item_sources_tasks[item].keys():
      for t in item_sources_tasks[item][source]:
        it_tasks.append(t)
    it_tasks = list(set(it_tasks))
    item_tasks[item] = it_tasks

  return item_tasks


def map_name_sources(nametype: str, return_original_names=True) -> dict:
  '''Gathers all models/datasets and their source libraries.

  Builds a dictionary where each entry is of the form:

    <item-name>: [<source-library-1>, <source-library-2>, ...]

  Args:
    nametype (str):
      Types of names to be used, i.e., either 'models' or 'datasets'.

    return_original_names: if true return source original names, if false return
      unified (lower case) names

  Returns (dict):
    Dictionary with an entry for all available items of the above form.
  '''
  item_sources_tasks = map_name_source_tasks(nametype, return_original_names)
  item_sources = dict()

  for item in item_sources_tasks:
    item_sources[item] = list(item_sources_tasks[item].keys())

  return item_sources


def map_name_info(nametype: str) -> dict:
  '''Gathers all items, listing supported tasks and source libraries.

  Builds a dictionary where each entry is of the form:

      <item-name>: {
          'tasks': [<supported-task-1>, <supported-task-2>, ...],
          'sources': [<supported-task-1>, <supported-task-2>, ...]
      }

  Returns (dict):
      Dictionary with an entry for all available items of the above form.
  '''
  item_tasks = map_name_tasks(nametype)
  item_sources = map_name_sources(nametype)
  item_info = dict()

  for item in item_tasks:
    item_info[item] = {'sources': item_sources[item], 'tasks': item_tasks[item]}

  return item_info


def get_source_from_model(model) -> str:
  '''Returns the source library's name from a model object.

  Args:
    model:
      Model object directly instantiated from a source library. Type is
      dependent on the source library.

  Returns:
    String with the name of the source library.
  '''
  if 'torchvision' in str(type(model)):
    return 'torchvision'
  if 'mxnet' in str(type(model)):
    return 'mxnet'
  if 'keras' in str(type(model)):
    return 'keras'
  raise NotImplementedError(
      'Need source extraction implementation for this type of model!')


def flatten_model(model) -> list:
  '''Returns a list with the model's layers.

  Some models are built with blocks of layers. This function flattens the
  blocks and returns a list of all layers of model. One of its uses is to find
  the number of layers and parameters for a model in a programatic way.

  Args:
    model:
      Model object directly instantiated from a source library. Type is
      dependent on the source library.

  Returns:
    A list of layers, which depend on the model's source library.
  '''
  source = get_source_from_model(model)
  if source in ['keras']:
    return list(model.submodules)

  layers = []
  flatten_model_recursively(model, source, layers)
  return layers


def flatten_model_recursively(block, source: str, layers: list):
  '''Recursive helper function to flatten a model's layers onto a list.

  Args:
    block:
      Model object directly instantiated from a source library, or a block of
      that model. Type is dependent on the source library.
    source: (string)
      The name of the model's source library.
    layers: (list)
      The list of layers to be recursively filled.

  TODO(tonioteran,hugoochoa) Clean this up and unit test! This code seems
  pretty messy...
  '''
  if source == 'mxnet':
    bottleneck_layer = mx.gluon.model_zoo.vision.BottleneckV1
    list1 = dir(bottleneck_layer)
    if 'features' in dir(block):
      flatten_model_recursively(block.features, source, layers)

    elif 'HybridSequential' in str(type(block)):
      for j in block:
        flatten_model_recursively(j, source, layers)

    elif 'Bottleneck' in str(type(block)):
      list2 = dir(block)
      for ll in list1:
        list2.remove(ll)
      subblocks = [x for x in list2 if not x.startswith('_')]
      for element in subblocks:
        attr = getattr(block, element)
        flatten_model_recursively(attr, source, layers)
    else:
      layers.append(block)

  else:
    for child in block.children():
      obj = str(type(child))
      if 'container' in obj or 'torch.nn' not in obj:
        flatten_model_recursively(child, source, layers)
      else:
        layers.append(child)


def get_input_type(model) -> str:
  '''Returns the type of the input data received by the model.

    Args:
      model: a raw model instance

    Returns:
      The input type as a string e.g. numpy.ndarray
  '''
  source = get_source_from_model(model)
  if source in [
      'torchvision', 'mxnet', 'segmentation_models_pytorch', 'pretrainedmodels',
      'fastai', 'mmf', 'gans_pytorch'
  ]:
    return 'torch.Tensor'
  elif source in ['isr', 'segmentation_models', 'keras', 'gans_keras']:
    return 'numpy.ndarray'
  elif source == 'detectron2':
    raise NotImplementedError


def get_input_shape(model) -> str:
  '''Returns the input shape of the input data received by the model.

    Args:
      model: a raw model instance

    Returns:
      The input shape as a tuple
  '''
  source = get_source_from_model(model)
  if source == 'keras':
    return model.layers[0].input_shape[0]
  else:
    raise NotImplementedError


def get_output_shape(model) -> str:
  '''Returns the output shape of the model

    Args:
      model: a raw model instance

    Returns:
      The output shape as a tuple
  '''
  source = get_source_from_model(model)
  if source == 'keras':
    return model.layers[-1].input_shape
  else:
    raise NotImplementedError


def get_dataset_shape(dataset) -> str:
  '''Returns the shape of the dataset

    Args:
      model: a raw dataset instance

    Returns:
      The dataset shape as a tuple
  '''
  source = get_source_from_dataset(dataset)
  if source == 'keras':
    return dataset[0].shape[1:]
  else:
    raise NotImplementedError


def get_num_channels_from_model(model) -> int:
  '''Returns the number of channels that the model requires.

  Three channels corresponds to a color data, while one channel corresponds to
  grayscale data.

  Args:
    model:
      Model object directly instantiated from a source library. Type is
      dependent on the source library.

  Returns:
    An integer with the required data channels for the model.
  '''
  layers = flatten_model(model)
  original_input_type = get_input_type(model)

  if original_input_type == 'torch.Tensor':
    return layers[0].weight.shape[1]

  n_channels = None
  for l in layers:  # noqa E741
    if 'conv' in str(type(l)):
      if len(l.weights) == 0:
        continue
      if len(l.weights) <= 2:
        if isinstance(l.weights[0], list):
          n_channels = np.array(l.weights[0])[0].shape[-2]
        else:
          n_channels = l.weights[0][0].shape[-2]
      else:
        n_channels = np.array(l.weights)[0].shape[-2]
      break
  return n_channels


def get_num_layers_from_model(model) -> int:
  '''Returns the number of layers from a model

  Args:
    model: a model instance

  Returns:
    An integer of the number of layers from a model
  '''
  n_layers = 0
  layers = flatten_model(model)
  for layer in layers:
    layer_name = str(type(layer)).lower()
    conv1_bool = 'conv1d' in layer_name
    conv2_bool = 'conv2d' in layer_name
    conv3_bool = 'conv3d' in layer_name
    linear_bool = 'linear' in layer_name or 'dense' in layer_name

    if conv1_bool or conv2_bool or conv3_bool or linear_bool:
      n_layers += 1
  return n_layers


def get_num_parameters_from_model(model) -> int:
  '''Calculate the number of parameters in model.

  This depends on the number of trainable weights and biases.

  TODO(tonioteran,hugoochoa) Clean this up and unit test! This code seems
  pretty messy... We should try to remove as many indentation levels as
  possible by simplifying the logic of the code.
  '''
  n_params = 0
  layers = flatten_model(model)
  source = get_source_from_model(model)

  # Tensorflow models and pytorch models have distinct attributes
  # Tensorflow has attribute `weigths` while pytorch has `weight`
  input_type = get_input_type(model)
  if input_type == 'torch.Tensor':
    for layer in layers:
      if 'weight' in dir(layer):
        if layer.weight is not None:
          weights = np.array(layer.weight.shape)
          # If a layer do not have a weight, then
          # it won't have a bias either

          if 'bias' in dir(layer):
            if layer.bias is not None:
              if source == 'mxnet':
                bias = layer.bias.shape[0]
              else:
                bias = len(layer.bias)
              params_layer = np.prod(weights) + bias
            else:
              params_layer = np.prod(weights)
          else:
            params_layer = np.prod(weights)
          n_params += params_layer

  else:
    # tf and keras based models
    for layer in layers:
      if 'get_weights' in dir(layer):

        if layer.get_weights() != []:
          if len(layer.get_weights()) <= 2:
            weights = np.array(layer.get_weights()[0]).shape
          else:
            weights = np.array(layer.get_weights()).shape

        # If a layer do not have a weight, then
        # it won't have a bias either

          if 'bias' in dir(layer):

            if layer.bias is not None and layer.use_bias:
              bias = layer.bias.shape[0]
              params_layer = np.prod(weights) + bias
            else:
              params_layer = np.prod(weights)

          else:
            params_layer = np.prod(weights)
          n_params += params_layer

  return n_params


def format_image(x):
  '''Format an image to the shape (1, h, w, c)

  Args:
    x:
      numpy.ndarray or torch.Tensor that represents an image

  Returns:
    Processed numpy.ndarray of shape (1, h, w, c)
  '''

  tensor_shape = x.shape

  if isinstance(x, torch.Tensor):
    x = x.numpy()

  if len(tensor_shape) == 2:
    # We only have one channel.
    x = x.reshape([1, 1, *x.shape])
  elif len(tensor_shape) == 3:
    # We have a dimension for the number of channels (dim [3]).
    x = x.reshape([1, *x.shape])

  if x.shape[3] != 1 and x.shape[3] != 3:
    # Chang, shape (1, c, h, w) to (1, h, w, c)
    x = np.transpose(x, [0, 2, 3, 1])


def get_source_from_dataset(dataset) -> str:
  '''Determines the source library from a dataset object.

  Args:
    dataset:
      Dataset object directly instantiated from a source library. Type
      is dependent on the source library.

  Returns:
    String with the name of the source library.
  '''
  source = None
  # Save the name of the type of the object, without the first 8th digits
  # to remove '<class '' characters.
  obj_type = str(type(dataset))
  if 'class' in obj_type:
    obj_type = obj_type[8:]
  if isinstance(dataset, tuple):
    if len(dataset) == 2 and isinstance(dataset[0], np.ndarray):
      # Keras dataset objects are numpy.ndarray tuples.
      source = 'keras'
  elif 'torch' in obj_type:
    source = 'torchvision'
  else:
    # Dataset source's name is read from the dataset type.
    source = obj_type.split('.')[0]
  if 'tensorflow' in source:
    source = 'tensorflow'
  return source


def get_size_from_dataset(dataset, split_name) -> int:
  '''Returns the total number of images or videos in the split.

  Args:
    dataset:
      Dataset object directly instantiated from a source library. Type
      is dependent on the source library.
    split_name (str):
      Corresponding name for this particular dataset's split.

  Returns:
    The size of the dataset's split.
  '''
  source = get_source_from_dataset(dataset)
  if source == 'keras':
    return len(dataset[0])
  elif source in ['mmf', 'mxnet', 'tensorflow']:
    return len(dataset)
  elif source == 'fastai':
    images = getattr(dataset, split_name + '_ds')
    return len(images)
  elif source == 'torchvision':
    if 'dataset' in dir(dataset):
      return len(dataset.dataset.data)
    else:
      return len(dataset)


def get_shape_from_dataset(dataset, name, split_name):
  '''Returns (height, width, channels) tuple.

    Args:
      dataset: a dataset instance
      name: the dataset name
      split_name: the dataset split name

    Returns:
      A tuple (height, width, channels)
  '''
  # Sample uniformly some images. If the shapes of each image
  # are different, then a None will be in the corresponding
  # dimension of the shape
  source = get_source_from_dataset(dataset)
  if source == 'tensorflow':
    _, ds_info = tfds.load(name, with_info=True)
    if 'image' in ds_info.features.keys():
      (h, w, c) = ds_info.features['image'].shape
    else:
      (h, w, c) = (None, None, None)

  else:
    n = get_size_from_dataset(dataset, split_name)

    # Chose 10 samples and list their shapes
    indexes = np.random.choice(range(n), 10, replace=False)
    shapes = []
    for i in indexes:
      shapes.append(dataset.__getitem__(i)['image'].shape)
    shapes = np.array(shapes)

    h, w, c = None, None, None

    # Check whether shapes are different
    if source == 'mmf':
      if len(set(shapes[:, 0])) == 1 and len(set(shapes[:, 1])) == 1:
        h = shapes[0][0]
        w = shapes[0][1]
    else:
      if len(set(shapes[:, 1])) == 1 and len(set(shapes[:, 2])) == 1:
        h = shapes[0][1]
        w = shapes[0][2]

      if len(set(shapes[:, 3])) == 1:
        c = shapes[0][3]

  return (h, w, c)


def get_classes_from_dataset(raw_object, source, name, split_name, size):
  '''Get the IDs and the names (if available) of the classes.

    Args:
        raw_object: dataset raw object as stored in CvDataset (split raw object)
        source: the source of the dataset in string
        name: the dataset name in string
        split_name: the split name to which the raw_object belongs to in string
        size: the size of the datset raw_object

    Returns:
        A pair of values, `classes` and `classes_names`. If no
        `classes_names` are available, the pair becomes `classes` and
        `None`.
    '''
  classes = None
  classes_names = None
  if source == 'mxnet':
    classes = set(raw_object[split_name][:][1])
    classes_names = None
  elif source == 'keras':
    classes = np.unique(raw_object[1])
    classes_names = None
  elif source == 'torchvision':
    if 'VOC' in name:
      # If dataset is an Object Detection Dataset
      classes_names = [
          'unlabeled/void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
          'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
          'tv/monitor'
      ]
      classes = list(range(21))
    elif 'class_to_idx' in dir(raw_object[split_name]):
      classes = list(raw_object[split_name].class_to_idx.values())
      classes_names = list(raw_object[split_name].class_to_idx.keys())
    elif 'dataset' in dir(raw_object[split_name]):
      if 'class_to_idx' in dir(raw_object[split_name]):
        classes = list(raw_object[split_name].class_to_idx.values())
        classes_names = list(raw_object[split_name].class_to_idx.keys())
      else:
        classes = list(set(raw_object[split_name].targets))
        classes_names = None
    elif 'labels' in dir(raw_object[split_name]):
      classes = list(set(raw_object[split_name].labels.numpy()))
      classes_names = None
    else:
      classes = []
      finished = True
      # Set limited time to go through dataset and obtain classes
      time_end = time.time() + 20
      for i in range(size):
        print(i)
        # Append the label of each example
        # Not working (delete get_item from abstraction)
        # classes.append(
        #     get_item(None, i, raw_object[split_name], source)['label'])
        if time.time() > time_end:
          # Execute stopping condition
          finished = False
          break
      if finished:
        classes = list(set(classes))
      else:
        classes = None
      classes_names = None

  elif source == 'tensorflow':
    _, ds_info = tfds.load(name, with_info=True)
    classes = None
    classes_names = None

    if 'label' in ds_info.features:
      n_classes = ds_info.features['label'].num_classes
      classes = range(n_classes)

  elif source == 'fastai':
    obj = getattr(raw_object[split_name], split_name + '_ds')
    classes_names = obj.y.classes
    classes = range(len(classes_names))

  if classes is not None:
    if list(range(len(classes))) == list(classes):
      # Avoid representing classes with a long list by using range
      classes = range(len(classes))

  return classes, classes_names


def extract_pixel_types(raw_object, name, source, split_name):
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
  if 'VOC' in name or 'SBD' in name:
    classes = [
        'unlabeled/void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]
    indexes = list(range(21))
  elif source == 'fastai':
    obj = getattr(raw_object, split_name + '_ds')
    classes = obj.y.classes
    indexes = None
  else:
    indexes, classes = None, None
  return indexes, classes


def compare_shapes(ground_truth_shape, shape):
  '''Compare to shapes to see whether they are equivalent

  Args:
    ground_truth_shape: the ground truth shape tuple. A None value in this tuple
    will act as a wildcard and will match any value in the compared shape.
    shape: the shape to compare against ground_truth_shape

  Returns:
    Boolean that tells whether shapes are equivalent or not
  '''
  equal = False
  if ground_truth_shape == shape:
    equal = True
  elif len(ground_truth_shape) == len(shape):
    matched_items = 0
    for truth, value in zip(ground_truth_shape, shape):
      if truth is None or truth == value:
        matched_items += 1
    if matched_items == len(ground_truth_shape):
      equal = True
  return equal
