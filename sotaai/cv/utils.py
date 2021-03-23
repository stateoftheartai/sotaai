# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Useful utility functions to navigate the library's available resources.'''
# TODO(tonioteran) Deprecate specific dataset/model functions for the
# generalized version.
import importlib
import numpy as np
import tensorflow_datasets as tfds
import time
import os
from re import search
import skimage.transform as ski_transform
from random import randrange

# Prevent Tensorflow to print warning and meta logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SOURCES = [
    'tensorflow', 'keras', 'torch', 'fastai', 'mxnet', 'pretrainedmodels', 'mmf'
]

IMPLEMENTED_SOURCES = ['keras', 'torch', 'tensorflow']

IMAGE_MINS = {
    'InceptionV3': 75,
    'InceptionResNetV2': 75,
    'Xception': 71,
    'VGG16': 32,
    'VGG19': 71,
    'vgg16_bn': 32,
    'ResNet50': 32,
    'ResNet101': 32,
    'ResNet152': 32,
    'ResNet50V2': 32,
    'ResNet101V2': 32,
    'ResNet152V2': 32,
    'resnet18': 32,
    'resnet34': 32,
    'resnext101_32x8d': 32,
    'resnext50_32x4d': 32,
    'MobileNet': 32,
    'MobileNetV2': 32,
    'mobilenet_v2': 32,
    'DenseNet121': 32,
    'DenseNet169': 32,
    'DenseNet201': 32,
    'densenet161': 32,
    'NASNetLarge': 32,
    'NASNetMobile': 32
}

PIXELS_CLASSES = {'lost_and_found': 44, 'cityscapes': 35, 'scene_parse150': 150}


def split_sources(sources):
  '''Temporal function to split sources in two groups: those that are already
  fully implmented, and those who are not

  Args:
    sources: array of sources name to split

  Returns:
    implemented: array of sources fully implemented
  '''
  implemented = filter(lambda s: s in IMPLEMENTED_SOURCES, sources)
  not_implemented = filter(lambda s: s not in IMPLEMENTED_SOURCES, sources)
  return list(implemented), list(not_implemented)


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
      unified (lower case) names. When more than one original name give the same
      unified name, then only one of those will be returned (the last one being
      captured in the `original_names` dictionary) e.g. resnet might have two
      original names (Resnet and ResNet), then only one of those will be
      returned.

  Returns (dict):
    Dictionary with an entry for all available items of the above form.

  TODO(tonioteran) THIS SHOULD BE CACHED EVERY TIME WE USE IT.
  '''
  items_breakdown = dict()
  original_names = dict()

  sources = SOURCES

  # TODO When original_names are replaced, the original name replaced
  # is the last one written to the original_names dict e.g. If vgg exists as
  # VGG and vgg in different sources, the original_names dict will only keep
  # one of those two. We need to fix this evenutally.

  # TODO(Hugo)
  # Once all sources are fully implemented remove this re-ordering
  # Sources are reoredered from not_implemented to implemented to keep the
  # original_name of the implemented version (in case the model/dataset exists
  # in a fully implemented source and in one not implemented as well). This
  # allow us to use as default the implemented one when no source name is
  # given to load_model or load_dataset.
  # Example: if alexnet exists in FastAI and Keras, we want to preserve the
  # original name of Keras so that we can import it from there by default.
  implemented, not_implemented = split_sources(sources)
  sources = not_implemented + implemented

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

  if not return_original_names:
    return items_breakdown

  # Uses the entries of `original_names` as keys to store the entries from
  # the `items_breakdown` dict, which uses lowercase names as keys.
  output_dict = dict()
  for itemname in items_breakdown:
    output_dict[original_names[itemname]] = items_breakdown[itemname]

  return output_dict


def map_source_metadata() -> dict:
  '''Return a map between the source name and its original name

  Crawls through all modules to arrange entries of the form:

    <source-name>: <source-original-name>

  Returns (dict):
    Dictionary with an entry for all available items of the above form.
  '''
  items_breakdown = dict()

  sources = SOURCES

  for source in sources:
    wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')
    items_breakdown[source] = wrapper.SOURCE_METADATA

  return items_breakdown


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
    sources = list(item_sources_tasks[item].keys())

    # TODO(Hugo)
    # Once all sources are fully implemented remove this re-ordering
    # Make sure to return implemented sources first, so that the source used by
    # default is an implemented one
    implemented, not_implemented = split_sources(sources)
    item_sources[item] = list(implemented) + list(not_implemented)

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
  # Non-implemented models
  if isinstance(model, dict) and 'source' in model:
    return model['source']
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
  # TODO(team)
  # Uncomment this code once mxnet is fully-implemented
  # import mxnet as mx
  # if source == 'mxnet':
  # bottleneck_layer = mx.gluon.model_zoo.vision.BottleneckV1
  # list1 = dir(bottleneck_layer)
  # if 'features' in dir(block):
  # flatten_model_recursively(block.features, source, layers)

  # elif 'HybridSequential' in str(type(block)):
  # for j in block:
  # flatten_model_recursively(j, source, layers)

  # elif 'Bottleneck' in str(type(block)):
  # list2 = dir(block)
  # for ll in list1:
  # list2.remove(ll)
  # subblocks = [x for x in list2 if not x.startswith('_')]
  # for element in subblocks:
  # attr = getattr(block, element)
  # flatten_model_recursively(attr, source, layers)
  # else:
  # layers.append(block)
  # else:
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
  '''Returns the input shape of the data received by the model (image shape)

    Args:
      model: a raw model instance

    Returns:
      The input shape as a tuple e.g. (224,224,3)
  '''
  source = get_source_from_model(model)
  if source == 'keras':
    input_shape = model.layers[0].input_shape
    if isinstance(input_shape, list):
      return input_shape[0][1:]
    else:
      return input_shape[1:]
  elif source == 'torchvision':
    ci, co, w, h = list(model.parameters())[0].shape
    input_shape = (ci, co, w, h)
    return input_shape
  else:
    raise NotImplementedError


def get_output_shape(model) -> str:
  '''Returns the output shape of the predicted data of the model (vector of the
    predicted classes/labels)

    Args:
      model: a raw model instance

    Returns:
      The output shape as a tuple e.g (1000,)
  '''
  source = get_source_from_model(model)
  if source == 'keras':
    # TODO(Hugo)
    # We need to further review this, output shape depends on whether the model
    # is pretrained or not, if pretrained the output_shape match the dataset
    # classes, but if not pretrained the output_shape can be anything in the
    # last layer. We need to check how to better use this or whether to define
    # new variables (this is important since model_to_dataset depends on it)
    # As of now, we just take the last item of the last layer shape e.g. with
    # a model pretrained in cifar10 the last layer is (None,10) but we return
    # (10,) which means the output_shape is a vector with 10 entries.
    last_layer_shape = model.layers[-1].output_shape
    last_item = last_layer_shape[-1]
    output_shape = (last_item,)
    return output_shape
  elif source == 'torchvision':
    last_output = 0
    modules = model.__dict__
    attributes = list(modules['_modules'])
    last_layer = getattr(model, attributes[-1])
    while hasattr(list(last_layer.children()), '__getitem__'):
      if len(list(last_layer.children())) > 0:
        last_layer = list(last_layer.children())[-1]
      else:
        break

    if hasattr(last_layer, 'out_features'):
      last_output = last_layer.out_features
      return (last_output,)
    elif hasattr(last_layer, 'out_channels'):
      # This case was added for Segmentation models, and as per Torch docs,
      # all its segmentition models require and input image size of (224,224)
      # which entails the output shape (the pixel mask) must be the same size as
      # well but containing one layer or extra dimension per pixel class:

      last_output = last_layer.out_channels
      return (last_output, 224, 224)
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

  return int(n_params)


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
  # Non-implemented dataset
  if isinstance(dataset, dict) and 'source' in dataset:
    return dataset['source']
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
  if source == 'keras':
    return dataset[0].shape[1:]
  if source == 'tensorflow':
    _, ds_info = tfds.load(name, with_info=True)
    # For Classification
    if 'image' in ds_info.features.keys():
      (h, w, c) = ds_info.features['image'].shape
    # For Segmentation
    elif 'image_left' in ds_info.features.keys():
      (h, w, c) = ds_info.features['image_left'].shape
    else:
      (h, w, c) = (None, None, None)

  else:
    n = get_size_from_dataset(dataset, split_name)

    # Chose 10 samples and list their shapes
    indexes = np.random.choice(range(n), 10, replace=False)
    shapes = []
    if source == 'torchvision':
      for i in indexes:
        item = dataset[i][0]
        if search('Image', str(type(item))):
          width, height = item.size
          channels = len(item.mode)
          shape = [channels, height, width]
          shapes.append(shape)
        else:
          shapes.append(item.shape)

    else:
      for i in indexes:
        shapes.append(dataset.__getitem__(i)['image'].shape)

    shapes = np.array(shapes)

    h, w, c = None, None, None
    # Check whether shapes are different
    if source == 'mmf':
      if len(set(shapes[:, 0])) == 1 and len(set(shapes[:, 1])) == 1:
        h = shapes[0][0]
        w = shapes[0][1]

    if source == 'torchvision':
      c = shapes[0][0]
      h = shapes[0][1]
      w = shapes[0][2]
      if len(set(shapes[:, 1])) == 1 and len(set(shapes[:, 2])) == 1:
        c = shapes[0][0]
        h = shapes[0][1]
        w = shapes[0][2]

    else:
      if len(set(shapes[:, 1])) == 1 and len(set(shapes[:, 2])) == 1:
        h = shapes[0][1]
        w = shapes[0][2]

      if len(set(shapes[:, 3])) == 1:
        c = shapes[0][3]

  return (h, w, c)


def get_classes_from_dataset(raw_object, source, name, split_name):
  '''Get the IDs and the names (if available) of the classes.

    Args:
        raw_object: dataset raw object as stored in CvDataset (split raw object)
        source: the source of the dataset in string
        name: the dataset name in string
        split_name: the split name to which the raw_object belongs to in string
        size: the size of the datset raw_object

    Returns:
        classes: the collection of classes values e.g. range(0,10)
        classes_names: the set of class names, if available else None
        classes_shape: the shape of a vector class e.g. (10,), if available else
          None
    '''
  classes = None
  classes_names = None
  classes_shape = None
  if source == 'mxnet':
    classes = set(raw_object[:][1])
    classes_names = None
  elif source == 'keras':
    classes = np.unique(raw_object[1])
    classes_names = None
    classes_shape = (len(classes),)
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
      classes_shape = (len(classes),)
    elif 'class_to_idx' in dir(raw_object):
      classes = list(raw_object.class_to_idx.values())
      classes_names = list(raw_object.class_to_idx.keys())
      classes_shape = (len(classes),)
    elif 'dataset' in dir(raw_object):
      if 'class_to_idx' in dir(raw_object):
        classes = list(raw_object.class_to_idx.values())
        classes_names = list(raw_object.class_to_idx.keys())
        classes_shape = (len(classes),)
      else:
        classes = list(set(raw_object.targets))
        classes_names = None
        classes_shape = (len(classes),)
    elif 'labels' in dir(raw_object):
      classes = list(set(raw_object.labels))
      classes_names = None
      classes_shape = (len(classes),)
    else:
      classes = []
      finished = True
      # Set limited time to go through dataset and obtain classes
      time_end = time.time() + 20
      for element in raw_object:
        # Append the label of each example
        classes.append(element[1])
        if time.time() > time_end:
          # Execute stopping condition
          finished = False
          break
      if finished:
        classes = list(set(classes))
        classes_shape = (len(classes),)
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
      classes_shape = (len(classes),)

  elif source == 'fastai':
    obj = getattr(raw_object[split_name], split_name + '_ds')
    classes_names = obj.y.classes
    classes = range(len(classes_names))

  if classes is not None:
    if list(range(len(classes))) == list(classes):
      # Avoid representing classes with a long list by using range
      classes = range(len(classes))

  return classes, classes_names, classes_shape


def extract_pixel_classes(raw_object, name, source, split_name):
  '''Return the IDs and the names (if available) of the pixel classes.

    Args:
      raw_object:
        Dataset object directly instantiated from a source library. Type
        is dependent on the source library.

    Returns:
      classes: an array of numbers belonging to the pixel classes (IDs) of
        the dataset
      clasases_names: an optional array of strings belonging to the pixel
        classes names of the dataset. If not available, then None.
    '''
  if 'VOC' in name or 'SBD' in name:
    classes = list(range(21))
    classes_names = [
        'unlabeled/void', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car ', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]
  elif source == 'tensorflow':
    # For TF datasets, this information cannot be obtained programatically, it
    # has to be collected manually:
    if name in PIXELS_CLASSES:
      classes = list(range(PIXELS_CLASSES[name]))
      classes_names = None
  elif source == 'fastai':
    obj = getattr(raw_object, split_name + '_ds')
    classes = None
    classes_names = obj.y.classes
  else:
    classes, classes_names = None, None
  return classes, classes_names


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


def resize_image(im, shape):
  '''Resize an image

  As of now this function uses scikit-image but implementation can be changed in
  the future to use another library or our own implementation

  Args:
    image: numpy array of the image to be resized
    shape: the new size (shape) as a tuple

  Returns:
    The numpy array of the image resized
  '''
  return ski_transform.resize(im, shape)


def create_segmentation_image(mask, pixel_classes):
  '''Returns an RGB image of the given mask.

  It assigns a random RGB color to each class, and creates an image where each
  pixel is set to the RGB color of its respective class.

  Args:
    mask: the predicted mask of an image (a single item of the output of
      a segmentation model)
    pixel_classes: the number of pixel classes available. This is the number
      of total classes the mask might contain (the pixel classes of the
      dataset the original image belongs to)
  '''

  classes_colors = []
  for _ in range(0, pixel_classes):
    classes_colors.append((randrange(256), randrange(256), randrange(256)))
  classes_colors = np.array(classes_colors)

  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)

  for l in range(0, pixel_classes):
    idx = mask == l
    r[idx] = classes_colors[l, 0]
    g[idx] = classes_colors[l, 1]
    b[idx] = classes_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb


def get_input_shape_min(model_name: str) -> tuple:
  '''Returns the model minimum allowed input shape for height and width

  Args:
    model_name: the model name as string

  Returns:
    A tuple with (min_height, min_width) which represents the model minimum
    input shape. If model does not have a minimum it returns None entries
  '''
  if model_name in IMAGE_MINS:
    return (IMAGE_MINS[model_name], IMAGE_MINS[model_name])
  else:
    return (None, None)
