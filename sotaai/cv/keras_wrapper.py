# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Keras https://keras.io/ wrapper module
'''

import tensorflow.keras as keras

DATASETS = {'classification': ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']}

# @author HO
# As of now, only missing EfficientNetBX
#
MODELS = {
    'classification': [
        'InceptionResNetV2', 'InceptionV3', 'ResNet101V2', 'ResNet152V2',
        'ResNet50V2', 'VGG16', 'VGG19', 'Xception', 'ResNet50', 'ResNet101',
        'ResNet152', 'DenseNet121', 'DenseNet169', 'DenseNet201',
        'NASNetMobile', 'NASNetLarge', 'MobileNet', 'MobileNetV2'
    ]
}

TEST_DATASETS = {
    'classification': [{
        'name': 'mnist',
        'train_size': 60000,
        'test_size': 10000
    }, {
        'name': 'cifar10',
        'train_size': 50000,
        'test_size': 10000
    }, {
        'name': 'cifar100',
        'train_size': 50000,
        'test_size': 10000
    }, {
        'name': 'fashion_mnist',
        'train_size': 60000,
        'test_size': 10000
    }]
}

TEST_MODELS = {
    'classification': [{
        'name': 'InceptionResNetV2',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'InceptionV3',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet101V2',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet152V2',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet50V2',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'VGG16',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'VGG19',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'Xception',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet50',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet101',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'ResNet152',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'DenseNet121',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'DenseNet169',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'DenseNet201',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'NASNetMobile',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'NASNetLarge',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'MobileNet',
        'input_type': 'numpy.ndarray'
    }, {
        'name': 'MobileNetV2',
        'input_type': 'numpy.ndarray'
    }]
}


def load_model(model_name,
               pretrained=False,
               alpha=1.0,
               depth_multiplier=1,
               dropout=0.001,
               input_tensor=None,
               input_shape=None,
               include_top=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
  '''Load a model with specific configuration.

    Args:
      model_name (string): name of the model/algorithm.
          include_top: whether to include the fully-connected layer at the
          top of the network.
      weights: one of None (random initialization), 'imagenet'
          (pre-training on ImageNet), or the path to the weights file to be
          loaded.
      input_tensor: optional Keras tensor (i.e. output of layers.Input())
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified if include_top
          is False (otherwise the input shape has to be (299, 299, 3). It
          should have exactly 3 inputs channels, and width and height should
          be no smaller than 71.
          E.g. (150, 150, 3) would be one valid value.
      pooling: Optional pooling mode for feature extraction when include_top
          is False.
          None means that the output of the model will be the 4D tensor
              output of the last convolutional block.
          avg means that global average pooling will be applied to the output
              of the  last convolutional block, and thus the output of the
              model will be a 2D tensor.
          max means that global max pooling will be applied.
      alpha: Controls the width of the network. This is known as the width
          multiplier in the MobileNet paper. - If alpha < 1.0, proportionally
          decreases the number of filters in each layer. - If alpha > 1.0,
          proportionally increases the number of filters in each layer.
          - If alpha = 1, default number of filters from the paper are used
          at each layer. Default to 1.0.
      depth_multiplier: Depth multiplier for depthwise convolution. This is
          called the resolution multiplier in the MobileNet paper. Default
          to 1.0.
      dropout: Dropout rate. Default to 0.001.
      classes: optional number of classes to classify images into, only to be
          specified if include_top is True, and if no weights argument is
          specified.
      classifier_activation: A str or callable. The activation function to
          use on the 'top' layer. Ignored unless include_top=True. Set
      classifier_activation=None to return the logits of the 'top' layer.

    Returns:
      tensorflow.python.keras model
    '''
  if pretrained:
    weights = 'imagenet'
  else:
    weights = None

  # Load the models.\model_name\ class
  trainer = getattr(keras.applications, model_name)

  # Load the model and return
  if model_name in [
      'ResNet50', 'ResNet101', 'ResNet152', 'DenseNet121', 'DenseNet169',
      'DenseNet201', 'NASNetMobile', 'NASNetLarge'
  ]:
    model = trainer(weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    include_top=include_top,
                    pooling=pooling,
                    classes=classes)
  elif model_name == 'MobileNet':
    model = trainer(weights=weights,
                    alpha=alpha,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes)
  elif model_name == 'MobileNetV2':
    model = trainer(weights=weights,
                    alpha=alpha,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes)
  else:
    model = trainer(weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    pooling=pooling,
                    classes=classes,
                    classifier_activation=classifier_activation)

  return model


def load_dataset(dataset_name):
  '''Load a given dataset with all its splits

    Args:
      dataset_name (string): name of dataset

    Returns:
      Dict with keys {'train':(x_train, y_train), 'test':(x_test,y_test),
      Each entry is a numpy array
    '''

  dataset = getattr(keras.datasets, dataset_name)
  dataset = dataset.load_data()
  dataset_dict = {'train': dataset[0], 'test': dataset[1]}

  return dataset_dict


class DatasetIterator():
  '''Keras dataset iterator class'''

  def __init__(self, raw) -> None:
    self._raw = raw
    self._iterator = self._create_iterator()

  def __next__(self):
    '''Get the next item from the dataset.

    Returns: a dict. The dict will contain a 'data' key which will hold the
      datapoint as a numpy array. The dict will also contain a 'label' key which
      will hold the label of the datapoint. The dict might contain other keys
      depending on the nature of the dataset.
    '''
    image = next(self._iterator['image'])
    label = next(self._iterator['label'])
    return self._create_item(image, label)

  def _create_iterator(self):
    return {'image': iter(self._raw[0]), 'label': iter(self._raw[1])}

  def _create_item(self, image, label):
    return {'image': image, 'label': label}
