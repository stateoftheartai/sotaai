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

TEST_MODELS = {
    'classification': [{
        'name': 'InceptionResNetV2',
        'num_layers': 245,
        'input_type': 'numpy.ndarray',
        'num_parameters': 55873736
    }, {
        'name': 'InceptionV3',
        'num_layers': 95,
        'input_type': 'numpy.ndarray',
        'num_parameters': 23851784
    }, {
        'name': 'ResNet101V2',
        'num_layers': 105,
        'input_type': 'numpy.ndarray',
        'num_parameters': 44675560
    }, {
        'name': 'ResNet152V2',
        'num_layers': 156,
        'input_type': 'numpy.ndarray',
        'num_parameters': 60380648
    }, {
        'name': 'ResNet50V2',
        'num_layers': 54,
        'input_type': 'numpy.ndarray',
        'num_parameters': 25613800
    }, {
        'name': 'VGG16',
        'num_layers': 16,
        'input_type': 'numpy.ndarray',
        'num_parameters': 138357544
    }, {
        'name': 'VGG19',
        'num_layers': 19,
        'input_type': 'numpy.ndarray',
        'num_parameters': 143667240
    }, {
        'name': 'Xception',
        'num_layers': 41,
        'input_type': 'numpy.ndarray',
        'num_parameters': 3357584
    }, {
        'name': 'ResNet50',
        'num_layers': 53,
        'input_type': 'numpy.ndarray',
        'num_parameters': 23587712
    }, {
        'name': 'ResNet101',
        'num_layers': 104,
        'input_type': 'numpy.ndarray',
        'num_parameters': 42658176
    }, {
        'name': 'ResNet152',
        'num_layers': 155,
        'input_type': 'numpy.ndarray',
        'num_parameters': 58370944
    }, {
        'name': 'DenseNet121',
        'num_layers': 120,
        'input_type': 'numpy.ndarray',
        'num_parameters': 7037504
    }, {
        'name': 'DenseNet169',
        'num_layers': 168,
        'input_type': 'numpy.ndarray',
        'num_parameters': 12642880
    }, {
        'name': 'DenseNet201',
        'num_layers': 200,
        'input_type': 'numpy.ndarray',
        'num_parameters': 18321984
    }, {
        'name': 'NASNetMobile',
        'num_layers': 196,
        'input_type': 'numpy.ndarray',
        'num_parameters': 2249533
    }, {
        'name': 'NASNetLarge',
        'num_layers': 268,
        'input_type': 'numpy.ndarray',
        'num_parameters': 43614774
    }, {
        'name': 'MobileNet',
        'num_layers': 28,
        'input_type': 'numpy.ndarray',
        'num_parameters': 4253864
    }, {
        'name': 'MobileNetV2',
        'num_layers': 53,
        'input_type': 'numpy.ndarray',
        'num_parameters': 3538984
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
    '''Get the next item from the dataset in a standardized format.

    Returns:
      A dict with two mandatory keys: the 'image' key which will hold the image
      as a numpy array, and the 'label' key which will hold the label as a numpy
      array as well. The dict might contain other keys depending on the nature
      of the dataset.
    '''
    image = next(self._iterator['image'])
    label = next(self._iterator['label'])
    return {'image': image, 'label': label}

  def _create_iterator(self):
    '''Create an iterator out of the raw dataset split object

    Returns:
      An object containing iterators for the dataset images and labels
    '''
    return {'image': iter(self._raw[0]), 'label': iter(self._raw[1])}
