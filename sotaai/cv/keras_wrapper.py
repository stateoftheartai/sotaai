# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Keras https://keras.io/ wrapper module
'''

from sotaai.cv import utils
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
import numpy as np

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


def model_to_dataset(cv_model, cv_dataset):
  '''If compatible, adjust model and dataset so that they can be executed
  against each other

  Args:
    cv_model: an abstracted cv model whose source is Keras
    cv_dataset: an abstracted cv dataset

  Returns:
    cv_model: the abstracted cv model adjusted to be executed against
      cv_dataset
    cv_dataset: the abstracted cv dataset adjust to be executed against
      cv_model
  '''

  print('Adjusting...')

  # Case 1:
  # All Keras models require 3 channels, thus we have to reshape the dataset
  # if less than 3 channels

  are_channels_compatible = len(cv_dataset.shape) == len(
      cv_model.original_input_shape
  ) and cv_dataset.shape[-1] == cv_model.original_input_shape[-1]

  if not are_channels_compatible:
    if len(cv_dataset.shape) == 2:
      fixed_channels_shape = cv_dataset.shape + (3,)
    else:
      fixed_channels_shape = cv_dataset.shape[:2] + (3,)
    print(' => Dataset Channels from {} to {}'.format(cv_dataset.shape,
                                                      fixed_channels_shape))
    cv_dataset.shape = fixed_channels_shape

  # Case 2:
  # As per Keras documentation, some models require a minimum width and height
  # for the input shape. For those models, we make sure the dataset meet those
  # minimums
  image_mins = {
      'InceptionV3': 75,
      'InceptionResNetV2': 75,
      'Xception': 71,
      'VGG16': 32,
      'VGG19': 71,
      'ResNet50': 32,
      'ResNet101': 32,
      'ResNet152': 32,
      'ResNet50V2': 32,
      'ResNet101V2': 32,
      'ResNet152V2': 32,
      'MobileNet': 32,
      'DenseNet121': 32,
      'DenseNet169': 32,
      'DenseNet201': 32,
      'NASNetLarge': 32,
      'NASNetMobile': 32
  }

  min_input_shape = None

  if cv_model.name in image_mins:
    min_input_shape = (image_mins[cv_model.name], image_mins[cv_model.name])

  has_min_shape = min_input_shape and cv_dataset.shape[:2] < min_input_shape

  if has_min_shape:
    original_dataset_shape = cv_dataset.shape
    cv_dataset.shape = min_input_shape + (3,)
    print(' => Dataset minimum shape from {} to {}'.format(
        original_dataset_shape, cv_dataset.shape))

  # Case 3:
  # If dataset and model input are not compatible, we have to (1) reshape
  # the dataset shape a bit more or (2) change the model input layer

  is_input_compatible = utils.compare_shapes(cv_model.original_input_shape,
                                             cv_dataset.shape)

  if not is_input_compatible:

    print(' => Model Input from {} to {}'.format(cv_model.original_input_shape,
                                                 cv_dataset.shape))

    input_tensor = Input(shape=cv_dataset.shape)
    raw_model = load_model(cv_model.name, input_tensor=input_tensor)

    cv_model.update_raw_model(raw_model)

  # Case 3:
  # If output is not compatible with dataset classes, we have to change the
  # model output layer
  is_output_compatible = utils.compare_shapes(cv_model.original_output_shape,
                                              cv_dataset.classes_shape)

  if not is_output_compatible:
    print(' => Model Output from {} to {}'.format(
        cv_model.original_output_shape, cv_dataset.classes_shape))

    # Some models were able to be modified by adding a new layer at the end,
    # however it does not work for all of them e.g. ResNet50. Thus it is better
    # to use the classes parameter to modify the output.

    # Apprach 1: Adding a new layer at the end
    # raw_model = Sequential()
    # raw_model.add(cv_model.raw)
    # raw_model.add(Dense(cv_dataset.classes_shape[0], activation='softmax'))

    # Approach 2: Using classes parameter
    input_tensor = Input(shape=cv_dataset.shape)
    raw_model = load_model(cv_model.name,
                           input_tensor=input_tensor,
                           include_top=True,
                           classes=cv_dataset.classes_shape[0])

    cv_model.update_raw_model(raw_model)

  # Some of the cases above are  managed at dataset iterator level, that
  # is why a callback is passed in. The iterator will reshape the dataset items
  # using this callback and thus taking into account the cases above as
  # required by the model.

  def image_preprocessing_callback(image):

    if has_min_shape:
      image = utils.resize_image(image, min_input_shape)

    if not are_channels_compatible:
      if len(image.shape) == 2:
        image = image.reshape(image.shape + (1,))
      image = np.repeat(image, 3, -1)

    return image

  cv_dataset.set_image_preprocessing(image_preprocessing_callback)

  # Finally, the compatibilized models and dataset are returned

  return cv_model, cv_dataset


class DatasetIterator():
  '''Keras dataset iterator class'''

  def __init__(self, raw) -> None:
    self._raw = raw
    self._iterator = self._create_iterator()
    self._image_preprocessing_callback = None

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

    if self._image_preprocessing_callback:
      image = self._image_preprocessing_callback(image)

    return {'image': image, 'label': label}

  def _create_iterator(self):
    '''Create an iterator out of the raw dataset split object. This is the
    Keras iterator being wrapped in our own iterator.

    Returns:
      An object containing iterators for the dataset images and labels
    '''
    return {'image': iter(self._raw[0]), 'label': iter(self._raw[1])}

  def set_image_preprocessing(self, image_preprocessing_callback):
    self._image_preprocessing_callback = image_preprocessing_callback
