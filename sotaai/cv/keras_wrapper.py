# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Keras https://keras.io/ wrapper module
'''

from sotaai.cv import utils
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
import numpy as np

SOURCE_METADATA = {
    'name': 'keras',
    'original_name': 'Keras',
    'url': 'https://keras.io/'
}

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


def load_model(
    model_name,
    pretrained=False,
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    input_tensor=None,
    input_shape=None,
    # TODO(Hugo)
    # Once standardized input is defined (configs), this param should be put by
    # the end-user
    # As per Keras docs, it is important to set include_top to
    # false to be able to modify model input/output
    include_top=False,
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
                    include_top=include_top,
                    pooling=pooling,
                    classes=classes)
  elif model_name == 'MobileNetV2':
    model = trainer(weights=weights,
                    alpha=alpha,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    include_top=include_top,
                    pooling=pooling,
                    classes=classes)
  else:
    model = trainer(weights=weights,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                    include_top=include_top,
                    pooling=pooling,
                    classes=classes,
                    classifier_activation=classifier_activation)

  return model


def load_dataset(dataset_name, download=True):
  '''Load a given dataset with all its splits

    Args:
      dataset_name (string): name of dataset
      download: temporal flag to skip download and only create the dataset
        instance with no data (used for JSONs creation)

    Returns:
      Dict with keys {'train':(x_train, y_train), 'test':(x_test,y_test),
      Each entry is a numpy array
    '''

  if download:
    dataset = getattr(keras.datasets, dataset_name)
    dataset = dataset.load_data()
    dataset_dict = {'train': dataset[0], 'test': dataset[1]}
  else:
    return {'train': {'name': dataset_name, 'source': 'tensorflow'}}

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

  print('Making compatible {} with {}...'.format(cv_model.name,
                                                 cv_dataset.name))

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

  min_input_shape = None

  if cv_model.name in utils.IMAGE_MINS:
    min_input_shape = (utils.IMAGE_MINS[cv_model.name],
                       utils.IMAGE_MINS[cv_model.name])

  # TODO(Hugo)
  # When datasets have a None width/height is not possible to globally know
  # whether it matches the min_input_shape since this has to be done per
  # image. We have to take this into account in the image_preprocessing_callback
  has_min_shape = False
  if cv_dataset.shape[:2] != (None, None):
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
    raw_model = load_model(
        cv_model.name,
        input_tensor=input_tensor,
        # As per Keras docs, it is important to set include_top to
        # false to be able to modify model input/output
        include_top=False)

    cv_model.update_raw_model(raw_model)

  # Case 4:
  # If output is not compatible with dataset classes, we have to change the
  # model output layer
  is_output_compatible = utils.compare_shapes(cv_model.original_output_shape,
                                              cv_dataset.classes_shape)

  if not is_output_compatible:
    print(' => Model Output from {} to {}'.format(
        cv_model.original_output_shape, cv_dataset.classes_shape))

    # TODO(Hugo)
    # Further review this.
    # As read in some Keras blogs for Transfer Learning, there are 3 possible
    # ways to change Keras output model to a different number of classes:
    # - Use classes parameter, however this only work when include_top=true
    #  which requires a fixed input shape which is not usually the case.
    #  Therefore, this way was discarded.
    # - Use Keras function API to add a new Flatten layer after the
    # last pooling layer in the raw model, then define a new classifier model
    # with a Dense fully connected layer and an output layer that will predict
    # the probability for dataset classes. This one did not work, it has issues
    # when model input shape is dynamic e.g. (None,None,3)
    # - Add an Average Pooling Layer at the end, then define a classifier with
    # a Dense fully connected layer and an output layer that will predict the
    # probability for the dataset classes. This is the one Keras uses in
    # their models when include_top=true and classes are given. This is the one
    # that worked well and the method used as of now to change model output,
    # however still not sure if it is the best way.
    avg_pooling_layer = GlobalAveragePooling2D(name='avg_pool')(
        cv_model.raw.layers[-1].output)
    output = Dense(cv_dataset.classes_shape[0],
                   activation='softmax')(avg_pooling_layer)
    raw_model = Model(inputs=cv_model.raw.inputs, outputs=output)

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
