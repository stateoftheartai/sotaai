# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''MxNet https://mxnet.apache.org/ wrapper module.'''
import mxnet as mx
from mxnet.gluon import nn
import copy

DATASETS = {'classification': []}

# All models here are for classification.
MODELS = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'inceptionv3',
    'mobilenet0.25',  # width multiplier 0.25.
    'mobilenet0.5',
    'mobilenet0.75',
    'mobilenet1.0',
    'mobilenetv2_0.25',
    'mobilenetv2_0.5',
    'mobilenetv2_0.75',
    'mobilenetv2_1.0',
    'resnet101_v1',
    'resnet101_v2',
    'resnet152_v1',
    'resnet152_v2',
    'resnet18_v1',
    'resnet18_v2',
    'resnet34_v1',
    'resnet34_v2',
    'resnet50_v1',
    'resnet50_v2',
    'squeezenet1.0',
    'squeezenet1.1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn'
]


def load_model(model_name, classes=1000, pretrained=False):
  '''
    Input:
        model_name: string, one of MODELS variable
        pretrained: bool, to get a pretrained backbone
        classes: int, number of categories to classify
    Output:
        A model of type mxnet.gluon.model_zoo.vision,
        which inherits torch.nn.Module
    '''
  mod = mx.gluon.model_zoo.vision.get_model(model_name,
                                            classes=classes,
                                            pretrained=pretrained)
  if not pretrained:
    mod.initialize()  # Random initialization
  return mod


def load_dataset(dataset_name):
  '''
    Input:
        dataset_name: str, one of DATASETS variable
    Output:
        tuple, where each entry is a mxnet.gluon.data class.
          The first entry is the training partition of the dataset,
          The second is the testing partition.
    '''
  # Load mx.gluon.data.vision.\dataset_name\ class
  ds = getattr(mx.gluon.data.vision, dataset_name)
  ds_train = ds(train=True)
  ds_test = ds(train=False)
  return {'train': ds_train, 'test': ds_test}


def predict(model, dataset):
  '''
    Input:
        model of type pretrained.models
        dataset: torch.Tensor of shape (N,C,H,W)

        4-dimensional torch tensor of shape (N,C,H,W), where
            N - nbr of images
            C - nbr of channels
            H - height of image
            W - width of image
    Output:
        Mask in torch.Tensor
        If model has classification head, then it also returns a classification
    '''
  model(dataset)


def take(dataset, index):
  '''
    Input:
        dataset: mxnet.gluon.data.vision.datasets

    Output:
        image: NDArray
        label: int
    '''
  image, label = dataset[index]
  return image, label


def adapt_last_layer(model, classes: int):
  mod = copy.deepcopy(model)
  if 'squeezenet' in str(type(mod)):
    hybrid_block = mod.output

    args_conv = hybrid_block[0]._kwargs  # pylint: disable=W0212
    bias = not args_conv['no_bias']

    act = hybrid_block[1]._act_type  # pylint: disable=W0212

    args_pool = hybrid_block[2]._kwargs  # pylint: disable=W0212
    ceil_mode = bool(args_pool['pooling_convention'] == 'full')

    net = nn.HybridSequential()
    net.add(
        nn.Conv2D(channels=classes,
                  kernel_size=args_conv['kernel'],
                  strides=args_conv['stride'],
                  padding=args_conv['pad'],
                  groups=args_conv['num_group'],
                  dilation=args_conv['dilate'],
                  layout=args_conv['layout'],
                  use_bias=bias,
                  in_channels=hybrid_block[0]._in_channels))  # pylint: disable=W0212
    net.add(nn.Activation(act))
    net.add(
        nn.AvgPool2D(pool_size=args_pool['kernel'],
                     strides=args_pool['stride'],
                     padding=args_pool['pad'],
                     ceil_mode=ceil_mode,
                     layout=args_pool['layout'],
                     count_include_pad=args_pool['count_include_pad']))
    net.add(nn.Flatten())
    net.initialize()

    mod.output = net
  else:
    last_l = mod.output
    bias = bool(last_l.bias)

    dense = nn.Dense(
        units=classes,
        activation=last_l.act,
        use_bias=bias,
        flatten=last_l._flatten,  # pylint: disable=W0212
        in_units=last_l._in_units)  # pylint: disable=W0212
    dense.initialize()
    mod.output = dense

  return mod
