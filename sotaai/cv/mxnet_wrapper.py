# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2020.
'''MxNet https://mxnet.apache.org/ wrapper module.'''

SOURCE_METADATA = {
    'name': 'mxnet',
    'original_name': 'MXNet',
    'url': 'https://mxnet.apache.org/'
}

DATASETS = {}

MODELS = {
    'classification': [
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
}


def load_model(name: str):
  return {'name': name, 'source': 'mxnet'}


def load_dataset(name: str):
  return {'train': {'name': name, 'source': 'mxnet'}}
