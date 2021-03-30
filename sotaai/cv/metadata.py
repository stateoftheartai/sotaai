# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Module to store metadata of CV models, datasets, and so forth
'''

DATASETS = {
    'mnist': {
        'name': 'mnist',
        'sources': ['keras'],
        'metadata': {
            'train_size': 60000,
            'test_size': 10000,
            'test_classes': range(0, 10),
            'train_classes': range(0, 10),
            'classes_names': None,
            'image': (28, 28),
            'label': ()
        }
    },
    'cifar10': {
        'name': 'cifar10',
        'sources': ['keras'],
        'metadata': {
            'train_size': 50000,
            'test_size': 10000,
            'test_classes': range(0, 10),
            'train_classes': range(0, 10),
            'classes_names': None,
            'image': (32, 32, 3),
            'label': (1,)
        }
    },
    'cifar100': {
        'name': 'cifar100',
        'sources': ['keras'],
        'metadata': {
            'train_size': 50000,
            'test_size': 10000,
            'test_classes': range(0, 100),
            'train_classes': range(0, 100),
            'classes_names': None,
            'image': (32, 32, 3),
            'label': (1,)
        }
    },
    'fashion_mnist': {
        'name': 'fashion_mnist',
        'sources': ['keras'],
        'metadata': {
            'train_size': 60000,
            'test_size': 10000,
            'test_classes': range(0, 10),
            'train_classes': range(0, 10),
            'classes_names': None,
            'image': (28, 28),
            'label': ()
        }
    },
    'beans': {
        'name': 'beans',
        'sources': ['tensorflow'],
        'metadata': {
            'image': (500, 500, 3),
            'label': ()
        }
    },
    'omniglot': {
        'name': 'omniglot',
        'sources': ['tensorflow'],
        'metadata': {
            'image': (105, 105, 3),
            'label': ()
        }
    },
    'wider_face': {
        'name': 'wider_face',
        'sources': ['tensorflow'],
        'metadata': {
            'image': (None, None, 3)
        }
    }
}

MODELS = {
    'InceptionResNetV2': {
        'name': 'InceptionResNetV2',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 245,
            'input_type': 'numpy.ndarray',
            'num_parameters': 55873736
        }
    },
    'InceptionV3': {
        'name': 'InceptionV3',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 95,
            'input_type': 'numpy.ndarray',
            'num_parameters': 23851784
        }
    },
    'ResNet101V2': {
        'name': 'ResNet101V2',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 105,
            'input_type': 'numpy.ndarray',
            'num_parameters': 44675560
        }
    },
    'ResNet152V2': {
        'name': 'ResNet152V2',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 156,
            'input_type': 'numpy.ndarray',
            'num_parameters': 60380648
        }
    },
    'ResNet50V2': {
        'name': 'ResNet50V2',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 54,
            'input_type': 'numpy.ndarray',
            'num_parameters': 25613800
        }
    },
    'VGG16': {
        'name': 'VGG16',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 16,
            'input_type': 'numpy.ndarray',
            'num_parameters': 138357544
        }
    },
    'VGG19': {
        'name': 'VGG19',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 19,
            'input_type': 'numpy.ndarray',
            'num_parameters': 143667240
        }
    },
    'Xception': {
        'name': 'Xception',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 41,
            'input_type': 'numpy.ndarray',
            'num_parameters': 3357584
        }
    },
    'ResNet50': {
        'name': 'ResNet50',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 53,
            'input_type': 'numpy.ndarray',
            'num_parameters': 23587712
        }
    },
    'ResNet101': {
        'name': 'ResNet101',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 104,
            'input_type': 'numpy.ndarray',
            'num_parameters': 42658176
        }
    },
    'ResNet152': {
        'name': 'ResNet152',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 155,
            'input_type': 'numpy.ndarray',
            'num_parameters': 58370944
        }
    },
    'DenseNet121': {
        'name': 'DenseNet121',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 120,
            'input_type': 'numpy.ndarray',
            'num_parameters': 7037504
        }
    },
    'DenseNet169': {
        'name': 'DenseNet169',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 168,
            'input_type': 'numpy.ndarray',
            'num_parameters': 12642880
        }
    },
    'DenseNet201': {
        'name': 'DenseNet201',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 200,
            'input_type': 'numpy.ndarray',
            'num_parameters': 18321984
        }
    },
    'NASNetMobile': {
        'name': 'NASNetMobile',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 196,
            'input_type': 'numpy.ndarray',
            'num_parameters': 2249533
        }
    },
    'NASNetLarge': {
        'name': 'NASNetLarge',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 268,
            'input_type': 'numpy.ndarray',
            'num_parameters': 43614774
        }
    },
    'MobileNet': {
        'name': 'MobileNet',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 28,
            'input_type': 'numpy.ndarray',
            'num_parameters': 4253864
        }
    },
    'MobileNetV2': {
        'name': 'MobileNetV2',
        'sources': ['keras'],
        'metadata': {
            'num_layers': 53,
            'input_type': 'numpy.ndarray',
            'num_parameters': 3538984
        }
    }
}

UNIFIED_MODELS = {
    'densenet': {
        'name': 'DenseNet: Dense Convolutional Network',
        'paper_name': 'Densely Connected Convolutional Networks',
        'paper_link': 'https://arxiv.org/abs/1608.06993',
    },
    'deeplabv3': {
        'name':
            'DeepLabv3 ResNet',
        'paper_name':
            'Rethinking Atrous Convolution for Semantic Image Segmentation',
        'paper_link':
            'https://arxiv.org/abs/1706.05587',
    },
    'dpn': {
        'name': 'Dual Path Networks',
        'paper_name': 'Dual Path Networks',
        'paper_link': 'https://arxiv.org/abs/1707.01629',
    },
    'fcn_resnet': {
        'name': 'FCN ResNet: Fully Convolutional Network ResNet',
        'paper_name': 'Fully Convolutional Networks for Semantic Segmentation',
        'paper_link': 'https://arxiv.org/abs/1605.06211',
    },
    'mnasnet': {
        'name':
            'MnasNet',
        'paper_name':
            'MnasNet: Platform-Aware Neural Architecture Search for Mobile',
        'paper_link':
            'https://arxiv.org/abs/1807.11626',
    },
    'mobilenet': {
        'name': 'MobileNets',
        'paper_name':
            'MobileNets: Efficient Convolutional Neural Networks for Mobile'
            ' Vision Applications',
        'paper_link': 'https://arxiv.org/abs/1704.04861',
    },
    'nasnet': {
        'name': 'NASNet: Neuron Attention Stage-by-Stage Net',
        'paper_name':
            'NASNet: A Neuron Attention Stage-by-Stage Net for Single Image'
            ' Deraining',
        'paper_link': 'https://arxiv.org/abs/1912.03151',
    },
    'resnet': {
        'name': 'ResNet: Residual Networks',
        'paper_name': 'Deep Residual Learning for Image Recognition',
        'paper_link': 'https://arxiv.org/abs/1512.03385',
    },
    'resnext': {
        'name':
            'ResNeXt',
        'paper_name':
            'Aggregated Residual Transformations for Deep Neural Networks',
        'paper_link':
            'https://arxiv.org/abs/1611.05431',
    },
    'se_resnet': {
        'name': 'SE-ResNet: Squeeze-and-Excitation ResNet',
        'paper_name': 'Squeeze-and-Excitation Networks',
        'paper_link': 'https://arxiv.org/abs/1709.01507',
    },
    'se_resnext': {
        'name': 'SE-ResNet: Squeeze-and-Excitation ResNeXt',
        'paper_name': 'Squeeze-and-Excitation Networks',
        'paper_link': 'https://arxiv.org/abs/1709.01507',
    },
    'senet': {
        'name': 'SENet: Squeeze-and-Excitation Networks',
        'paper_name': 'Squeeze-and-Excitation Networks',
        'paper_link': 'https://arxiv.org/abs/1709.01507',
    },
    'shufflenet': {
        'name': 'ShuffleNet',
        'paper_name':
            'ShuffleNet: An Extremely Efficient Convolutional Neural Network'
            'for Mobile Devices',
        'paper_link': 'https://arxiv.org/abs/1707.01083',
    },
    'squeezenet': {
        'name': 'SqueezeNet',
        'paper_name':
            'SqueezeNet: AlexNet-level accuracy with 50x fewer parameters'
            'and <0.5MB model size',
        'paper_link': 'https://arxiv.org/abs/1602.07360',
    },
    'vgg': {
        'name': 'VGG',
        'paper_name': 'Very Deep Convolutional Networks for Large-Scale Image'
                      ' Recognition',
        'paper_link': 'https://arxiv.org/abs/1409.1556',
    },
    'xresnet': {
        'name': 'X-ResNet',
        'paper_name': '',
        'paper_link': '',
    },
}

UNIFIED_DATASETS = {}

IGNORE_THESE_DATASETS = [
    'i_naturalist2017', 'CelebA', 'lfw', 'SBD/segmentation'
]


def get_unified_name(item_type=None, name=None):
  '''Return the unified name to be displayed in front end.

  Args:
    item_type: whether 'models' or 'datasets'
    name: a valid dataset or model name e.g. 'mnist'.

  Returns:
    String with the unified name.
  '''
  items = UNIFIED_MODELS if item_type == 'models' else UNIFIED_DATASETS
  unified_keys = [key for key, val in items.items() if key in name.lower()]
  # Check that both strings start with the same pattern to avoid mismatches.
  for k in unified_keys:
    substr_len = min(len(k), len(name))
    if name.lower()[:substr_len] == k[:substr_len]:
      return UNIFIED_MODELS[k]['name']
  # If we are here, it means that no unification matches were found, so we
  # just simply return the intial name.
  return name


def get(item_type=None, name=None, source=None):
  '''Return datasets/models metadata given source or name

  Args:
    item_type: whether 'models' or 'datasets'
    name: a valid dataset or model name e.g. 'mnist', if given source is ignored
    source: a valid source name e.g. 'keras'. If name given this attribute is
    ignored.

  Returns:
    If name given, return the metadata object for the given dataset or model. If
    source given, return an array of metadatas for the datasets or models that
    matched the given source.
  '''
  if name is not None:
    items = DATASETS if item_type == 'datasets' else MODELS
    return items[name]

  items = DATASETS.values() if item_type == 'datasets' else MODELS.values()
  return filter(lambda item: source in item['sources'], items)
