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
