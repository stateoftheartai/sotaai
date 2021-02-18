# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
Module to store metadata of CV models, datasets, and so forth
'''

DATASETS = {
    'mnist': {
        'name': 'mnist',
        'source': 'keras',
        'train_size': 60000,
        'test_size': 10000,
        'image': (28, 28),
        'label': ()
    },
    'cifar10': {
        'name': 'cifar10',
        'source': 'keras',
        'train_size': 60000,
        'test_size': 10000,
        'image': (32, 32, 3),
        'label': (1,)
    },
    'cifar100': {
        'name': 'cifar100',
        'source': 'keras',
        'train_size': 60000,
        'test_size': 10000,
        'image': (32, 32, 3),
        'label': (1,)
    },
    'fashion_mnist': {
        'name': 'fashion_mnist',
        'source': 'keras',
        'train_size': 60000,
        'test_size': 10000,
        'image': (28, 28),
        'label': ()
    },
    'beans': {
        'name': 'beans',
        'source': 'tensorflow',
        'train_size': 60000,
        'image': (500, 500, 3),
        'label': ()
    },
    'omniglot': {
        'name': 'omniglot',
        'source': 'tensorflow',
        'train_size': 60000,
        'image': (105, 105, 3),
        'label': ()
    }
}
