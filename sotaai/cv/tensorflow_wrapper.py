# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
'''Module used to interface with Tensorflow's datasets.'''
import tensorflow_datasets as tfds
import resource
import os
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Prevent Tensorflow to print warning and meta logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASETS = {
    'video': [
        'bair_robot_pushing_small',
        'moving_mnist',
        'starcraft_video',
        # 'ucf101'  # Bug tensorflow
    ],
    'object detection': [
        'celeb_a_hq',  # manual download
        'coco',
        'flic',
        'kitti',
        # 'open_images_challenge2019_detection', Apache beam
        # 'open_images_v4', Apache beam
        'voc',
        'the300w_lp'
        # 'wider_face' # Wrong checksum
    ],
    'classification': [
        'beans',
        'binary_alpha_digits',
        # 'caltech101',  # Wrong checksum
        'caltech_birds2010',
        'caltech_birds2011',
        'cars196',
        'cats_vs_dogs',
        'celeb_a',
        'celeb_a_hq',  # manual download
        'cifar10_1',
        'cifar10_corrupted',
        # 'citrus_leaves',  # Wrong checksum
        'cmaterdb',
        'colorectal_histology',
        'colorectal_histology_large',
        'curated_breast_imaging_ddsm',  # manual download
        'cycle_gan',
        'deep_weeds',  # manual download
        'diabetic_retinopathy_detection',
        'downsampled_imagenet',
        'dtd',
        'emnist',
        'eurosat',
        'food101',
        'geirhos_conflict_stimuli',
        'horses_or_humans',
        'i_naturalist2017',
        'imagenet2012',  # manual download
        'imagenet2012_corrupted',  # manual download
        'imagenet2012_subset',  # manual download
        'imagenet_resized',
        'imagenette',
        'imagewang',
        'kmnist',
        'lfw',
        'malaria',
        'mnist_corrupted',
        'omniglot',
        'oxford_flowers102',
        'oxford_iiit_pet',
        'patch_camelyon',
        'places365_small',
        'quickdraw_bitmap',
        'resisc45',  # manual download
        'rock_paper_scissors',
        'smallnorb',
        'so2sat',
        'stanford_dogs',
        'stanford_online_products',
        'stl10',
        'sun397',
        'svhn_cropped',
        'tf_flowers',
        'uc_merced',
        'vgg_face2',  # manual download
        'visual_domain_decathlon'
    ],
    'segmentation': [
        'cityscapes',  # manual download
        'lost_and_found',
        'scene_parse150'
    ],
    'image super resolution': ['div2k',],
    'key point detection': ['aflw2k3d', 'celeb_a', 'the300w_lp'],
    'pose estimation': ['flic', 'the300w_lp'],
    'face alignment': ['the300w_lp'],
    'visual reasoning': ['clevr'],
    'visual question answering': ['clevr'],
    'image generation': ['dsprites', 'shapes3d'],
    '3d image generation': ['shapes3d',],
    'other': [
        'binarized_mnist',
        'chexpert',  # manual download
        'coil100',
        'lsun'
    ]
}


def load_dataset(dataset_name):
  '''Return a tensorflow dataset in its iterable version

  Args:
    dataset_name: the dataset name in string

  Returns:
    A dict where each key is a dataset split and the value is a dataset
    in its iterable numpy version (IterableDataset). Each item in the iterator
    has the 'image' and 'label' keys which are in turn numpy arrays of the image
    and label respectively.
  '''
  ds = tfds.load(dataset_name)
  return ds
  # return tfds.as_numpy(ds)
  # ds_dic = {}
  # for split in ds:
  # for item in ds[split]:
  # print(type(item['image']), item['image'].shape, item['label'].shape)
  # break
  # return ds_dic


def get_dataset_item():
  '''Return a single datapoint or item

    Args:
      raw: raw tensorflow dataset object
      i (int): index to get item

    Returns:
      A dict. The dict will contain a 'data' key which will hold the
      datapoint as a numpy array. The dict will also contain a 'label' key which
      will hold the label of the datapoint. The dict might contain other keys
      depending on the nature of the dataset.
  '''
  return {'image': None, 'label': None}
