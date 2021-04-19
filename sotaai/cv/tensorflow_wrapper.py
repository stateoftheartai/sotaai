# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
'''Module used to interface with Tensorflow's datasets.'''
import numpy as np
import tensorflow_datasets as tfds
import resource
import os

# As per reported in https://github.com/tensorflow/datasets/issues/1441 the
# minimum number of open files required by the TF shuffler to work is 1000, this
# has to be manually set for some environments. Using (high, high) might work
# for some environments, but not for all of them.
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1000, high))

# Prevent Tensorflow to print warning and meta logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SOURCE_METADATA = {
    'name': 'tensorflow',
    'original_name': 'TensorFlow Datasets',
    'url': 'https://www.tensorflow.org/datasets/'
}

DATASETS = {
    'classification': [
        'beans',
        'binary_alpha_digits',
        # 'caltech101',  # Wrong checksum
        'caltech_birds2010',
        'caltech_birds2011',
        'cars196',
        'cats_vs_dogs',
        # 'celeb_a',  # manual download
        # 'celeb_a_hq',  # manual download
        'cifar10_1',
        'cifar10_corrupted',
        # 'citrus_leaves',  # Wrong checksum
        'cmaterdb',
        'colorectal_histology',
        'colorectal_histology_large',
        # 'curated_breast_imaging_ddsm',  # manual download
        'cycle_gan',
        # 'deep_weeds',  # manual download
        # 'diabetic_retinopathy_detection', # manual download
        # 'downsampled_imagenet', # Error link source
        'dtd',
        'emnist',
        'eurosat',
        'food101',
        'geirhos_conflict_stimuli',
        'horses_or_humans',
        'i_naturalist2017',
        # 'imagenet2012',  # manual download
        # 'imagenet2012_corrupted',  # manual download
        # 'imagenet2012_subset',  # manual download
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
        # 'resisc45',  # manual download
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
        # 'vgg_face2',  # manual download
        'visual_domain_decathlon'
    ],
    'segmentation': [
        # 'cityscapes',  # manual download
        'lost_and_found',
        # 'scene_parse150' #error torch soruce download
    ],
    'object_detection': [
        # 'celeb_a_hq',  # manual download
        'coco',  # tested
        'flic',  # tested
        'kitti',
        'open_images_challenge2019_detection',  # Apache beam
        'open_images_v4',  # Apache beam
        'voc',
        'the300w_lp',
        'wider_face'  # Wrong checksum
    ],
    # TODO(team)
    # Eventually implement the remaining tasks...
    'video': [
        # 'bair_robot_pushing_small',
        # 'moving_mnist',
        # 'starcraft_video',
        # 'ucf101'  # Bug tensorflow
    ],
    'image_super_resolution': ['div2k',],
    'keypoint_detection': ['aflw2k3d', 'celeb_a', 'the300w_lp'],
    'pose_estimation': ['flic', 'the300w_lp'],
    'face_alignment': ['the300w_lp'],
    'visual_reasoning': ['clevr'],
    'visual_question_answering': ['clevr'],
    'image_generation': ['dsprites', 'shapes3d'],
    '3d_image_generation': ['shapes3d',],
    'other': [
        # 'binarized_mnist',
        # 'chexpert',  # manual download
        # 'coil100',
        # 'lsun'
    ]
}

MODELS = {}


def load_dataset(dataset_name, download=True):
  '''Return a tensorflow dataset in its iterable version

  Args:
    dataset_name: the dataset name in string
    download: temporal flag to skip download and only create the dataset
      instance with no data (used for JSONs creation)

  Returns:
    A dict where each key is a dataset split and the value is a dataset
    in its iterable numpy version (IterableDataset). Each item in the iterator
    has the 'image' and 'label' keys which are in turn numpy arrays of the image
    and label respectively.
  '''
  if download:
    ds = tfds.load(dataset_name)
    return tfds.as_numpy(ds)
  else:
    return {'train': {'name': dataset_name, 'source': 'tensorflow'}}


class DatasetIterator():
  '''Tensorflow dataset iterator class'''

  def __init__(self, raw) -> None:
    self._raw = raw
    self._iterator = self.create_iterator()
    self._image_preprocessing_callback = None
    self._keypoint_preprocessing_cb = None

  def __next__(self):
    '''Get the next item from the dataset in a standardized format.

    Returns:
      A dict with two mandatory keys: the 'image' key which will hold the image
      as a numpy array, and the 'label' key which will hold the label as a numpy
      array as well. The dict might contain other keys depending on the nature
      of the dataset.
    '''
    item = next(self._iterator)

    image = None
    # For Classification
    if 'image' in item:
      image = item['image']
    # For Segmentation
    elif 'image_left' in item:
      image = item['image_left']

    if self._image_preprocessing_callback:
      # For Object Detection
      if 'torsobox' in item:
        image, target = self._image_preprocessing_callback(item)

        return image, target
      else:
        image = self._image_preprocessing_callback(image)

    std_item = {'image': image}

    #For Keypoint Detection

    if 'landmarks_68_3d_xy_normalized' in item:
      std_item['keypoints'] = item['landmarks_68_3d_xy_normalized']

    if 'landmarks_2d' in item:
      std_item['keypoints'] = item['landmarks_2d']

    if 'landmarks' in item:
      landmarks = item['landmarks']
      keypoints = [
          [landmarks['lefteye_x'], landmarks['lefteye_y']],
          [landmarks['leftmouth_x'], landmarks['leftmouth_y']],
          [landmarks['nose_x'], landmarks['nose_y']],
          [landmarks['righteye_x'], landmarks['righteye_y']],
          [landmarks['rightmouth_x'], landmarks['rightmouth_y']],
      ]
      std_item['keypoints'] = np.array(keypoints)
    #Check if there is a data preprocessing.
    # This is set in model_to_dataset methods
    if self._keypoint_preprocessing_cb:
      image, target = self._keypoint_preprocessing_cb(std_item)
      return image, target

    if 'label' in item:
      std_item['label'] = item['label']
    elif 'segmentation_label' in item:
      std_item['label'] = item['segmentation_label']

    return std_item

  def create_iterator(self):
    '''Create an iterator out of the raw dataset split object. This is the
    Tensorflow iterator being wrapped in our own iterator.

    Returns:
      An object containing iterators for the dataset images and labels
    '''
    return iter(self._raw)

  def set_image_preprocessing(self, image_preprocessing_callback):
    self._image_preprocessing_callback = image_preprocessing_callback

  def set_keypoint_preprocessing(self, keypoint_preprocessing_cb):
    ''' Set keypoint detection callback. Call in every iterator item. 
    '''
    self._keypoint_preprocessing_cb = keypoint_preprocessing_cb
