# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''Module used to interface with Torchvision's models and datasets.'''

from torchvision import models
from torchvision import datasets as dset
from re import search
import os
import numpy as np

DATASETS = {
    'classification': [
        # 'CelebA',
        # 'EMNIST',
        # 'KMNIST',
        # 'LSUN',  # No download
        'QMNIST',
        'SEMEION',
        'SVHN',
        'USPS',
        # 'STL10',  # Unsupervised learning.
        # 'ImageNet',  # No download
    ],
    'object_detection': [
        'CelebA',
        'Flickr30k',  # No download.
        'VOCDetection/2007',
        'VOCDetection/2008',
        # 'VOCDetection/2009', Corrupted
        'VOCDetection/2010',
        'VOCDetection/2011',
        'VOCDetection/2012',
    ],
    'segmentation': [
        'Cityscapes',  # No download.
        'VOCSegmentation/2007',
        'VOCSegmentation/2008',
        # 'VOCSegmentation/2009', Corrupted
        'VOCSegmentation/2010',
        'VOCSegmentation/2011',
        'VOCSegmentation/2012',
        'SBD/segmentation',
        'SBD/boundaries',
    ],
    'captioning': [
        'CocoCaptions',  # No download.
        'Flickr8k',  # No download.
        'Flickr30k',  # No download.
        'SBU'
    ],
    'human activity recognition': [
        'HMDB51',  # No download.
        'Kinetics400',  # No download.
        'UCF101',  # No download.
    ],
    'local image descriptors': [
        'PhotoTour/notredame',
        'PhotoTour/yosemite',
        'PhotoTour/liberty',
        'PhotoTour/notredame_harris',
        'PhotoTour/yosemite_harris',
        'PhotoTour/liberty_harris',
    ],
}

MODELS = {
    'classification': [
        'alexnet', 'densenet161', 'googlenet', 'mnasnet0_5', 'mnasnet0_75',
        'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet18', 'resnet34',
        'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5',
        'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
        'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
        'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'wide_resnet101_2',
        'wide_resnet50_2'
    ],
    'segmentation': [
        'deeplabv3_resnet101', 'deeplabv3_resnet50', 'fcn_resnet101',
        'fcn_resnet50'
    ],
    'object_detection': [
        'fasterrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn',
        'maskrcnn_resnet50_fpn'
    ],
    'video': ['mc3_18', 'r2plus1d_18', 'r3d_18']
}


def load_model(model_name, pretrained=False):
  '''
    Input:
        model_name: str, one from MODELS variable
        pretrained: bool, to load a pretrained or
                          randomly initialized model
    Output:
        torchvision.models
  '''

  # Load the corresponding model class
  if model_name in MODELS['segmentation']:
    trainer = getattr(models.segmentation, model_name)
  elif model_name in MODELS['object_detection']:
    trainer = getattr(models.detection, model_name)
  elif model_name in MODELS['video']:
    trainer = getattr(models.video, model_name)
  else:
    trainer = getattr(models, model_name)

  if model_name in ['googlenet', 'inception_v3']:
    model = trainer(pretrained=pretrained, init_weights=False)
  else:
    model = trainer(pretrained=pretrained)

  return model


def load_dataset(dataset_name,
                 root='default',
                 ann_file=None,
                 target_transform=None,
                 transform=None,
                 extensions=None,
                 frames_per_clip=None):
  '''
    Input:
        dataset_name: str, one from MODELS variable
        root: str, location of dataset
        ann_file: str. Path to json annotation file is necessary
                       if dataset_name is one of the following:
                         CocoCaptions,
                         CocoDetection,
                         Flickr8k,
                         Flickr30k,
                         HMDB51
                         UCF101
        frames_per_clip: Nbr of frames in a clip if dataset is a
                         video dataset
        transform: callable/optional, A function/transform that
                              takes in an PIL image and returns a
                              transformed version. E.g,
                              transforms.ToTensor
        extensions: (tuple[string]), A list of allowed
                extensions. both extensions
                and is_valid_file should not be passed
        target_transform:(callable, optional), A function/transform
                that takes in the target and transforms it.
    Output:
        dict, with keys indicating the partition of the dataset,
                and the values are of type DataLoader
  '''

  if root == 'default':
    root = '~/.torch/' + dataset_name

  if 'SBD' in dataset_name:
    mode = dataset_name.split('/')[1]
    dataset_name = 'SBDataset'

  elif 'PhotoTour' in dataset_name:
    name = dataset_name.split('/')[1]
    dataset_name = 'PhotoTour'
  elif 'VOC' in dataset_name:
    year = dataset_name.split('/')[1]
    dataset_name = dataset_name.split('/')[0]
  ds = getattr(dset, dataset_name)
  # Datasets saved in dic
  # with corresponding splits of dataset
  ds_dic = {}

  datasets_w_train = ['KMNIST', 'QMNIST', 'USPS']
  datasets_w_split = ['SVHN', 'CelebA']

  download_train = True  # os.path.exists(root+'/train')
  download_test = True  # os.path.exists(root+'/test')

  if dataset_name in datasets_w_train:
    ds_dic['train'] = ds(root + '/train',
                         train=True,
                         target_transform=target_transform,
                         transform=transform,
                         download=download_train)
    ds_dic['test'] = ds(root + '/test',
                        train=False,
                        target_transform=target_transform,
                        transform=transform,
                        download=download_test)

  elif dataset_name in datasets_w_split:
    ds_dic['train'] = ds(root + '/train',
                         split='train',
                         target_transform=target_transform,
                         transform=transform,
                         download=download_train)
    ds_dic['test'] = ds(root + '/test',
                        split='test',
                        target_transform=target_transform,
                        transform=transform,
                        download=download_test)
    if dataset_name == 'SVHN':
      ds_dic['extra_training_set'] = ds(root + '/extra',
                                        split='extra',
                                        target_transform=target_transform,
                                        transform=transform,
                                        download=True)
    elif dataset_name == 'CelebA':
      ds_dic['val'] = ds(root + '/val',
                         split='valid',
                         target_transform=target_transform,
                         transform=transform,
                         download=True)

  elif dataset_name == 'Cityscapes':
    ds_dic['train'] = ds(
        root + '/train',
        split='train',
        target_transform=target_transform,
        transform=transform,
    )
    ds_dic['test'] = ds(
        root + '/test',
        split='test',
        target_transform=target_transform,
        transform=transform,
    )
    ds_dic['val'] = ds(
        root + '/val',
        split='val',
        target_transform=target_transform,
        transform=transform,
    )

  elif 'PhotoTour' in dataset_name:

    ds_dic['train'] = ds(root + '/train',
                         name=name,
                         download=download_train,
                         transform=transform,
                         train=True)
    ds_dic['test'] = ds(root + '/test',
                        name=name,
                        transform=transform,
                        download=download_test,
                        train=False)

  elif dataset_name in ['SBU', 'SEMEION']:
    ds_dic['data'] = ds(root, transform=transform, download=True)

  elif dataset_name in ['VOCSegmentation', 'VOCDetection']:
    ds_dic['train'] = ds(root + '/train',
                         year=year,
                         transform=transform,
                         target_transform=target_transform,
                         image_set='train',
                         download=download_train)
    download_val = not os.path.exists(root + '/val')
    ds_dic['val'] = ds(root + '/val',
                       year=year,
                       image_set='val',
                       download=download_val)

  elif 'SBD' in dataset_name:
    download_train = not os.path.exists(root + mode + '/train')
    download_val = not os.path.exists(root + mode + '/val')
    ds_dic['train'] = ds(root + mode + '/train',
                         image_set='train',
                         mode=mode,
                         target_transform=target_transform,
                         transform=transform,
                         download=download_train)

    ds_dic['val'] = ds(root + mode + '/val',
                       image_set='val',
                       mode=mode,
                       download=download_val)

  elif dataset_name == 'LSUN':
    ds_dic['train'] = dset.LSUN(root,
                                target_transform=target_transform,
                                transform=transform,
                                classes='train')
    ds_dic['val'] = dset.LSUN(root, classes='val')
    ds_dic['test'] = dset.LSUN(root,
                               target_transform=target_transform,
                               transform=transform,
                               classes='test')

  elif dataset_name in [
      'CocoDetection', 'CocoCaptions', 'Flickr8k', 'Flickr30k'
  ]:
    ds_dic['data'] = ds(root, ann_file)

  elif dataset_name in ['HMDB51', 'UCF101']:
    ds_dic['train'] = ds(root + 'train',
                         ann_file,
                         frames_per_clip,
                         transform=transform,
                         train=True)
    ds_dic['test'] = ds(root + 'test',
                        ann_file,
                        frames_per_clip,
                        transform=transform,
                        train=False)

  elif dataset_name == 'Kinetics400':
    ds_dic['data'] = ds(root, frames_per_clip, extensions=extensions)

  # ds_dic['test'] = iter(ds_dic['test']) if ds_dic['test'] else ds_dic['test']
  # ds_dic['train'] = iter(
  #     ds_dic['train']) if ds_dic['train'] else ds_dic['train']

  # if 'test' in ds_dic:
  #   ds_dic['test'] = iter(ds_dic['test'])
  # if 'train' in ds_dic:
  #   ds_dic['train'] = iter(ds_dic['train'])
  return ds_dic


class DatasetIterator():
  '''Torch dataset iterator class'''

  def __init__(self, raw) -> None:
    self._raw = raw
    self._iterator = self.create_iterator()

  def __next__(self):
    '''Get the next item from the dataset in a standardized format.

    Returns:
    '''
    if search('DataLoader', str(type(self._raw))):
      print(self._iterator)
      image = np.array(next(self._iterator)[0])
      label = next(self._iterator)[1].numpy() if search(
          'Tensor', str(type(next(self._iterator)[1]))) else next(
              self._iterator)[1]
      # image, label = self._iterator.next()
      return {'image': image[0], 'label': label[0]}
    else:
      image = np.array(next(self._iterator, [0, 0])[0])
      label = next(self._iterator, [0, 0])[1]

      return {'image': image, 'label': label}

  def create_iterator(self):
    '''Create an iterator out of the raw dataset split object

    Returns:
      An object containing iterators for the dataset images and labels
    '''
    return iter(self._raw)
