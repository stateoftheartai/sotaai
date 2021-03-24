# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''Module used to interface with Torchvision's models and datasets.'''

from sotaai.cv import utils
from torchvision import models
from torchvision import datasets as dset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from re import search
# from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

SOURCE_METADATA = {
    'name': 'torch',
    'original_name': 'Torchvision',
    'url': 'http://pytorch.org/vision/stable/index.html'
}

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
        # TODO(Jorge)
        # Finish object_detection implementation
        # 'Flickr30k',  # No download.
        'VOCDetection/2007',
        'VOCDetection/2008',
        # 'VOCDetection/2009', Corrupted
        'VOCDetection/2010',
        'VOCDetection/2011',
        'VOCDetection/2012',
        'CelebA',
    ],
    'segmentation': [
        # 'Cityscapes',  # No download.
        'VOCSegmentation/2007',
        'VOCSegmentation/2008',
        # 'VOCSegmentation/2009', Corrupted
        'VOCSegmentation/2010',
        'VOCSegmentation/2011',
        'VOCSegmentation/2012',
        'SBD/segmentation',
        'SBD/boundaries',
    ],
    # TODO(team)
    # Eventually implement the remaining tasks...
    'captioning': [
        # 'CocoCaptions',  # No download.
        # 'Flickr8k',  # No download.
        # 'Flickr30k',  # No download.
        # 'SBU'
    ],
    'human activity recognition': [
        # 'HMDB51',  # No download.
        # 'Kinetics400',  # No download.
        # 'UCF101',  # No download.
    ],
    'local image descriptors': [
        # 'PhotoTour/notredame',
        # 'PhotoTour/yosemite',
        # 'PhotoTour/liberty',
        # 'PhotoTour/notredame_harris',
        # 'PhotoTour/yosemite_harris',
        # 'PhotoTour/liberty_harris',
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
    'video': [
        # 'mc3_18',
        # 'r2plus1d_18',
        # 'r3d_18'
    ]
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
                 frames_per_clip=None,
                 download=True):
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
        download: temporal flag to skip download and only create the dataset
          instance with no data (used for JSONs creation)
    Output:
        dict, with keys indicating the partition of the dataset,
                and the values are of type DataLoader
  '''
  if not download:
    return {'train': {'name': dataset_name, 'source': 'tensorflow'}}

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
    self._image_preprocessing_callback = None

  def __next__(self):
    '''Get the next item from the dataset in a standardized format.

    Returns:
    '''
    if 'VOCDetection' in str(type(self._raw)):
      image, target = next(self._iterator)
      element = [image, target]
      image, target = self._image_preprocessing_callback(element)
      return image, target
    else:
      datapoint = next(self._iterator)
      if search('DataLoader', str(type(self._raw))):
        image = np.array(datapoint[0])
        label = datapoint[1].numpy() if search('Tensor', str(type(
            datapoint[1]))) else datapoint[1]
        img = image[0]
        if self._image_preprocessing_callback:
          img = self._image_preprocessing_callback(image[0])
        # image, label = self._iterator.next()
        return {'image': img, 'label': label[0]}
      else:
        image = np.array(datapoint[0])
        label = datapoint[1]

        if self._image_preprocessing_callback:
          image = self._image_preprocessing_callback(image)

        return {'image': image, 'label': label}

  def create_iterator(self):
    '''Create an iterator out of the raw dataset split object

    Returns:
      An object containing iterators for the dataset images and labels
    '''
    return iter(self._raw)

  def set_image_preprocessing(self, image_preprocessing_callback):
    self._image_preprocessing_callback = image_preprocessing_callback


def model_to_dataset_classification(cv_model, cv_dataset):
  '''If compatible, adjust model and dataset so that they can be executed
  against each other

  Args:
    cv_model: an abstracted cv model whose source is Torch of classification
    cv_dataset: an abstracted cv dataset of classification

  Returns:
    cv_model: the abstracted cv model adjusted to be executed against
      cv_dataset
    cv_dataset: the abstracted cv dataset adjust to be executed against
      cv_model
  '''

  raw_model = cv_model.raw
  are_channels_compatible = len(cv_dataset.shape) == len(
      cv_model.original_input_shape)

  model_input = cv_model.original_input_shape

  model_input_channels = model_input[1]

  # batch = model_input[0]
  batch = 2

  size = cv_dataset.shape[1]

  min_size = 64
  if cv_model.name in utils.IMAGE_MINS:
    min_size = utils.IMAGE_MINS[cv_model.name]

  if not isinstance(size, int):
    if size is None:
      size = min_size
    else:
      size = size.item()
  if size < min_size:  #minimum image size for torch models (64x64)
    size = min_size

  # Case 1:
  # Reshape dataset channels according with models input channels

  if not are_channels_compatible:

    def preprocess_image(image):
      preprocess = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Resize((size, size))])
      input_tensor = preprocess(image)
      standarized_element = input_tensor.unsqueeze(0)
      w = standarized_element.shape[2]
      h = standarized_element.shape[3]
      standarized_element = standarized_element.expand(batch,
                                                       model_input_channels, w,
                                                       h)
      return standarized_element

    cv_dataset.set_image_preprocessing(preprocess_image)
    cv_dataset.shape = cv_model.original_input_shape

  is_output_compatible = utils.compare_shapes(cv_model.original_output_shape,
                                              cv_dataset.classes_shape)

  # Case 2:
  # If output is not compatible with dataset classes, we have to change the
  # model output layer
  if not is_output_compatible:
    classes = cv_dataset.classes_shape
    num_classes = classes[0]
    modules = cv_model.raw.__dict__
    attributes = list(modules['_modules'])
    last_layer = getattr(cv_model.raw, attributes[-1])
    if hasattr(last_layer, '__getitem__'):
      group_layer = last_layer

      if hasattr(group_layer[-1], 'out_features'):
        in_features = group_layer[-1].in_features
        group_layer[-1] = nn.Linear(in_features, num_classes, True)
      else:
        in_channels = group_layer[1].in_channels
        kernel_size = group_layer[1].kernel_size
        stride = group_layer[1].stride

        group_layer[1] = nn.Conv2d(in_channels, num_classes, kernel_size,
                                   stride)

      setattr(raw_model, attributes[-1], group_layer)
    else:
      in_features = last_layer.in_features
      last_layer = nn.Linear(in_features, num_classes, True)

      setattr(raw_model, attributes[-1], last_layer)

  cv_model.update_raw_model(raw_model)
  return cv_model, cv_dataset


def model_to_dataset_segmentation(cv_model, cv_dataset):
  '''If compatible, adjust model and dataset so that they can be executed
  against each other

  Args:
    cv_model: an abstracted cv model whose source is Segmentation Torch
    cv_dataset: an abstracted segmentation cv dataset

  Returns:
    cv_model: the abstracted cv model adjusted to be executed against
      cv_dataset
    cv_dataset: the abstracted cv dataset adjust to be executed against
      cv_model
  '''

  print('\nModel ', cv_model.name)
  print(' Input: ', cv_model.original_input_shape)
  print(' Output: ', cv_model.original_output_shape)
  print('Dataset: ', cv_dataset.name)
  print(' Shape:   ', cv_dataset.shape)
  print(' Pixel Classes: ', len(cv_dataset.pixel_classes))

  print('\nAdjusting...')

  raw_model = cv_model.raw

  def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    standarized_element = preprocess(image)
    return standarized_element

  cv_dataset.set_image_preprocessing(preprocess_image)

  # Adjust the model output
  num_pixels = len(cv_dataset.pixel_classes)

  in_channels = raw_model.classifier[-1].in_channels
  kernel = raw_model.classifier[-1].kernel_size
  stride = raw_model.classifier[-1].stride

  raw_model.classifier[-1] = nn.Conv2d(in_channels, num_pixels, kernel, stride)

  cv_model.update_raw_model(raw_model)

  return cv_model, cv_dataset


def model_to_dataset_object_detection(cv_model, cv_dataset):
  '''If compatible, adjust model and dataset so that they can be executed
  against each other

  Args:
    cv_model: an abstracted cv model whose source is Segmentation Torch
    cv_dataset: an abstracted segmentation cv dataset

  Returns:
    cv_model: the abstracted cv model adjusted to be executed against
      cv_dataset
    cv_dataset: the abstracted cv dataset adjust to be executed against
      cv_model
  '''
  source = cv_dataset.source
  raw_model = cv_model.raw

  classes_labels = (
      '__background__ ',
      'aeroplane',
      'bicycle',
      'bird',
      'boat',
      'bottle',
      'bus',
      'car',
      'cat',
      'chair',
      'cow',
      'diningtable',
      'dog',
      'horse',
      'motorbike',
      'person',
      'pottedplant',
      'sheep',
      'sofa',
      'train',
      'tvmonitor',
  )

  def preprocess_image(datapoint):
    # image, target
    # preprocess = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Resize(300)])
    preprocess = transforms.Compose([transforms.ToTensor()])

    if source == 'tensorflow':
      image = datapoint['image']
      bbox = datapoint['torsobox']
      image = preprocess(image)
      target = {}
      boxes = []
      ymin = bbox[1]
      ymax = bbox[3]
      xmin = bbox[0]
      xmax = bbox[2]
      boxes.append([xmin, ymin, xmax, ymax])
      boxes = torch.as_tensor(boxes, dtype=torch.float32)

      # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

      target['boxes'] = boxes
      # target['area'] = area

      return image, target
    else:

      image = datapoint[0]
      target = datapoint[1]

      anno = target['annotation']
      boxes = []
      classes = []
      area = []
      iscrowd = []
      objects = anno['object']

      if not isinstance(objects, list):
        objects = [objects]
      for obj in objects:
        bbox = obj['bndbox']
        bbox = [
            int(float(bbox[n])) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']
        ]
        boxes.append(bbox)
        classes.append(classes_labels.index(obj['name']))
        iscrowd.append(int(float(obj['difficult'])))
        area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      classes = torch.as_tensor(classes)
      area = torch.as_tensor(area)
      iscrowd = torch.as_tensor(iscrowd)

      image_id = anno['filename'][5:-4]
      image_id = torch.as_tensor([int(image_id)])

      target = {}

      target['boxes'] = boxes
      target['labels'] = classes
      target['image_id'] = image_id

      # for conversion to coco api
      target['area'] = area
      target['iscrowd'] = iscrowd

      image = preprocess(image)  # reshape bounding boxex
      # w = image.shape[0]
      # h = image.shape[1]
      # image = image.expand(1, w, h)
      return image, target

  cv_dataset.set_image_preprocessing(preprocess_image)

  in_features = raw_model.roi_heads.box_predictor.cls_score.in_features
  raw_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                        len(classes_labels))

  cv_model.update_raw_model(raw_model)

  return cv_model, cv_dataset
