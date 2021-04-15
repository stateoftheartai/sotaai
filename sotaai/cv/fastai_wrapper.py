# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
'''
fastai https://www.fast.ai/ wrapper module
'''

# TODO(Hugo)
# Finish this implementation, it was temporary commented out to allow JSON
# creation for wrapper
# from fastai.vision import models, URLs, ImageList, untar_data

# TODO(team)
# Fully implement this wrapper

SOURCE_METADATA = {
    'name': 'fastai',
    'original_name': 'fast.ai',
    'url': 'https://fastai1.fast.ai/index.html'
}

MODELS = {
    'classification': [
        'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
        'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11_bn', 'vgg13_bn',
        'vgg16_bn', 'vgg19_bn', 'xresnet101', 'xresnet152', 'xresnet18',
        'xresnet18_deep', 'xresnet34', 'xresnet34_deep', 'xresnet50',
        'xresnet50_deep'
    ]
}

DATASETS = {
    'classification': [
        'CALTECH_101', 'DOGS', 'IMAGENETTE', 'IMAGENETTE_160', 'IMAGENETTE_320',
        'IMAGEWOOF', 'IMAGEWOOF_160', 'IMAGEWOOF_320', 'MNIST_SAMPLE',
        'MNIST_TINY', 'MNIST_VAR_SIZE_TINY', 'PETS', 'SKIN_LESION'
    ],
    'key_point_detection': ['BIWI_SAMPLE'],
    'object_detection': ['COCO_SAMPLE', 'COCO_TINY'],
    'multi_label_classification': ['PLANET_SAMPLE', 'PLANET_TINY'],
    'segmentation': ['CAMVID', 'CAMVID_TINY']
}


def load_model(name: str):
  return {'name': name, 'source': 'mxnet'}

  # TODO(Hugo)
  # Finish this implementation, it was temporary commented out to allow JSON
  # creation for these models
  # model = getattr(models, name)(pretrained=pretrained)
  # return model


def load_dataset(name: str):
  return {'train': {'name': name, 'source': 'mxnet'}}

  # TODO(Hugo)
  # Finish this implementation, it was temporary commented out to allow JSON
  # creation for these datasets
  # ds_url = getattr(URLs, name)
  # path = untar_data(ds_url)

  # ds_dic = {}

  # if path.exists():
  # for element in path.ls():
  # if 'train' in str(element):
  # ds_dic['train'] = ImageList.from_folder(
  # element).split_none().label_from_folder().databunch()
  # if 'val' in str(element):
  # ds_dic['val'] = ImageList.from_folder(
  # element).split_none().label_from_folder().databunch()
  # if 'test' in str(element):
  # ds_dic['test'] = ImageList.from_folder(
  # element).split_none().label_from_folder().databunch()
  # if 'images' in str(element):
  # ds_dic['train'] = ImageList.from_folder(
  # element).split_none().label_from_folder().databunch().train_ds
  # ds_dic['test'] = ImageList.from_folder(
  # element).split_none().label_from_folder().databunch().test_ds

  # else:
  # ds_dic = {'train': {'name': name, 'source': 'fastai'}}

  # return ds_dic
