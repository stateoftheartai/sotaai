# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
"""Module used to interface with fastai"s library."""
from fastai.vision import ImageDataBunch, ObjectItemList
from fastai.vision import PointsItemList, SegmentationItemList
from fastai.vision import URLs, untar_data, models
from fastai.vision import get_image_files, imagenet_stats
from fastai.vision import get_transforms, get_annotations
from fastai.vision import pickle, bb_pad_collate
import numpy as np
import os
from os.path import expanduser

MODELS = [
    "alexnet", "densenet121", "densenet161", "densenet169", "densenet201",
    "mobilenet_v2", "resnet101", "resnet152", "resnet18", "resnet34",
    "resnet50", "squeezenet1_0", "squeezenet1_1", "vgg11_bn", "vgg13_bn",
    "vgg16_bn", "vgg19_bn", "xresnet101", "xresnet152", "xresnet18",
    "xresnet18_deep", "xresnet34", "xresnet34_deep", "xresnet50",
    "xresnet50_deep"
]

DATASETS = {
    "classification": [
        "CALTECH_101", "CIFAR", "CIFAR_100", "DOGS", "IMAGENETTE",
        "IMAGENETTE_160", "IMAGENETTE_320", "IMAGEWOOF", "IMAGEWOOF_160",
        "IMAGEWOOF_320", "MNIST", "MNIST_SAMPLE", "MNIST_TINY",
        "MNIST_VAR_SIZE_TINY", "PETS", "SKIN_LESION"
    ],
    "key-point detection": ["BIWI_SAMPLE"],
    "object detection": ["COCO_SAMPLE", "COCO_TINY"],
    "multi-label classification": ["PLANET_SAMPLE", "PLANET_TINY"],
    "segmentation": ["CAMVID", "CAMVID_TINY"]
}
""" Not yet wrapped

 #"PASCAL_2007",         #Checksum does not match. No download
 #"PASCAL_2012",         #Checksum does not match. No download
 "FLOWERS",
 "BIWI_HEAD_POSE", #
 "CARS",#Classification
 "CUB_200_2011", #Classification, detection            #Caltech UCSD Birds
 "FOOD",#Classification
 "LSUN_BEDROOMS",#Classification
"""


def load_model(model_name: str, pretrained: bool = False):
  """
    Input:
        model_name: str, one from MODELS variable
        pretrained: bool, to load a pretrained or randomly initialized model
    Output:
        A `torchvision.models` class
    """
  trainer = getattr(models, model_name)
  model = trainer(pretrained=pretrained)
  model.source = "fastai"
  return model


def load_dataset(dataset_name):
  """
    Input:
        dataset_name: str, one from MODELS variable
    Output:
        A `fastai.image.data` class
    """
  url_dataset = getattr(URLs, dataset_name)
  path = untar_data(url_dataset)
  if dataset_name == "MNIST":
    data = ImageDataBunch.from_folder(path, train="training", test="testing")

  if dataset_name in [
      "MNIST_SAMPLE", "MNIST_TINY", "MNIST_VAR_SIZE_TINY", "CIFAR", "CIFAR_100",
      "DOGS"
  ]:
    if dataset_name == "DOGS":
      data = ImageDataBunch.from_folder(path, test="test1", no_check=True)
    data = ImageDataBunch.from_folder(path, test="test", no_check=True)

  if dataset_name == "SKIN_LESION":
    data = ImageDataBunch.from_folder(path, no_check=True)

  if dataset_name == "PETS":
    path_img = path / "images"
    fnames = get_image_files(path_img)
    re = r"/([^/]+)_\d+.jpg$"
    data = ImageDataBunch.from_name_re(path_img,
                                       fnames,
                                       re,
                                       ds_tfms=get_transforms(),
                                       size=224).normalize(imagenet_stats)

  # Multicategory
  if dataset_name in ["PLANET_SAMPLE", "PLANET_TINY"]:
    data = ImageDataBunch.from_csv(path,
                                   folder="train",
                                   suffix=".jpg",
                                   label_delim=" ")

  if dataset_name == "CALTECH_101":
    home = expanduser("~")
    original = home + "/.fastai/data/101_ObjectCategories"
    target = home + "/.fastai/data/caltech_101"
    os.rename(original, target)
    data = ImageDataBunch.from_folder(path, valid_pct=0.2, no_check=True)

  if dataset_name in [
      "IMAGENETTE", "IMAGENETTE_160", "IMAGENETTE_320", "IMAGEWOOF",
      "IMAGEWOOF_160", "IMAGEWOOF_320"
  ]:
    data = ImageDataBunch.from_folder(path, valid="val")

  if dataset_name in ["COCO_SAMPLE", "COCO_TINY"]:
    if dataset_name == "COCO_SAMPLE":
      annot = path / "annotations/train_sample.json"
      imgs_folder = "train_sample"
    else:
      annot = path / "train.json"
      imgs_folder = "train"
    images, lbl_bbox = get_annotations(annot)
    img2bbox = dict(zip(images, lbl_bbox))

    def get_y_func(o):
      return img2bbox[o.name]

    data = (ObjectItemList.from_folder(
        path / imgs_folder).random_split_by_pct().label_from_func(
            get_y_func).transform(
                get_transforms(),
                tfm_y=True,
                padding_mode="zeros",
                size=128,
            ).databunch(bs=16, collate_fn=bb_pad_collate))

  if dataset_name == "BIWI_SAMPLE":
    fn2ctr = pickle.load(open(path / "centers.pkl", "rb"))
    data = (PointsItemList.from_folder(path).split_by_rand_pct(
        seed=42).label_from_func(lambda o: fn2ctr[o.name]).transform(
            get_transforms(), tfm_y=True,
            size=(120, 160)).databunch().normalize(imagenet_stats))

  if dataset_name in ["CAMVID", "CAMVID_TINY"]:
    path_lbl = path / "labels"
    path_img = path / "images"
    codes = np.loadtxt(path / "codes.txt", dtype=str)

    def get_y_fn(x):
      return path_lbl / f"{x.stem}_P{x.suffix}"

    data = (SegmentationItemList.from_folder(
        path_img).split_by_rand_pct().label_from_func(
            get_y_fn, classes=codes).transform(
                get_transforms(), tfm_y=True,
                size=128).databunch(bs=16, path=path).normalize(imagenet_stats))

  ds_dic = {}
  splits = ["train_ds", "valid_ds", "test_ds"]
  for split in splits:
    if split in dir(data):
      datasplit = getattr(data, split)

      if datasplit is not None:
        if len(datasplit) > 1:
          if "valid" in split:
            ds_dic["val"] = data
          else:
            ds_dic[split.split("_")[0]] = data

  return ds_dic


def take(dataset, partition, index):
  """
    Input:
        dataset: `fastai.vision.data`
        partition: str. Either "train", "test", or "valid".
        index: the index of interest
    Output:
        a tuple (image, label), where
          image is a torch.Tensor
          label is either a torch.Tensor or int, depending on the dataset

    """
  data_point = getattr(dataset, partition + "_ds")[index]
  x, y = data_point[0].data, data_point[1].data
  return x, y


def predict():
  """
    Input:
        model: torch.nn.Module
        dataset: torch.Tensor of shape (N,C,H,W)
    Output: result of the model, which depends on the task
    """
  return NotImplementedError


def model_to_dataset():
  return Exception("Not yet implemented")
