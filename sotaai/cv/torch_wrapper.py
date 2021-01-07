# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
"""Module used to interface with Torchvision"s models and datasets."""
from torchvision import models
from torchvision import datasets as dset
from torch import nn
import os
import torch

DATASETS = {
    "classification": [
        "CIFAR10",
        "CIFAR100",
        "CelebA",
        "EMNIST",
        "FashionMNIST",
        "KMNIST",
        "LSUN",  # No download
        "MNIST",
        "Omniglot",
        "QMNIST",
        "SEMEION",
        "SVHN",
        "USPS",
        "STL10",  # Unsupervised learning.
        "ImageNet",  # No download
    ],
    "object detection": [
        "CelebA",
        "CocoDetection",  # No download.
        "Flickr30k",  # No download.
        "VOCDetection/2007",
        "VOCDetection/2008",
        # "VOCDetection/2009", Corrupted
        "VOCDetection/2010",
        "VOCDetection/2011",
        "VOCDetection/2012",
    ],
    "segmentation": [
        "Cityscapes",  # No download.
        "VOCSegmentation/2007",
        "VOCSegmentation/2008",
        # "VOCSegmentation/2009", Corrupted
        "VOCSegmentation/2010",
        "VOCSegmentation/2011",
        "VOCSegmentation/2012",
        "SBD/segmentation",
        "SBD/boundaries",
    ],
    "captioning": [
        "CocoCaptions",  # No download.
        "Flickr8k",  # No download.
        "Flickr30k",  # No download.
        "SBU"
    ],
    "human activity recognition": [
        "HMDB51",  # No download.
        "Kinetics400",  # No download.
        "UCF101",  # No download.
    ],
    "one-shot-learning": ["Omniglot"],
    "few-shot-learning": ["Omniglot"],
    "local image descriptors": [
        "PhotoTour/notredame",
        "PhotoTour/yosemite",
        "PhotoTour/liberty",
        "PhotoTour/notredame_harris",
        "PhotoTour/yosemite_harris",
        "PhotoTour/liberty_harris",
    ],
}

MODELS_dic = {
    "classification": [
        "alexnet", "densenet121", "densenet161", "densenet169", "densenet201",
        "googlenet", "inception_v3", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0",
        "mnasnet1_3", "mobilenet_v2", "resnet101", "resnet152", "resnet18",
        "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0", "squeezenet1_0", "squeezenet1_1", "vgg11",
        "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19",
        "vgg19_bn", "wide_resnet101_2", "wide_resnet50_2"
    ],
    "segmentation": [
        "deeplabv3_resnet101", "deeplabv3_resnet50", "fcn_resnet101",
        "fcn_resnet50"
    ],
    "object detection": [
        "fasterrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn",
        "maskrcnn_resnet50_fpn"
    ],
    "video": ["mc3_18", "r2plus1d_18", "r3d_18"]
}


def flatten(ll):
  flat = []
  for sublist in ll:
    for item in sublist:
      flat.append(item)
  return flat


MODELS = flatten(list(MODELS_dic.values()))


def tasks():
  return list(DATASETS.keys())


def available_datasets(task: str = "all"):
  """
    Function that lists the available datasets by task
    """
  if task == "all":
    flat_ds = flatten(DATASETS.values())
    return flat_ds

  return DATASETS[task]


def available_models(task: str = "all"):
  """
    Function that lists the available datasets by task
    """
  if task == "all":
    flat_models = flatten(models.values())
    return flat_models

  return models[task]


def load_model(model_name, pretrained=False):
  """
    Input:
        model_name: str, one from MODELS variable
        pretrained: bool, to load a pretrained or
                          randomly initialized model
    Output:
        torchvision.models
    """

  # Load the corresponding model class
  if model_name in MODELS_dic["segmentation"]:
    trainer = getattr(models.segmentation, model_name)
  elif model_name in MODELS_dic["object detection"]:
    trainer = getattr(models.detection, model_name)
  elif model_name in MODELS_dic["video"]:
    trainer = getattr(models.video, model_name)
  else:
    trainer = getattr(models, model_name)

  return trainer(pretrained=pretrained)


def load_dataset(dataset_name,
                 root="default",
                 ann_file=None,
                 frames_per_clip=None):
  """
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
    Output:
        dict, with keys indicating the partition of the dataset,
                and the values are of type DataLoader
    """

  if root == "default":
    root = "./torch/" + dataset_name

  if "SBD" in dataset_name:
    mode = dataset_name.split("/")[1]
    dataset_name = "SBDataset"

  elif "PhotoTour" in dataset_name:
    name = dataset_name.split("/")[1]
    dataset_name = "PhotoTour"
  elif "VOC" in dataset_name:
    year = dataset_name.split("/")[1]
    dataset_name = dataset_name.split("/")[0]
  ds = getattr(dset, dataset_name)
  # Datasets saved in dic
  # with corresponding splits of dataset
  ds_dic = {}

  datasets_w_train = [
      "CIFAR10", "CIFAR100", "FashionMNIST", "KMNIST", "MNIST", "QMNIST", "USPS"
  ]
  datasets_w_split = ["SVHN", "STL10", "CelebA"]

  download_train = True  # os.path.exists(root+"/train")
  download_test = True  # os.path.exists(root+"/test")

  if dataset_name in datasets_w_train:
    ds_dic["train"] = torch.utils.data.DataLoader(ds(root + "/train",
                                                     train=True,
                                                     download=download_train),
                                                  batch_size=100,
                                                  shuffle=False,
                                                  num_workers=0)
    ds_dic["test"] = torch.utils.data.DataLoader(ds(root + "/test",
                                                    train=False,
                                                    download=download_test),
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

  elif dataset_name in datasets_w_split:
    ds_dic["train"] = ds(root + "/train",
                         split="train",
                         download=download_train)
    ds_dic["test"] = ds(root + "/test", split="test", download=download_test)
    if dataset_name == "SVHN":
      ds_dic["extra_training_set"] = ds(root + "/extra",
                                        split="extra",
                                        download=True)
    elif dataset_name == "STL10":
      ds_dic["unlabeled"] = ds(root + "/extra",
                               split="unlabeled",
                               download=True)
    elif dataset_name == "CelebA":
      ds_dic["val"] = ds(root + "/val", split="valid", download=True)

  elif dataset_name == "Cityscapes":
    ds_dic["train"] = ds(root + "/train", split="train")
    ds_dic["test"] = ds(root + "/test", split="test")
    ds_dic["val"] = ds(root + "/val", split="val")

  elif dataset_name == "EMNIST":
    # split= balanced,byclass,bymerge,letters,digits,mnist
    ds_dic["train"] = ds(root + "/train",
                         split="balanced",
                         train=True,
                         download=download_train)
    ds_dic["test"] = ds(root + "/test",
                        split="balanced",
                        train=False,
                        download=download_test)

  elif "PhotoTour" in dataset_name:

    ds_dic["train"] = ds(root + "/train",
                         name=name,
                         download=download_train,
                         train=True)
    ds_dic["test"] = ds(root + "/test",
                        name=name,
                        download=download_test,
                        train=False)

  elif dataset_name == "Omniglot":
    download_bg = not os.path.exists(root + "/background")
    download_eval = not os.path.exists(root + "/eval")
    ds_dic["background"] = ds(root + "/background",
                              background=True,
                              download=download_bg)
    ds_dic["evaluation"] = ds(root + "eval",
                              background=False,
                              download=download_eval)

  elif dataset_name in ["SBU", "SEMEION"]:
    ds_dic["data"] = ds(root, download=True)

  elif dataset_name in ["VOCSegmentation", "VOCDetection"]:
    ds_dic["train"] = ds(root + "/train",
                         year=year,
                         image_set="train",
                         download=download_train)
    download_val = not os.path.exists(root + "/val")
    ds_dic["val"] = ds(root + "/val",
                       year=year,
                       image_set="val",
                       download=download_val)

  elif "SBD" in dataset_name:
    download_train = not os.path.exists(root + mode + "/train")
    download_val = not os.path.exists(root + mode + "/val")
    ds_dic["train"] = ds(root + mode + "/train",
                         image_set="train",
                         mode=mode,
                         download=download_train)

    ds_dic["val"] = ds(root + mode + "/val",
                       image_set="val",
                       mode=mode,
                       download=download_val)

  elif dataset_name == "LSUN":
    ds_dic["train"] = dset.LSUN(root, classes="train")
    ds_dic["val"] = dset.LSUN(root, classes="val")
    ds_dic["test"] = dset.LSUN(root, classes="test")

  elif dataset_name == "ImageNet":
    ds_dic["train"] = ds(root + "/train", split="train")
    ds_dic["val"] = ds(root + "/val", split="val")

  elif dataset_name in [
      "CocoDetection", "CocoCaptions", "Flickr8k", "Flickr30k"
  ]:
    ds_dic["data"] = ds(root, ann_file)

  elif dataset_name in ["HMDB51", "UCF101"]:
    ds_dic["train"] = ds(root + "train", ann_file, frames_per_clip, train=True)
    ds_dic["test"] = ds(root + "test", ann_file, frames_per_clip, train=False)

  elif dataset_name == "Kinetics400":
    ds_dic["data"] = ds(root, frames_per_clip)
  return ds_dic


def predict(model, dataset):
  """
    Input:
        model: torch.nn.Module
        dataset: torch.Tensor of shape (N,C,H,W)
    Output: result of the model, which depends on the task
    """
  return model(dataset)


def take(dataset, index):
  """
    Input:
        dataset: Dataloader
        index: int
    Output: (image, label), where image: torch.Tensor and label is int.
    """
  return dataset.dataset.data[index], dataset.dataset.targets[index]


# TODO: Model_to_dataset(). Below are functions necessary for that


def adapt_last_layer(model, classes):
  """ Change last layer of network given the number of classes in the
    dataset. This function is for classification models only"""
  layers = list(model.children())  # get children and their names
  submodule = False

  # if last layer is encapsulated in a block, get the layers of that block
  if isinstance(layers[-1], torch.nn.modules.container.Sequential):
    submodule = True
    layers = list(layers.children())
    body = nn.Sequential(*list(model.children())[:-1])

  # Find last convolutional or linear layer, as they are the ones
  # that determine the output size of the neural network
  for i in range(1, len(layers)):
    if isinstance(layers[-i], nn.Linear):
      cut = -i

      bias = bool(layers[-i].bias)

      new_layer = nn.Linear(out_features=classes,
                            in_features=layers[-i].in_features,
                            bias=bias)

      break
    bool1 = isinstance(layers[-i], nn.Conv1d)
    bool2 = isinstance(layers[-i], nn.Conv2d)
    bool3 = isinstance(layers[-i], nn.Conv3d)
    if bool1 or bool2 or bool3:
      l_type = str(type(layers[-i][1])).split(".")[-1][:6]
      conv = getattr(nn, l_type)

      bias = bool(layers[-i].bias)

      new_layer = new_layer = conv(in_channels=layers[-i].in_channels,
                                   out_channels=classes,
                                   kernel_size=layers[-i].kernel_size,
                                   stride=layers[-i].stride,
                                   padding=layers[-i].padding,
                                   dilation=layers[-i].dilation,
                                   groups=layers[-i].groups,
                                   bias=bias,
                                   padding_mode=layers[-i].padding_mode)
      break
  try:
    if not submodule:
      if cut != -1:
        list_modules_head = [new_layer] + layers[cut + 1:]
      else:
        list_modules_head = [new_layer]
      body = nn.Sequential(*layers[:cut])
      head = nn.Sequential(*list_modules_head)
      new_model = nn.Sequential(body, head)
    else:
      list_modules_head = layers[:cut] + [new_layer]
      if cut != -1:
        list_modules_head += layers[cut + 1:]
      head = nn.Sequential(*list_modules_head)
      new_model = nn.Sequential(body, head)
  except:  # pylint: disable=W0702
    msg = "The provided model is not a classification model, "
    cont = "or it does not have a linear nor a convolutional layer"

    print(msg, cont)
  return new_model


def model_to_dataset():
  return Exception("Not yet implemented")
