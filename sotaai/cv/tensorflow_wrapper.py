# -*- coding: utf-8 -*-
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
"""Module used to interface with Tensorflow"s datasets."""
import tensorflow_datasets as tfds
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

DATASETS = {
    "video": [
        "bair_robot_pushing_small",
        "moving_mnist",
        "starcraft_video",
        # "ucf101"  # Bug tensorflow
    ],
    "object detection": [
        "celeb_a_hq",  # manual download
        "coco",
        "flic",
        "kitti",
        # "open_images_challenge2019_detection", Apache beam
        # "open_images_v4", Apache beam
        "voc",
        "the300w_lp"
        # "wider_face" # Wrong checksum
    ],
    "classification": [
        "beans",
        "binary_alpha_digits",
        # "caltech101",  # Wrong checksum
        "caltech_birds2010",
        "caltech_birds2011",
        "cars196",
        "cats_vs_dogs",
        "celeb_a",
        "celeb_a_hq",  # manual download
        "cifar10",
        "cifar100",
        "cifar10_1",
        "cifar10_corrupted",
        # "citrus_leaves",  # Wrong checksum
        "cmaterdb",
        "colorectal_histology",
        "colorectal_histology_large",
        "curated_breast_imaging_ddsm",  # manual download
        "cycle_gan",
        "deep_weeds",  # manual download
        "diabetic_retinopathy_detection",
        "downsampled_imagenet",
        "dtd",
        "emnist",
        "eurosat",
        "fashion_mnist",
        "food101",
        "geirhos_conflict_stimuli",
        "horses_or_humans",
        "i_naturalist2017",
        "imagenet2012",  # manual download
        "imagenet2012_corrupted",  # manual download
        "imagenet2012_subset",  # manual download
        "imagenet_resized",
        "imagenette",
        "imagewang",
        "kmnist",
        "lfw",
        "malaria",
        "mnist_corrupted",
        "omniglot",
        "oxford_flowers102",
        "oxford_iiit_pet",
        "patch_camelyon",
        "places365_small",
        "quickdraw_bitmap",
        "resisc45",  # manual download
        "rock_paper_scissors",
        "smallnorb",
        "so2sat",
        "stanford_dogs",
        "stanford_online_products",
        "stl10",
        "sun397",
        "svhn_cropped",
        "tf_flowers",
        "uc_merced",
        "vgg_face2",  # manual download
        "visual_domain_decathlon"
    ],
    "segmentation": [
        "cityscapes",  # manual download
        "lost_and_found",
        "scene_parse150"
    ],
    "image super resolution": ["div2k",],
    "key point detection": ["aflw2k3d", "celeb_a", "the300w_lp"],
    "pose estimation": ["flic", "the300w_lp"],
    "face alignment": ["the300w_lp"],
    "visual reasoning": ["clevr"],
    "visual question answering": ["clevr"],
    "image generation": ["dsprites", "shapes3d"],
    "3d image generation": ["shapes3d",],
    "other": [
        "binarized_mnist",
        "chexpert",  # manual download
        "coil100",
        "lsun"
    ]
}


def flatten(data_str):
  flat = []
  for sublist in data_str.values():
    for item in sublist:
      flat.append(item)
  return flat


def available_datasets(task: str = "all"):
  if task == "all":
    flat_ds = flatten(DATASETS.values())
    return flat_ds

  return DATASETS[task]


def load_dataset(name_dataset):
  _, ds_info = tfds.load(name_dataset, with_info=True)

  ls = list(ds_info.splits.keys())
  ds = tfds.load(name_dataset, split=ls, shuffle_files=True)

  ds_dic = {}
  for i in range(len(ls)):
    ds_dic[ls[i]] = ds[i]
  return ds_dic


def take(ds, index):
  """
    Input:
        ds - tensorflow dataset.
        index - int, the number of example to extract from ds
    Output:
        The example of the dataset. If the example contains an image,
        it is automatically normalized
    """
  example = []
  i = 0
  for j in ds.as_numpy_iterator():
    if i == index:
      example = j
      break
    i += 1
  if example == []:
    return print("index out of range")

  return example
