# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Author: Liubove Orlov Savko
# Copyright: Stateoftheart AI PBC 2020.
"""Abstract classes for standardized models and datasets."""
import numpy as np
import torch
from torchvision.transforms import functional
import tensorflow_datasets as tfds
import time
import PIL
import mxnet

import sotaai.cv.pretrainedmodels_wrapper as pretrainedmodels_wrapper
import sotaai.cv.mxnet_wrapper as mxnet_wrapper
import sotaai.cv.torch_wrapper as torch_wrapper
import sotaai.cv.utils as utils


class AbstractCvDataset(object):
  """Our attempt at a standardized, task-agnostic dataset wrapper.

    Each abstract dataset represents a specific split of a full dataset.
    """

  def __init__(self, raw_object, name: str, data_type: str, split_name: str,
               tasks: list):
    """Constructor using `raw_object` from a source library.

        Args:
            raw_object:
              Dataset object directly instantiated from a source library. Type
              is dependent on the source library.
            name (str):
              Name of the dataset.
            data_type (str):
              Either "image" or "video".
            split_name (str):
              Name of the data"s corresponding split, e.g., "train" or "test".
            tasks (list):
              A list of strings corresponding to the tasks of the dataset.
        """
    self.raw = raw_object
    self.name = name
    self.data_type = data_type
    self.split_name = split_name
    self.tasks = tasks

    # Determine source library from `type` of `raw_object`.
    self.source = self._extract_source(raw_object)
    self.size = self._get_size()
    self.shape = self._get_shape()

    # Populated for datasets supporting classification or detection tasks.
    self.classes = None
    self.classes_names = None
    if "classification" in self.tasks or "object detection" in self.tasks:
      self.classes, self.classes_names = self._extract_classes(raw_object)

    # Only populated for datasets that support segmentation tasks.
    self.pixel_types = None
    self.pixel_types_names = None
    if "segmentation" in self.tasks:
      self.pixel_types, self.pixel_types_names = (
          self._extract_pixel_types(raw_object))

    # Only populated for datasets that support image captioning tasks.
    self.captions = None
    if "captioning" in self.tasks:
      self.captions = self._extract_captions(raw_object)

    self.annotations = None
    self.vocab = None
    if "visual question answering" in self.tasks:
      self.annotations = raw_object.datasets[0].annotation_db.data
      self.vocab = self._get_vocabulary(raw_object)
      self.classes, self.classes_names = self._get_classes()

  # Public functions ---

  def __getitem__(self, i: int):
    """Draw the `i`-th item from the dataset.

        Args:
            i (int):
              Index for the item to be gotten.

        Returns:
            The i-th sample as a dict. The first element of dict is a
            numpy.ndarray with shape (N, H, W, C), where N = 1
            represents the number of images, H the image height, W the
            image width, and C the number of channels. The next are labeled
            that depend on the task of the dataset
        """
    raw_object = self.raw

    if self.source == "keras":
      x, y = raw_object[0][i], raw_object[1][i]
      x = self._format_image(x)
      return {"image": x, "label": y}

    elif self.source == "mxnet":
      ds, labels = raw_object[:][0], raw_object[:][1]
      ds = ds.asnumpy()
      x, y = ds[i], labels[i]
      x = self._format_image(x)
      return {"image": x, "label": y}

    elif self.source == "torchvision":
      item_dic = {}
      if "dataset" in dir(raw_object):
        x = raw_object.dataset.data[i]
        y = np.asarray(raw_object.dataset.targets, dtype=np.int32)[i]
      elif "data" in dir(raw_object):
        x = raw_object.data[i]
        y = raw_object.labels[i]

      else:
        x, y = raw_object[i]
        if isinstance(x, PIL.Image.Image):
          x = functional.to_tensor(x)

        if "segmentation" in self.tasks:
          if isinstance(y, PIL.Image.Image):
            # Segmentation task
            mask = functional.to_tensor(y)
            mask = mask.numpy()
            mask = np.transpose(mask, [1, 2, 0]) * 255
          else:
            mask = y
        elif isinstance(y, dict):
          bbox = y["annotation"]["object"]

      x = self._format_image(x)

      item_dic["image"] = x
      if "classification" in self.tasks:
        item_dic["label"] = y
      if "segmentation" in self.tasks:
        item_dic["segmentation"] = mask
      if "object detection" in self.tasks:
        item_dic["bounding boxes"] = bbox
      if "captioning" in self.tasks:
        item_dic["caption"] = y
      if "local image descriptors" in self.tasks:
        item_dic["label"] = y

      return item_dic

    elif self.source == "tensorflow":
      example = []
      tmp = 0
      for j in raw_object.as_numpy_iterator():
        if tmp == i:
          example = j
          break
        tmp += 1
      if example == []:
        raise IndexError("Index out of bounds")
      images_ids = ["image", "image2", "lr", "hr"]
      for im_ids in images_ids:
        if im_ids in example.keys():
          image = example[im_ids]
          image = self._format_image(image)
          example[im_ids] = image
      return example

    elif self.source == "fastai":
      data_point = getattr(raw_object, self.split_name + "_ds")[i]
      x, y = data_point[0].data, data_point[1].data
      x = x.numpy()
      x = x.reshape([1, *x.shape])

    elif self.source == "mmf":
      # This will Sample datatype instead of numpy as the task
      # of VQA, captioning, etc., are substantially different
      # from rest of the task
      data = raw_object.datasets[0]
      features = data[i]["image_feature_0"].numpy()
      text = data[i]["text"].numpy()
      rest = []
      for key in data[i].keys():
        if key not in "text" and key not in "image_feature_0":
          rest.append(data[i][key])
      # Note that the returned features is not the image
      return (features, text, *rest)

    return {"image": x, "label": y}

  def get_downsampled_item(self, i: int, s: float):
    """Same as `get_item`, but with a downsized item.

        Args:
            i (int):
              Index for the item to be gotten.
            s (float):
              A float, 0 < s < 1, denoting the downsampling scale.

        Returns:
            The downsampled i-th item as numpy.array, just like in `get_item`.
        """
    # Get the i-th image of the dataset
    item = self.__getitem__(i)
    if "image" not in item.keys():
      return None
    image = self.__getitem__(i)["image"]

    # Get image size (h, w)
    shape = image.shape[1:3]
    c = image.shape[-1]
    h, w = np.ceil(np.array(shape) * s)
    h, w = int(h), int(w)

    resized_image = self._reshape(image, h, w, c)

    return resized_image

  # Private functions ---

  def _get_size(self):
    """Returns the total number of images or videos in the split."""
    if self.source == "keras":
      return len(self.raw[0])
    elif self.source in ["mmf", "mxnet", "tensorflow"]:
      return len(self.raw)
    elif self.source == "fastai":
      images = getattr(self.raw, self.split_name + "_ds")
      return len(images)
    elif self.source == "torchvision":
      if "dataset" in dir(self.raw):
        return len(self.raw.dataset.data)
      else:
        return len(self.raw)

  def _get_shape(self):
    """Returns a tuple with the shape of the images/videos.

        Returns:
            A tuple of the form (height, width, channels).
        """
    # Sample uniformly some images. If the shapes of each image
    # are different, then a None will be in the corresponding
    # dimension of the shape

    if self.source == "tensorflow":
      _, ds_info = tfds.load(self.name, with_info=True)
      if "image" in ds_info.features.keys():
        (h, w, c) = ds_info.features["image"].shape
      else:
        (h, w, c) = (None, None, None)

    else:
      n = self.size

      # Chose 10 samples and list their shapes
      indexes = np.random.choice(range(n), 10, replace=False)
      shapes = []
      for i in indexes:
        shapes.append(self.__getitem__(i)["image"].shape)
      shapes = np.array(shapes)

      h, w, c = None, None, None

      # Check whether shapes are different
      if self.source == "mmf":
        if len(set(shapes[:, 0])) == 1 and len(set(shapes[:, 1])) == 1:
          h = shapes[0][0]
          w = shapes[0][1]
      else:
        if len(set(shapes[:, 1])) == 1 and len(set(shapes[:, 2])) == 1:
          h = shapes[0][1]
          w = shapes[0][2]

        if len(set(shapes[:, 3])) == 1:
          c = shapes[0][3]

    return (h, w, c)

  def _format_image(self, x):
    """
        Args:
            x:
              numpy.ndarray or torch.Tensor that represents an image

            Returns:
              Processed numpy.ndarray of shape (1, h, w, c)
        """

    tensor_shape = x.shape

    if isinstance(x, torch.Tensor):
      x = x.numpy()

    if len(tensor_shape) == 2:
      # We only have one channel.
      x = x.reshape([1, 1, *x.shape])
    elif len(tensor_shape) == 3:
      # We have a dimension for the number of channels (dim [3]).
      x = x.reshape([1, *x.shape])

    if x.shape[3] != 1 and x.shape[3] != 3:
      # Change shape (1, c, h, w) to (1, h, w, c)
      x = np.transpose(x, [0, 2, 3, 1])
    return x

  def _extract_source(self, raw_object):
    """Determine source library from `type` of `raw_object`.

        Args:
            raw_object:
              Dataset object directly instantiated from a source library. Type
              is dependent on the source library.

        Returns:
            A string with the name of the source library.
        """
    # Save the name of the type of the object, without the first 8th digits
    # to remove "<class "" characters
    obj_type = str(type(raw_object))
    if "class" in obj_type:
      obj_type = obj_type[8:]
    if isinstance(raw_object, tuple):
      if len(raw_object) == 2 and isinstance(raw_object[0], np.ndarray):
        # Keras dataset objects are numpy.ndarray tuples.
        return "keras"
    elif "torch" in obj_type:
      return "torchvision"
    else:
      # Dataset source"s name is read from the dataset type.
      source = obj_type.split(".")[0]
      return source

  def _reshape(self, img: np.ndarray, h: int, w: int, c: int):
    """Reshapes the split"s data.

        Args:
            img (np.ndarray):
              The image to be resized. The shape is (1, h, w, c)
            h (int):
              Desired height for the data"s items.
            w (int):
              Desired width for the data"s items.
            c (int):
              Desired number of channels for the data"s items.

        Returns:
            Resized `img` as np.ndarray with desired shape (1, h", w", c").
        """
    shape = img.shape

    # PIL is more robust with torch.Tensor
    img2 = np.transpose(img[0], [2, 0, 1])
    if isinstance(img2, np.ndarray):
      img2 = torch.Tensor(img2)

    # Convert numpy to PIL image
    pil_image = functional.to_pil_image(img2)

    # Resize image
    new_image = pil_image.resize((w, h))

    # If image is RGB and c = 1, convert image to grayscale
    if c == 1 and shape[-1] == 3:
      new_image = functional.to_grayscale(new_image)

    # Convert back to numpy
    image_tensor = functional.to_tensor(new_image)
    image = image_tensor.numpy()

    # If c = 3 and image is grayscale, then stack array together
    if c == 3 and shape[-1] == 1:
      image = np.concatenate([image, image, image])

    # Obtain a (1, h, w, c) ndarray
    image = np.transpose(image, [1, 2, 0])
    image = image.reshape([1, *image.shape])
    return image

  def _extract_classes(self, raw_object):
    """Get the IDs and the names (if available) of the classes.

        Args:
            raw_object:
              Dataset object directly instantiated from a source library. Type
              is dependent on the source library.

        Returns:
            A pair of values, `classes` and `classes_names`. If no
            `classes_names` are available, the pair becomes `classes` and
            `None`.
        """
    if self.source == "mxnet":
      classes = set(raw_object[:][1])
      classes_names = None
    elif self.source == "keras":
      classes = np.unique(raw_object[1])
      classes_names = None
    elif self.source == "torchvision":
      if "VOC" in self.name:
        # If dataset is an Object Detection Dataset
        classes_names = [
            "unlabeled/void", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car ", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "potted plant", "sheep", "sofa", "train",
            "tv/monitor"
        ]
        classes = list(range(21))
      elif "class_to_idx" in dir(raw_object):
        classes = list(raw_object.class_to_idx.values())
        classes_names = list(raw_object.class_to_idx.keys())
      elif "dataset" in dir(raw_object):
        if "class_to_idx" in dir(raw_object):
          classes = list(raw_object.class_to_idx.values())
          classes_names = list(raw_object.class_to_idx.keys())
        else:
          classes = list(set(raw_object.targets))
          classes_names = None
      elif "labels" in dir(raw_object):
        classes = list(set(raw_object.labels.numpy()))
        classes_names = None
      else:
        classes = []
        finished = True
        # Set limited time to go through dataset and obtain classes
        time_end = time.time() + 20
        for i in range(self.size):
          # Append the label of each example
          classes.append(self.__getitem__(i)["label"])
          if time.time() > time_end:
            # Execute stopping condition
            finished = False
            break
        if finished:
          classes = list(set(classes))
        else:
          classes = None
        classes_names = None

    elif self.source == "tensorflow":
      _, ds_info = tfds.load(self.name, with_info=True)
      n_classes = ds_info.features["label"].num_classes
      classes = range(n_classes)
      classes_names = None

    elif self.source == "fastai":
      obj = getattr(raw_object, self.split_name + "_ds")
      classes_names = obj.y.classes
      classes = range(len(classes_names))

    if classes is not None:
      if list(range(len(classes))) == list(classes):
        # Avoid representing classes with a long list by using range
        classes = range(len(classes))

    return classes, classes_names

  def _extract_pixel_types(self, raw_object):
    """Get the IDs and the names (if available) of the pixel types.

        Args:
            raw_object:
              Dataset object directly instantiated from a source library. Type
              is dependent on the source library.

        Returns:
            A pair of values, `pixel_types` and `pixel_types_names`. If no
            `pixel_types_names` are available, the pair becomes `pixel_types`
            and `None`.
        """
    if "VOC" in self.name or "SBD" in self.name:
      classes = [
          "unlabeled/void", "aeroplane", "bicycle", "bird", "boat", "bottle",
          "bus", "car ", "cat", "chair", "cow", "diningtable", "dog", "horse",
          "motorbike", "person", "potted plant", "sheep", "sofa", "train",
          "tv/monitor"
      ]
      indexes = list(range(21))
    elif self.source == "fastai":
      obj = getattr(raw_object, self.split_name + "_ds")
      classes = obj.y.classes
      indexes = None
    else:
      indexes, classes = None, None
    return indexes, classes

  def _extract_captions(self, raw_object):
    """Get the captions from the split"s data."""
    if self.source == "torchvision":
      captions = raw_object.captions
    return captions

  def _get_vocabulary(self, raw_object):
    ds = raw_object.datasets[0]
    words = ds.text_processor.processor.vocab.vocab.get_itos()
    return words

  def _get_classes(self):
    if self.name == "visual_entailment":
      classes = [0, 1, 2]
      classes_names = ["entailment", "neutral", "contradiction"]
    else:
      classes = None
      classes_names = [
          "drama", "comedy", "romance", "thriller", "crime", "action",
          "adventure", "horror", "documentary", "mystery", "sci-fi", "fantasy",
          "family", "biography", "war", "history", "music", "animation",
          "musical", "western", "sport", "short", "film-noir"
      ]
    return classes, classes_names


def find_task(dataset_name, datasets):
  task_list = []
  for j in range(len(datasets.values())):
    if dataset_name in list(datasets.values())[j]:
      task_list.append(list(datasets.keys())[j])
  return task_list


# Model classes ---


class AbstractCvModel(object):
  """Our attempt at a standardized, model wrapper.

    Each abstract model represents a model from one of the sources.
    """

  def __init__(self, raw_object, name: str):
    """Constructor using `raw_object` from a source library.

        Args:
            raw_object:
              Model object directly instantiated from a source library. Type
              is dependent on the source library.
            name (str):
              Name of the model.
        """
    self.raw = raw_object
    self.name = name

    # Determine source library from `type` of `raw_object`.
    self.source = self._extract_source(raw_object)

    # Determine the input data type it receives
    self.input_type = self._extract_input_type(self.source)

    self.min_size = self._calculate_min_size(raw_object, self.source)
    self.num_channels = self._calculate_num_channels()

    # Model parameters
    self.nbr_layers = self._n_layers()
    self.nbr_parameters = self._calculate_parameters()
    # List to keep track of associated or compatible datasets, for the
    # purpose of interacting with the front-end element of the platform.
    self.associated_datasets = []

  # Public functions ---

  def __call__(self, image):
    """ (1, C, H, W) """
    result = self.raw(image)
    return result

  def adapt_last_layer(self, num_classes):
    """
        TODO add description
        """
    if self.source == "mxnet":
      self.raw = mxnet_wrapper.adapt_last_layer(self.raw, num_classes)
    elif self.source == "torchvision":
      self.raw = torch_wrapper.adapt_last_layer(self.raw, num_classes)
    elif self.source == "pretrainedmodels":
      self.raw = pretrainedmodels_wrapper.adapt_last_layer(
          self.raw, num_classes)

  def flatten_model(self):
    """ Some models are built with blocks of layers.
        This function flattens the blocks and returns a list
        of all layers of model.
        One of its uses is to find the number of layers
        and parameters in a programatic way


        Args:
            None. It works on the raw model
        Returns:
            list of layers. The type of layers depend on the model"s source

        """
    if self.source == "isr":
      model_tmp = self.raw.model
      k = list(model_tmp.submodules)
    elif self.source in ["keras", "segmentation_models"]:
      k = list(self.raw.submodules)
    else:
      k = []
      self._flatten(self.raw, k)
    return k

  # Private functions ---

  def _flatten(self, block, k: list):
    """ Recursive function that is used in `flatten_model()`

        Args:
            Block:
               A block of layers (type depends on source of model)
            k (list):
               Current list of layers (for recursion)

        Returns:

        """

    if self.source == "mxnet":
      bottleneck_layer = mxnet.gluon.model_zoo.vision.BottleneckV1
      list1 = dir(bottleneck_layer)
      if "features" in dir(block):
        self._flatten(block.features, k)

      elif "HybridSequential" in str(type(block)):
        for j in block:
          self._flatten(j, k)

      elif "Bottleneck" in str(type(block)):
        list2 = dir(block)
        for ll in list1:
          list2.remove(ll)
        subblocks = [x for x in list2 if not x.startswith("_")]
        for element in subblocks:
          attr = getattr(block, element)
          self._flatten(attr, k)
      else:
        k.append(block)

    else:
      for child in block.children():
        obj = str(type(child))
        if "container" in obj or "torch.nn" not in obj:
          self._flatten(child, k)
        else:
          k.append(child)

  def _calculate_parameters(self):
    """ Calculate the number of parameters in model
        This depends on the number of trainable weights and biases"""
    n_params = 0
    layers = self.flatten_model()

    # Tensorflow models and pytorch models have distinct attributes
    # Tensorflow has attribute `weigths` while pytorch has `weight`
    if self.input_type == "torch.Tensor":
      for layer in layers:
        if "weight" in dir(layer):
          if layer.weight is not None:
            weights = np.array(layer.weight.shape)
            # If a layer do not have a weight, then
            # it won"t have a bias either

            if "bias" in dir(layer):
              if layer.bias is not None:
                if self.source == "mxnet":
                  bias = layer.bias.shape[0]
                else:
                  bias = len(layer.bias)
                params_layer = np.prod(weights) + bias
              else:
                params_layer = np.prod(weights)
            else:
              params_layer = np.prod(weights)
            n_params += params_layer

    else:
      # tf and keras based models
      for layer in layers:
        if "get_weights" in dir(layer):

          if layer.get_weights() != []:
            if len(layer.get_weights()) <= 2:
              weights = np.array(layer.get_weights()[0]).shape
            else:
              weights = np.array(layer.get_weights()).shape

          # If a layer do not have a weight, then
          # it won"t have a bias either

            if "bias" in dir(layer):

              if layer.bias is not None and layer.use_bias:
                bias = layer.bias.shape[0]
                params_layer = np.prod(weights) + bias
              else:
                params_layer = np.prod(weights)

            else:
              params_layer = np.prod(weights)
            n_params += params_layer

    return n_params

  def _n_layers(self):
    n_layers = 0
    layers = self.flatten_model()
    for layer in layers:
      layer_name = str(type(layer)).lower()
      conv1_bool = "conv1d" in layer_name
      conv2_bool = "conv2d" in layer_name
      conv3_bool = "conv3d" in layer_name
      linear_bool = "linear" in layer_name or "dense" in layer_name

      if conv1_bool or conv2_bool or conv3_bool or linear_bool:
        n_layers += 1
    return n_layers

  def _extract_source(self, raw_object):
    """Determine source library from `type` of `raw_object`.

        Args:
            raw_object:
              Dataset object directly instantiated from a source library.
              Type is dependent on the source library.

        Returns:
            A string with the name of the source library.
        """
    # Some models, particularly from segmentation_models, fastai
    # and GANS libraries, have the `source` attribute instantiated
    # in their corresponding wrappers
    if "source" in dir(raw_object):
      return raw_object.source
    elif "torchvision" in str(type(raw_object)):
      return "torchvision"
    elif "pretrainedmodels" in str(type(raw_object)):
      return "pretrainedmodels"
    elif "mxnet" in str(type(raw_object)):
      return "mxnet"
    elif "keras" in str(type(raw_object)):
      return "keras"
    elif "detectron2" in str(type(raw_object)):
      return "detectron2"
    elif "segmentation_models_pytorch" in str(type(raw_object)):
      return "segmentation_models_pytorch"
    elif "ISR" in str(type(raw_object)):
      return "isr"
    elif "mmf" in str(type(raw_object)):
      return "mmf"

  def _extract_input_type(self, source):
    if source in [
        "torchvision", "mxnet", "segmentation_models_pytorch",
        "pretrainedmodels", "fastai", "mmf", "gans_pytorch"
    ]:
      return "torch.Tensor"
    elif source in ["isr", "segmentation_models", "keras", "gans_keras"]:
      return "numpy.ndarray"
    elif source == "detectron2":
      raise NotImplementedError

  def _calculate_min_size(self, raw_object, source):
    """ Calculates the minimum size of image that the model accepts
        Any size smaller than this will cause error when running the model

        Args:
            raw_object:
              Dataset object directly instantiated from a source library. Type
              is dependent on the source library.
            source (str):
              A string with the name of the source library.

        Returns:
            int or None if timed out.
        """
    if source == "mxnet":
      size = utils.find_size_mxnet(raw_object)
      return size
    elif source in ["torchvision", "segmentation_models_pytorch", "fastai"]:
      size = utils.find_size_torch(raw_object)
      return size
    elif source in ["keras", "segmentation_models"]:
      size = utils.find_size_keras(raw_object)
      return size
    elif source == "mmf":
      return 1
    elif source == "pretrainedmodels":
      return utils.find_size_cadene(raw_object)
    elif source == "isr":
      return utils.find_size_keras(raw_object.model)
    # TODO(liuba): detectron2, GANS pytorch, GANS Keras

  def _calculate_num_channels(self) -> int:
    """ Finds the number of channels that the image must have
        when passed to model. 3 channels correspond to a colored
        image, and 1 channel are for gray images """
    if self.source == "mmf":
      # First extract the image model of the mmf model
      layers = self.flatten_model()
      # Find first convolutional layer
      n_channels = None
      for j in layers:
        if "Conv" in str(type(j)):
          n_channels = j.weight.shape[1]
          break
    else:
      layers = self.flatten_model()
      if self.input_type == "torch.Tensor":
        n_channels = layers[0].weight.shape[1]
      else:
        n_channels = None
        for l in layers:  # noqa E741
          if "conv" in str(type(l)):
            if len(l.weights) == 0:
              continue
            if len(l.weights) <= 2:
              if isinstance(l.weights[0], list):
                n_channels = np.array(l.weights[0])[0].shape[-2]
              else:
                n_channels = l.weights[0][0].shape[-2]
            else:
              n_channels = np.array(l.weights)[0].shape[-2]
            break
    return n_channels
