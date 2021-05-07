# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main CV module to abstract away library specific API and standardize.'''
from sotaai.cv import utils
from sotaai.cv import abstractions
from sotaai.cv import keras_wrapper
from sotaai.cv import torchvision_wrapper
from sotaai.cv import metadata
import importlib

datasets_source_map = utils.map_name_sources('datasets')


def load_model(name: str,
               source: str = '',
               pretrained=None,
               keras_alpha=1.0,
               keras_depth_multiplier=1,
               keras_dropout=0.001,
               keras_input_tensor=None,
               keras_input_shape=None,
               keras_include_top=False,
               keras_pooling=None,
               keras_classes=1000,
               keras_classifier_activation='softmax',
               import_library=True) -> abstractions.CvModel:
  '''Fetch a model from a specific source, and return standardized object.

      Args:
        name (str):
          Name of the model.
        source (str):
          Optional parameter to indicate a specific source library.

      Returns (abstractions.CvModel):
        The standardized model.
      '''
  model_source_map = utils.map_name_sources('models',
                                            return_original_names=False)
  lower_name = name.lower()
  valid_sources = model_source_map[lower_name]

  if source and source not in valid_sources:
    raise NameError(
        'Source {} not available for model {}.'.format(source, name) +
        ' Available sources are: {}'.format(valid_sources))
  else:
    source = valid_sources[0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')

  # TODO(Hugo)
  # Create an abstraction for the input of models that standardizes model inputs
  # across different libraries (configs)
  # As of now, we only have one input: pretrained or not

  # if source in ['torch', 'keras']:
  if not import_library:
    return abstractions.CvModel(name=name)
  elif source == 'keras':
    raw_object = wrapper.load_model(
        name,
        pretrained=pretrained,
        alpha=keras_alpha,
        depth_multiplier=keras_depth_multiplier,
        dropout=keras_dropout,
        input_tensor=keras_input_tensor,
        input_shape=keras_input_shape,
        include_top=keras_include_top,
        pooling=keras_pooling,
        classes=keras_classes,
        classifier_activation=keras_classifier_activation)
  elif source == 'torchvision':
    raw_object = wrapper.load_model(name, pretrained=pretrained)
  # Non fully implemented sources fall in this case
  else:
    raw_object = wrapper.load_model(name)

  return abstractions.CvModel(name, raw_object)


def load_dataset(name: str,
                 download=True,
                 source: str = '',
                 torch_transform=None,
                 torch_root='default',
                 torch_target_transform=None,
                 torch_ann_file=None,
                 torch_extensions=None,
                 torch_frames_per_clip=None,
                 import_library=True) -> abstractions.CvDataset:
  '''Fetch a dataset from a specific source, and return standardized object.

      Args:
        name (str):
          Name of the dataset.
        source (str):
          Optional parameter to indicate a specific source library.

      Returns (abstractions.CvDataset):
        The standardized dataset.

      # TODO(tonioteran) Add input sanitizer checks to make sure
      # we're loading only available models.
      '''
  valid_sources = datasets_source_map[name]
  if not import_library:
    return abstractions.CvDataset(name=name)
  if source and source not in valid_sources:
    raise NameError(
        'Source {} not available for dataset {}.'.format(source, name) +
        ' Available sources are: {}'.format(valid_sources))
  else:
    source = datasets_source_map[name][0]

  wrapper = importlib.import_module('sotaai.cv.' + source + '_wrapper')

  # TODO(Hugo)
  # Remove this variable or comment it out to create the full JSON data
  # This is a temporal variable to test the JSON creation only for a subset of
  # datasets to save memory. If a dataset is in this array, then its JSON will
  # be created with its full data and the dataset will be downloaded
  test_datasets = [
      # 'mnist',
      # 'cifar10',
      # 'cifar100',
      # 'fashion_mnist',
      # 'beans',
      # 'binary_alpha_digits',
      # 'caltech_birds2010',
      # 'caltech_birds2011',
      # 'cars196',
      # 'cats_vs_dogs',
      # 'omniglot',
      # 'lost_and_found',
      # 'wider_face',
      # 'cats_vs_dogs',
      # 'cmaterdb',
      # 'colorectal_histology',
      # 'colorectal_histology_large',
      # 'cycle_gan',
      # 'diabetic_retinopathy_detection',
      # 'downsampled_imagenet',
      # 'dtd',
      # 'emnist',
      # 'eurosat',
      # 'food101',
      # 'geirhos_conflict_stimuli',
      # 'horses_or_humans',
      # 'i_naturalist2017',
      # 'imagenet_resized',
      # 'imagenette',
      # 'imagewang',
      # 'kmnist',
      # 'lfw',
      # 'malaria',
      # 'mnist_corrupted',
      # 'omniglot',
      # 'oxford_flowers102',
      # 'oxford_iiit_pet',
      # 'patch_camelyon',
      # 'places365_small',
      # 'quickdraw_bitmap',
  ]

  if name in test_datasets:
    # TODO(Hugo)
    # As more sources are being added (fully-implemented), update the IF
    # statement.
    # The IF was added temporary to make sure only fully implemented sources
    # have the raw object and can actually be used in code
    if source == 'torchvision':
      raw_object = wrapper.load_dataset(name,
                                        transform=torch_transform,
                                        ann_file=torch_ann_file,
                                        target_transform=torch_target_transform,
                                        root=torch_root,
                                        extensions=torch_extensions,
                                        frames_per_clip=torch_frames_per_clip,
                                        download=download)
    else:
      raw_object = wrapper.load_dataset(name)
  # The next two cases, will return the dataset instance (raw_object) but skip
  # the download. This allows the JSON creation for all datasets no matter if
  # they are downloaded or not
  elif source in ['keras', 'tensorflow', 'torchvision']:
    raw_object = wrapper.load_dataset(name, download=download)
  else:
    raw_object = wrapper.load_dataset(name)

  # Build a standardized `CvDataset` object per dataset split:
  std_dataset = dict()
  for split_name in raw_object:
    raw = raw_object[split_name]

    # TODO(Hugo)
    # As of now, iterator does not exists for those sources not fully
    # implemented or tested, once all sources are implemented this if will be
    # irrelevant since all wrappers will have their iterator class
    iterator = None
    if raw and hasattr(wrapper, 'DatasetIterator'):
      iterator = wrapper.DatasetIterator(raw)

    std_dataset[split_name] = abstractions.CvDataset(name=name,
                                                     raw_dataset=raw,
                                                     iterator=iterator,
                                                     split_name=split_name)

  return std_dataset


def model_to_dataset(cv_model, cv_dataset, cv_task=None):
  '''If compatible, adjust model and dataset so that they can be executed
      against each other

      Args:
        cv_model: an abstracted cv model
        cv_dataset: an abstracted cv dataset
        cv_task: a cv task. In case an abstracted cv
                 dataset is found in multiple tasks

      Returns:
        cv_model: the abstracted cv model adjusted to be executed against
          cv_dataset
        cv_dataset: the abstracted cv dataset adjust to be executed against
          cv_model
      '''

  # Uncomment following prints to test model_to_dataset input and outputs...
  # print('\nModel ', cv_model.name)
  # print(' Input: ', cv_model.original_input_shape)
  # print(' Output: ', cv_model.original_output_shape)
  # print(' Input Type', cv_model.original_input_type)
  # print('Dataset: ', cv_dataset.name)
  # print(' Shape:   ', cv_dataset.shape)
  # print(' Classes: ', cv_dataset.classes_shape)

  if cv_model.source == 'keras':
    cv_model, cv_dataset = keras_wrapper.model_to_dataset(cv_model, cv_dataset)

  elif cv_model.source == 'torchvision':
    task = cv_task if cv_task in cv_dataset.tasks else cv_dataset.tasks[0]
    if task == 'classification':
      torchvision_wrapper.model_to_dataset_classification(cv_model, cv_dataset)
    elif task == 'segmentation':
      torchvision_wrapper.model_to_dataset_segmentation(cv_model, cv_dataset)
    elif task in ('object_detection', 'pose estimation'):

      if not cv_dataset.name in utils.OBJECT_DETECTION_COMPATIBILITY[
          cv_model.name]:
        raise Exception(
            f'{cv_dataset.name} is not compatible with {cv_model.name}')

      torchvision_wrapper.model_to_dataset_object_detection(
          cv_model, cv_dataset)
    elif task == 'keypoint_detection':
      torchvision_wrapper.model_to_dataset_keypoint_detection(
          cv_model, cv_dataset)

  return cv_model, cv_dataset


def create_models_dict(model_names,
                       models_sources_map,
                       import_library=False,
                       log=False):
  '''Given a list of model names, return a list with the JSON representation
      of each model as an standardized dict

      Args:
        model_names (list): list of model names to return the standardized dict
        models_sources_map: a dict map between model names and sources
        as returned by the utils function map_name_sources('models')

      Returns:
        A list of dictionaries with the JSON representation of each CV model
      '''

  models = []

  for i, model_name in enumerate(model_names):
    unified_name = metadata.get_unified_name('models', model_name)

    if log:
      print(' - ({}/{}) {}, unified: {}'.format(i + 1, len(model_names),
                                                'models.' + model_name,
                                                unified_name))
    model = load_model(name=model_name, import_library=import_library)
    model_dict = model.to_dict()

    model_dict['sources'] = models_sources_map[model_dict['name']]
    del model_dict['source']

    model_dict['implemented_sources'] = utils.get_implemented_sources(
        model_dict['sources'])

    model_dict['unified_name'] = unified_name

    models.append(model_dict)

  # Return only one model per unified name
  # TODO(Hugo)
  # Unified models do not have all the attributes of a model e.g.
  # cv_input_shape_height, cv_num_layers, etc. We still have to define how to
  # treat these fields when models are unified, since these attributes may vary
  # from one model to other even when they have the same unified name e.g.
  # ResNet102 and ResNet152 are both ResNet but they have a different
  # cv_num_layers value.
  unified_models = {}
  for model in models:
    if model['unified_name'] not in unified_models:
      unified_models[model['unified_name']] = {
          'name': model['unified_name'],
          'type': model['type'],
          'paper': model['paper'],
          'name_alt': [model['name']],
          'tasks': model['tasks'],
          'sources': model['sources'],
          'implemented_sources': model['implemented_sources']
      }
    else:
      unified_models[model['unified_name']]['name_alt'].append(model['name'])
      unified_models[model['unified_name']]['tasks'] = unified_models[
          model['unified_name']]['tasks'] + model['tasks']
      unified_models[model['unified_name']]['sources'] = unified_models[
          model['unified_name']]['sources'] + model['sources']
      unified_models[model['unified_name']][
          'implemented_sources'] = unified_models[model['unified_name']][
              'implemented_sources'] + model['implemented_sources']

  unified_models_list = list(unified_models.values())

  for model in unified_models_list:
    model['name_alt'] = list(set(model['name_alt']))
    model['tasks'] = list(set(model['tasks']))
    model['sources'] = list(set(model['sources']))
    model['implemented_sources'] = list(set(model['implemented_sources']))

  print('\nNOT UNIFIED MODELS: {}'.format(len(models)))
  print('UNIFIED MODELS: {}'.format(len(unified_models_list)))

  return unified_models_list


def create_datasets_dict(dataset_names,
                         dataset_sources_map,
                         import_library=False,
                         log=False):
  '''Given a list of dataset names, return a list with the JSON representation
      of each dataset as an standardized dict

      Args:
        dataset_names (list): list of dataset names to return
        the standardized dict dataset
        dataset_sources_map: a dict map between dataset names and sources as
          returned by the utils function map_name_sources('datasets')

      Returns:
        A list of dictionaries with the JSON representation of each CV model
      '''

  datasets = []

  for i, dataset_name in enumerate(dataset_names):

    if log:
      print(' - ({}/{}) {}'.format(i + 1, len(dataset_names),
                                   'datasets.' + dataset_name))

    # Abstract datasets are created per split, but for the JSON representation
    # we only want one global representation which contains the split metadata
    # as an attribute, that's why we have to iterate over the splits to extract
    # the splits information and then extend the dataset dict with this split
    # data
    dataset_splits = load_dataset(name=dataset_name,
                                  import_library=import_library)

    dataset_dict = None
    split_names = []
    split_num_items = []
    total_items = 0

    if not import_library:
      dataset_dict = dataset_splits.to_dict()
    else:
      for split_name in dataset_splits:
        dataset = dataset_splits[split_name]
        dataset_dict = dataset.to_dict()
        split_names.append(split_name)

        if dataset_dict['cv_num_items']:
          split_num_items.append(dataset_dict['cv_num_items'])

        if dataset_dict['cv_num_items'] is not None:
          total_items += dataset_dict['cv_num_items']

        del dataset_dict['source']
        del dataset_dict['cv_num_items']

    dataset_dict['sources'] = dataset_sources_map[dataset_dict['name']]

    dataset_dict['implemented_sources'] = utils.get_implemented_sources(
        dataset_dict['sources'])

    dataset_dict['cv_split_names'] = split_names
    dataset_dict['cv_split_num_items'] = split_num_items
    dataset_dict['cv_total_items'] = total_items

    dataset_dict['unified_name'] = dataset_name  # TODO(tonio) unify.

    datasets.append(dataset_dict)

  return datasets
