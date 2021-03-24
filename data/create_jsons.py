# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Script used to create JSON files containing sotaai models and datasets metadata
'''
import importlib
import json
import sys


def main(areas: list, output_file_path='./data/output/items.json'):
  '''Create the JSON items for the given areas. The JSON will contain:
    - models: the list of models of the given areas
    - datasets: the list of datasets of the given areas
    - sources: the list of sources of the given areas
    - tasks: the list of tasks of the given areas
    - areas: the list of areas

    Args:
      areas (list): list of areas IDs to get JSONS e.g. ['cv', 'nlp']
      output_file_path (str): the file path where JSON data will be stored
  '''

  area_original_names = {
      'cv': 'Computer Vision',
      'nlp': 'Natural Language Processing',
      'neuro': 'Neurosymbolic Reasoning',
      'rl': 'Reinforcement Learning',
      'robotics': 'Robotics'
  }

  data = {'models': [], 'datasets': [], 'sources': {}, 'tasks': {}, 'areas': []}

  for area in areas:
    models, datasets, sources, tasks = get_data(area)

    data['models'] = data['models'] + models
    data['datasets'] = data['datasets'] + datasets

    add_items(data['sources'], sources, 'name', 'area')
    add_items(data['tasks'], tasks, 'name', 'area')

    data['areas'].append({
        'name': area,
        'original_name': area_original_names[area]
    })

  data['sources'] = list(data['sources'].values())
  data['tasks'] = list(data['tasks'].values())

  save_json(data, output_file_path)

  print('\nJSONs files created successfully.\n')


def get_data(area: str):
  '''Get models, datasets, tasks and sources of a given area in an standardized
  JSON format

  Args:
    area: one of cv, nlp, rl, neuro
  '''

  print('\nGetting data...')

  try:

    sotaai_module = importlib.import_module('sotaai.{}'.format(area))
    sotaai_utils_module = importlib.import_module(
        'sotaai.{}.utils'.format(area))

    models_sources_map = sotaai_utils_module.map_name_sources('models')
    datasets_sources_map = sotaai_utils_module.map_name_sources('datasets')
    sources_metadata_map = sotaai_utils_module.map_source_metadata()

    models_by_source = items_by_source(models_sources_map)
    datasets_by_source = items_by_source(datasets_sources_map)

    model_names = models_sources_map.keys()
    dataset_names = datasets_sources_map.keys()

    models = sotaai_module.create_models_dict(model_names, models_sources_map)
    datasets = sotaai_module.create_datasets_dict(dataset_names,
                                                  datasets_sources_map)
    sources = list(sources_metadata_map.values())
    tasks = get_catalogue('tasks', models + datasets)

    print('\nArea: {}'.format(area.upper()))
    print('Unique Models: {}'.format(len(models)))
    for source in models_by_source:
      print(' - {}: {}'.format(source, models_by_source[source]))
    print('Unique Datasets: {}'.format(len(datasets)))
    for source in datasets_by_source:
      print(' - {}: {}'.format(source, datasets_by_source[source]))
    print('Unique Tasks: {}'.format(len(tasks)))

    add_area(area, models)
    add_area(area, datasets)
    add_area(area, sources)
    add_area(area, tasks)

    return models, datasets, sources, tasks

  except Exception as e:
    raise NotImplementedError(
        'Getting data for {} failed or still not implemented'.format(
            area)) from e


def add_items(items: dict, new_items: list, unified_field: str,
              append_field: str):
  '''Add items to the global items dictionary, keep the unified by the given
    unified_field, and per each unified item it appends the different values of
    the append_field of each item e.g. unify items by name, and for each unified
    item concats the 'area' value, so that we can know each unified item to
    which areas belongs

    Args:
      items (dict): global items dictionary where unified items will be stored
      new_items (list): new items to add to the dictionary
      unified_field (str): the field used for unification
      append_field (str): the field to append
  '''

  for item in new_items:
    item_unified_name = item[unified_field]
    if item_unified_name in items:
      current = items[item_unified_name][append_field]
      new = item[append_field]
      items[item_unified_name][append_field] = list(set(current + [new]))
    else:
      item[append_field] = [item[append_field]]
      items[item_unified_name] = item


def items_by_source(items_sources_map: dict) -> dict:
  '''Gets a count of the number of items by source

  It is used mainly for logging and debugging purposes

  Args:
    items_sources_map: a valid model or datasets map as returned by
    utils.map_name_sources

  Returns:
    A dict of the form:
      <source-name>: number of items e.g. models or datasets in that source
  '''
  by_source = {}
  for item_name in items_sources_map:
    sources = items_sources_map[item_name]
    for source in sources:
      if source not in by_source:
        by_source[source] = 0
      by_source[source] += 1
  return by_source


def add_area(area: str, items):
  '''Update the given items to add the area field as a string

  Args:
    area (str): the area value to add to each of the items e.g. 'cv'
    items (list): the list of items to add the area field
  '''
  for item in items:
    item['area'] = area


def get_catalogue(field: str, items: list) -> list:
  '''Given a list of items with a certain field, returns the unique values of
  the given fields i.e. the catalogue of values of the field

  Args:
    field: field to get catalogue from e.g. 'tasks'
    items: list to of items that have the field

  Returns:
    The catalogue as a list of objects with name and original_name.
  '''
  catalogue = {}
  for item in items:
    value = item[field]
    if isinstance(value, str):
      value = [value]
    for name in value:
      if name not in catalogue:
        original_name = create_original_name(name)
        catalogue[name] = {'name': name, 'original_name': original_name}
  return list(catalogue.values())


def create_original_name(name) -> str:
  '''Create an original_name given a name. It capitalizes words
  and replace _ with spaces

  Args:
    name (str): name from which original_name will be created

  Returns:
    The original name as string
  '''
  # Temporal fix due to those tasks which are still not named with _ and lower
  # case
  name = name.replace(' ', '_').lower()

  original_name = None
  for item in name.split('_'):
    if original_name is None:
      original_name = item.capitalize()
    else:
      original_name += ' ' + item.capitalize()
  original_name = original_name.strip()
  return original_name


def save_json(data, file_path):
  '''Store the given data in a JSON file

  Args:
    data (dict): a valid JSON value (dict or list of dicts)
    file_path (str): the file path where to store the JSON file e.g. cv.json
  '''
  data_file = open(file_path, 'w')
  json.dump(data, data_file, indent=2)
  data_file.close()


args = sys.argv[1:]

if len(args) == 0:
  _areas = ['cv', 'nlp', 'neuro', 'rl']
else:
  _areas = [args[0]]

main(_areas)
