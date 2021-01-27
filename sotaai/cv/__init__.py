# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
"""Main CV module to abstract away library specific API and standardize."""
from sotaai.cv import utils


def load_model(name: str):
  return name


def load_dataset(name: str):
  return name
