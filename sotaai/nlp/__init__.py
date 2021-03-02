# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main NLP module to abstract away library specific API and standardize.'''
from sotaai.nlp.abstractions import NlpModel, NlpDataset


def load_model(name: str) -> NlpModel:
  '''TODO(lalito) describe.'''
  # Get the source library for model `name`.
  source = utils.get_source_from_name(name)  # Would return something like
  # "huggingface".
  wrapper = get_wrapper_from_source(source)  # Would have some module like
  # huggingface_wrapper.
  raw_model = wrapper.load_model(name)
  return NlpModel(raw_model)


def load_dataset(name: str) -> NlpDataset:
  '''TODO(lalito) describe.'''
  # Get the source library for dataset `name`.
  source = utils.get_source_from_name(name)  # Would return something like
  # "huggingface".
  wrapper = get_wrapper_from_source(source)  # Would have some module like
  # huggingface_wrapper.
  raw_dataset = wrapper.load_dataset(name)
  return NlpDataset(raw_dataset)


def model_to_dataset(model: NlpModel,
                     dataset: NlpDataset) -> (NlpModel, NlpDataset):
  '''TODO(lalito) describe.'''
  # 1. Check whether to modify the model or the datset.
  # 2. Modify whatever is necessary, e.g.,
  #
  #  compatible_model = utils.modify_layers(model, dataset), o
  #
  #  compatible_dataset = utils.resize_samples(model, dataset)
  #
  raise NotImplementedError("TODO(lalito) implement me")


'''

model = nlp.load_model("LSTM")
dataset = nlp.load_dataset("Wikipedia")

# No necesarimente podrian correr. Esto podria causar error:
sample = dataset.get_sample()
prediction = model.evaluate(sample)  # Me va a causar error.


# En cambio, hacer esto SIEMPRE funciona.
model = nlp.load_model("LSTM")
dataset = nlp.load_dataset("Wikipedia")
(compatible_model, compatible_dataset) = nlp.model_to_dataset(model, dataset)

sample = compatible_dataset.get_sample()
prediction = compatible_model.evaluate(sample)  # SIEMPRE JALA.

'''
