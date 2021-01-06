# fast.ai

Wrapper: [`fastai_wrapper.py`](../../../sotaai/cv/fastai_wrapper.py)

General Notes:

- Wraps CV datasets and models of the [fast.ai](https://www.fast.ai/) library.
- Provides standardized functions to load datasets and models: `load_dataset`
  and `load_model`.

## Datasets

Function: `load_dataset`

Return Type: `FastaiDatasetDict` (see below)

## Models

Function: `load_model`

Return Type: ... TODO(huguito)

## Return Types

### `FastaiDatasetDict`

A python
[`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).
Each dictionary key corresponds to a dataset split, the dictionary values will
be `ImageDataBunch` objects. The dictionary may contain some of the following
keys:

- TODO(tonioteran) Run unit test to gather all dict keys.

For more details check [fast.ai's ImageDataBunch
class](https://fastai1.fast.ai/vision.data.html#ImageDataBunch).
