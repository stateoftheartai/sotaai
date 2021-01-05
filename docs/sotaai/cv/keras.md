# Keras

Wrapper: `keras_wrapper.py`

General Notes:

- Wraps datasets and models of the [Keras](https://keras.io/) library
- Provides standardized functions to load datasets and models: `load_dataset` and `load_model`

## Datasets

Function: `load_dataset`

Return Type: `KerasDatasetDict` (see below)

## Models

Function: `load_model`

Return Type: `Tensorflow.Functional` (see below)

Notes:

- Tensorflow `Functional` instances are wrapped in the Keras Functionl API [Model class or `keras.Model`](https://keras.io/api/models/model/)

## Return Types

### `Tensorflow.Functional`

A tensorflow functional object as defined in [tensorflow.python.keras.engine.functional.Functional](https://www.tensorflow.org/api_docs/python/tf/keras/Model#top_of_page)

### `KerasDatasetDict`

A python [`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries). Each dictionary key corresponds to a dataset split, the dictionary values will be Numpy arrays with the corresponding data. The dictionary is to contain the following keys i.e. splits:

- `train`
- `test`

For more details check [Keras Datasets](https://keras.io/api/datasets/) or the docs of a particular dataset like [Keras MNIST](https://keras.io/api/datasets/mnist/)
