# TensorFlow Datasets

Wrapper: [`tensorflow_wrapper.py`](../../../sotaai/cv/tensorflow_wrapper.py)

General Notes:

- Wraps **only datasets** of the [TensorFlow
  Datasets](https://www.tensorflow.org/datasets/catalog/overview) library.
- Provides standardized functions only to load datasets: `load_dataset`.

## Datasets

Function: `load_dataset`

Return Type: `TensorFlowDatasetDict` (see below)

Data stored in: `~/.tensorflow_datasets` (predefined by Tensorflow and cannot be changed)

## Models

Not available.

## Return Types

### `TensorFlowDatasetDict`

A python
[`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
where each key will be a split. The value will be a tensorflow IterableDataset.
This is the original value returned by [as\_numpy](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_numpy) method of tensorflow\_datasets (tfds). Available splits are:

- `train`:
- `trainA`:
- `trainB`:
- `test`:
- `testA`:
- `testB`:
- `test2015`:
- `validation`:
- `small1`:
- `small2`:
- `unlabelled`:
- `MARK`:
- `extra`:
- `A`:
- `B`:

For more details see [tfds](https://www.tensorflow.org/datasets/api_docs/python/tfds)
