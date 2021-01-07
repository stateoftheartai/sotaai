# TensorFlow Datasets

Wrapper: [`tensorflow_wrapper.py`](../../../sotaai/cv/tensorflow_wrapper.py)

General Notes:

- Wraps **only datasets** of the [TensorFlow
  Datasets](https://www.tensorflow.org/datasets/) library.
- Provides standardized functions only to load datasets: `load_dataset`.

## Datasets

Function: `load_dataset`

```
Return Type: `TensorFlowDatasetDict` (see below)
```

## Models

Not available.

## Return Types

### `TensorFlowDatasetDict`

A python
[`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).
Each dictionary key corresponds to a dataset split, the dictionary values will
be `tensorflow.python.data.ops.dataset_ops._OptionsDataset` objects. The
dictionary will contain some of the following keys:

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

For more details check the [Splits and slicing
section](https://www.tensorflow.org/datasets/splits) of their documentation.
