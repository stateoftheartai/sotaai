# MxNet

Wrapper: [`mxnet_wrapper.py`](../../../sotaai/cv/mxnet_wrapper.py)

General Notes:

- Wraps CV datasets and models of the
  [MxNet](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/) library.
- Provides standardized functions to load datasets and models: `load_dataset`
  and `load_model`.

## Datasets

Function: `load_dataset`

```
Return Type: `MxNetDatasetDict` (see below)
```

## Models

Function: `load_model`

Return Type: ... TODO(huguito)

## Return Types

### `MxNetDatasetDict`

A python
[`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).
Each dictionary key corresponds to a dataset split, the dictionary values will
be `mxnet.gluon.data` objects. The dictionary contains the following keys:

- `train`
- `test`

For more details check [mxnet's Gluon `Datasets` and
`DataLoader`](https://mxnet.apache.org/versions/1.7/api/python/docs/tutorials/packages/gluon/data/datasets.html).
