# Standardizing CV through abstractions

## Datasets

## Standardized model attributes and methods

Public:

- raw: variable to hold the raw model object from the chosen source library.

- name: string with the name of the variable.

- source: string with the name of the source library for this model.

- input\_type: string indicating the type of data being handled, i.e., either
  `image` or `video`.

- min\_size: `int` with the minimum size of image that a model accepts.

- num\_channels: int to indicate either a grayscale data type (`1`), or a color
  data type (`3`).

- Model parameters:

  - num\_layers: `int` with the number of layers
  - num\_params: `int` with the number of parameters

- Data for interfacing with the front-end and producing the data packet to be sent:

  - associated\_datasets: ?
  - paper: ?

- `__call__(self, input_data)`: a method to process a dataset sample, producing
  an output of type `???` (model/network structure dependent?). This is in order to be able to do:

```python
# Get a dataset and a single sample.
dataset = load_dataset("cifar")
dataset_sample = dataset[0]

# Get a model and process the data.
model = load_model("alexnet")
output = model(dataset_sample)
```

## Private:
