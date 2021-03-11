# Standardizing CV through abstractions

## Standardized datasets attributes and methods

- `raw`: variable to hold the raw dataset object from the chosen source library.

- `name`: string with the name of the dataset.

- `source`: string with the name of the source library for this dataset.

- `data_type`: string indicating the type of data contained in the dataset,
  i.e., either `image` or `video`.

- `split_name`: string indicating with the name of the corresponding dataset's
  split.

- `tasks`: array of strings with the name of the supported tasks.

- `size`: int with the size of the split (number of elements inside the split).

- `shape`: a tuple holding the shape of the split's data in the format (width,
  height, num of channels).

- For image classification or detection tasks:

  - `classes`: array with the number of classes.

  - `classes_names`: array with the corresponding names of classes.

- For segmentation tasks:

  - `pixel_classes`: array containing each of the pixel classes (number)

  - `pixel_classes_names`: array containing each of the pixel classes names
    (strings)

- For image captioning tasks:

  - `captions`: array with captions?

- For visual question answering tasks:

  - `annotations`: array with annotations?

  - `vocab`: array with vocabulary?

## Standardized model attributes and methods

Public:

- `raw`: variable to hold the raw model object from the chosen source library.

- `name`: string with the name of the model.

- `source`: string with the name of the source library for this model.

- `data_type`: string indicating the type of data being handled, i.e., either
  `image` or `video`.

- `min_size`: `int` with the minimum size of image that a model accepts.

- `num_channels`: int to indicate either a grayscale data type (`1`), or a color
  data type (`3`).

- Model parameters:

  - `num_layers`: `int` with the number of layers
  - `num_params`: `int` with the number of parameters

- Data for interfacing with the front-end and producing the data packet to be sent:

  - `associated_datasets`: ?
  - `paper`: ?

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
