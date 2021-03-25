# Computer Vision

We seek to include the following **libraries** to our computer vision
section:

- [Torchvision](https://github.com/pytorch/vision)
- [Keras](https://github.com/keras-team/keras)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Pretrainedmodels
  (Cadene)](https://github.com/Cadene/pretrained-models.pytorch)
- [Segmentation Models
  (PyTorch)](https://github.com/qubvel/segmentation_models.pytorch)
- [Segmentation Models
  (Keras)](https://github.com/qubvel/segmentation_models)
- [Image Super-Resolution](https://github.com/idealo/image-super-resolution)
- [MXNet](https://github.com/apache/incubator-mxnet)
- [GANs Keras](https://github.com/eriklindernoren/Keras-GAN)
- [GANs PyTorch](https://github.com/eriklindernoren/PyTorch-GAN)
- [Visual Question Answering](https://github.com/Cadene/vqa.pytorch)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Together, these libraries offer **models** and **datasets** for tasks spanning
from object detection and scene segmentation to image super-resolution and human
activity recognition.

As previously mentioned, the common interfaces between all libraries are still
under development. The progress for each of them with respect to the
libraries to be included is shown in the following table. Functions are denoted
as readily available (:white\_check\_mark:), in progress (:yellow\_circle:), and
implementation not yet started (:red\_circle:). In case a library does not offer
a functionality, a "not applicable" (N/A) is used.

|                            |    `load_model()`    |   `load_dataset()`   | `model_to_dataset()` |
| :------------------------: | :------------------: | :------------------: | :------------------: |
|        Torchvision         | :white\_check\_mark: | :white\_check\_mark: | :white\_check\_mark: |
|         Tensorflow         |         N/A          | :white\_check\_mark: | :white\_check\_mark: |
|           Keras            | :white\_check\_mark: | :white\_check\_mark: | :white\_check\_mark: |
| Pretrainedmodels (Cadene)  |   :yellow\_circle:   |   :yellow\_circle:   |   :yellow\_circle:   |
|           MXNet            |   :yellow\_circle:   |   :yellow\_circle:   |   :yellow\_circle:   |
|          fast.ai           |   :yellow\_circle:   |   :yellow\_circle:   |   :yellow\_circle:   |
| SegmentationModels pytorch |    :red\_circle:     |         N/A          |    :red\_circle:     |
|  SegmentationModels keras  |    :red\_circle:     |         N/A          |    :red\_circle:     |
|            ISR             |    :red\_circle:     |         N/A          |    :red\_circle:     |
|         Gans Keras         |    :red\_circle:     |         N/A          |    :red\_circle:     |
|        Gans Pytorch        |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     |
|            VQA             |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     |
|         Detectron2         |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     |

The goal is to be able to run a model of any of the available libraries with a
dataset of any of the available libraries. Thus, the following compatibility
matrix pictorially depicts which connections have already been successfully
established. The rows correspond to the models of a library, and the columns
correspond to the datasets. Hence, cell _(i,j)_ says that model from library _i_
can run with a dataset from library _j_.

|                              |     Torchvision      |        Keras         |      Tensorflow      |     MXNet     |      VQA      |
| :--------------------------: | :------------------: | :------------------: | :------------------: | :-----------: | :-----------: |
|         Torchvision          | :white\_check\_mark: | :white\_check\_mark: | :white\_check\_mark: | :red\_circle: | :red\_circle: |
|            Keras             | :white\_check\_mark: | :white\_check\_mark: | :white\_check\_mark: | :red\_circle: | :red\_circle: |
|       Pretrainedmodels       |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
| Segmentation\_models pytorch |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|  Segmentation\_models keras  |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|             ISR              |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|            MXNet             |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|          Gans Keras          |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|         Gans Pytorch         |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |
|             VQA              |    :red\_circle:     |    :red\_circle:     |    :red\_circle:     | :red\_circle: | :red\_circle: |

For a full list of the available Source Libraries, Models and Datasets go to [Stateoftheart.ai Dev
Library](https://www.stateoftheart.ai/dev-library)

## Specific Notes and Implementation Details

The main challenge in the training pipeline is to make a model compatible with a
dataset. For instance, specific requirements have to be fulfilled by the input
image so that the model can adequately process it. Additionally, it is common
that the last layer of the model requires modifications in accordance with the
dataset's properties, e.g., number of labels.

All of the above (e.g., compatibility checks and modifications thereof) is
encapsulated by the `model_to_dataset()` function, which does the following:

- Converts the dataset to a data type that a model understands. For example, a
  torchvision model accepts only tensors, hence a dataset obtained from
  tensorflow or mxnet will not immediately work in torchvision. Thus, this
  function converts a dataset to the type that the model accepts.
- A Computer Vision model has, among others, convolutional and pooling layers
  that reduce the image's dimension when passing through them. The image needs
  to be large enough so that the dimension stays positive, otherwise an error
  occurs. Hence, the function calculates the dimension reduction occurring
  inside the model, and resizes the image in case it is smaller than the
  minimum acceptable size.
- The output of the model must be in accordance with the dataset. For
  instance, in classification tasks, the number of categories varies from
  dataset to dataset. Appropriate changes to the last layer of the model have
  to be made so that it complies with the dataset at hand.
- As of now, we **do not** provide an API to **modify, tune, and train models**.
  However, we provide access to the raw instance as provided by the source
  library. This way the end user can modify, tune or train a model by using
  the source library API directly.
- Datasets come from only one source, if a dataset exists in multiple source
  libraries, we selected one of them by default. This source cannot be changed
  as of now.
- On the other hand, models come from multiple sources, if a model exists in
  multiple source libraries you can specify which source to use. This way you
  can modify, tune or train the model using the API you know the most.

## Overview

The code on this overview can be found in [examples/example1.py](https://github.com/stateoftheartai/sotaai/blob/master/examples/example1.py)

### Load a Model and Dataset

Import the main functions from the CV module:

```
from sotaai.cv import load_dataset, load_model, model_to_dataset
```

Instantiate your desired model and dataset:

```
model = load_model('ResNet152')
dataset = load_dataset('mnist')
```

You can also select the source of the model if desired `model = load_model('ResNet152, 'keras')`

A dataset is a dictionary where each key is a split and the
value is an object belonging to the `CvDataset` class. To get a specific
dataset split do:

```
dataset_split = dataset['train']
```

### Compatibility

A model implementation might not be directly compatible with a dataset. However
if both of them are theoretically compatible the `model_to_dataset` function
do the work to make both implementations compatible:

```
model, dataset_split = model_to_dataset(model, dataset_split)
```

This function will return a new model and dataset instances whose
implementations where modified so that both can run against each other.

### Modify, Tune or Train

As mentioned above, currently we do not provide an API to modify, tune or
train models. However you can access the raw instance as provided by the source
library:

```
model = load_model('ResNet152', 'keras')
source_model = model.raw

...
Here you can make any modifications using source_model which for this case
is the Keras instance
...

```

Once the model was modified, tuned or trained as desired, you should pass it
back:

```
model.update_raw_model(source_model)
```

### Get Predictions

To get predictions, create a batch of data:

```
batch = []
batch_size = 10

for i, item in enumerate(dataset_split):

  if i == batch_size:
    break

  image_sample = item['image']
  batch.append(image_sample)

batch = np.array(batch)
```

Obtain predictions for the given batch of data:

```
predictions = model(batch)
```
