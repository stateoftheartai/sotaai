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

Return Type: `FastaiModel`

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

### `FastaiModel`

As stated [here](https://fastai1.fast.ai/vision.models.html),
most of the Fastai models come from Torch, although they also provide their own models but those are a few).
As of now, the returned model is to be a
[Torchvision model](https://pytorch.org/docs/stable/torchvision/models.html) and this returned instance
will belong to a custom class e.g. `fastai.vision.models.xresnet.XResNet` or `torchvision.models.vgg.VGG`.
However, at the end those custom classes inherit from
[nn.Module Class](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=nn%20module#torch.nn.Module).

Notes:

- Available fastai models and its parameters are documented
  [here](https://pytorch.org/docs/stable/torchvision/models.html). As of now, all of them come from Torch.
- All of those pre-built models were coded using `nn.Module` as briefly documented
  [here](https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-module)
