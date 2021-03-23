# Stateoftheart AI

Welcome! The objective of this repository is to simplify the usage of the many
machine learning libraries available out there.

**How do we do this?**

- We collect all models and datasets into a single, centralized place (here).
- We offer a same and single function to instantiate a model or a dataset,
  irrespective of the source library (`load_model` and `load_dataset`).
- We provide the functionality to connect any model/dataset pair for evaluation
  and training purposes via a `model_to_dataset` function, irrespective of the
  source library for either of them (naturally with intrinsic limitations, e.g.,
  cannot pair a CV model with a RL environement).
- We provide the functionality to obtain predictions i.e. evaluating a model on
  a given dataset `predictions = model(dataset)`

## Installation and quick start

This library is provided as a pip module to be installed locally.

```
pip install sotaai
```

______________________________________________________________________

______________________________________________________________________

# Overview of the platform

We provide the following **main functions** to train any model on any
available dataset and get predictions:

- `model = load_model()`: creates an instance of a model from any source library.
- `dataset = load_dataset()`: creates an instance of a dataset from any source library.
- `model, dataset = model_to_dataset()`: given a model and a dataset, this pairing function
  adjusts both objects to ensure they are compatible.
- `predictions = model(dataset)`: each model instance is callable, thus you can get
  predictions by calling the model and passing the dataset to get predictions
  from.

Depending on the type of model, dataset, area, and source library, the inner
workings of the aforementioned functions can wildly vary. Additionally, extra
functions might be needed to fully establish the pipeline between models and
datasets (e.g., in NLP where we need to account for tokenizers and embeddings).

The main contribution of this library is the development of these interfaces
between external libraries warranted for the interconnection of models from one
source and datasets from another, with the ultimate objective being to obtain a
unique, simplified and unified approach to interacting with and leveraging all
of the existing machine learning libraries.

In the following sections we document the progress of this massive undertaking,
explicitly showing where the connections between libraries have already been
established, and where is more work needed.

______________________________________________________________________

______________________________________________________________________

# Areas Supported

Currently we support the areas listed below. To see more information and
documentation of a specific area navigate to the respective section:

- [Computer Vision](https://github.com/stateoftheartai/sotaai/blob/SOTA-1655/docs/sotaai/cv/overview.md)
- Natural Language Process **\[WORK IN PROGRESS\]**
- Neurosymbolic Reasoning **\[WORK IN PROGRESS\]**
- Reinforcement Learning **\[WORK IN PROGRESS\]**
- Robotics **\[WORK IN PROGRESS\]**
