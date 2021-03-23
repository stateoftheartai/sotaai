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

## Installation and quick start

This library is provided as a pip module to be installed locally.

```
pip install sotaai
```

______________________________________________________________________

______________________________________________________________________

# Overview of the platform

We provide the following **main functions** to train any model on any
available dataset:

- `load_model()`: creates an instance of a model from any source library.
- `load_dataset()`: creates an instance of a dataset from any source library.
- `model_to_dataset()`: given a model and a dataset, this pairing function
  adjusts both objects to ensure they are compatible. Compatibility means the
  model is ready to predict data given dataset subset: `predicitons = model(dataset)`.

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
