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

______________________________________________________________________

## Overview of the platform

We provide the following **main functions** to train any model on any
available dataset and get predictions:

- `model = load_model('model-name')`: creates an instance of a model from any source library.
- `dataset = load_dataset('dataset-name')`: creates an instance of a dataset from any source library.
- `model, dataset = model_to_dataset(model,dataset)`: given a model and a dataset, this pairing function
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

## Areas Supported

Currently we support the areas listed below. To see more information and
documentation of a specific area feel free to navigate to the respective section:

- [Computer Vision](https://github.com/stateoftheartai/sotaai-beta/blob/master/docs/sotaai/cv/README.md)
- Natural Language Processing **\[WORK IN PROGRESS\]**
- Neurosymbolic Reasoning **\[WORK IN PROGRESS\]**
- Reinforcement Learning **\[WORK IN PROGRESS\]**
- Robotics **\[WORK IN PROGRESS\]**

______________________________________________________________________

## Installation

### Requirements

- Python3 is required. Recommended to use **Python \<= 3.8.0**.
- As a best practice, we recommend the use of
  [pyenv](https://github.com/pyenv/pyenv), this way you can have different
  versions of Python installed in your machine and switch among them, being able
  to meet any Python version requirement.
- Regarding the installation process, we have detected some issues with
  the `pip` version. As of now the library installation has been fully tested
  and proved working with the following versions of pip: `19.2.3`, `21.0.1`
  (latest as of March 2021)

### PIP Installation

The library can be installed through `pip`. However, since `sotaai` depends on many
libraries, we split the library into subpackages so that end users can install
only the subpackage and dependencies they require, to install the desired
subpackage use the following instructions:

For Computer Vision:

```
pip install sotaai[cv]
```

For Natural Language Processing:

```
pip install sotaai[nlp]
```

For Neurosymbolic Reasoning:

```
pip install sotaai[neuro]
```

For Reinforcement Learning:

```
pip install sotaai[rl]

```

For Reinforcement Learning it is also necessary to install some dependencies
independently:

```
pip install pybullet-gym@git+https://github.com/benelot/pybullet-gym
```

**Important:** if you make use of `zsh` you have to add a `\` before each bracket (\[\])
in order for `zsh` to understand the command e.g. `pip install sotaai\[cv\]`.

### Installation Issues

#### Tensorflow SSL: CERTIFICATE\_VERIFY\_FAILED

This is a [known issue in Tensorlow](https://github.com/tensorflow/models/issues/5871).
It is documented to occur in MacOSX. As of now, a working solution is to manually install
certificates by running the `Install Certificates.command` that exists inside the python
installation directory:

```
Applications/Python X.Y/Install Certificates.command
```

Replace `Python X.Y` with your installed version e.g. `Python 3.8`. Take into
account that the location of this file might depend on your Python and/or OS version.
