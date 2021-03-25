# PyPI

Documentation useful to manage the build and deployment
of the library to [PyPI](https://pypi.org/)

## Overview

- Everything is managed under stateoftheart.ai official accounts
  in PyPI for both test/production PyPI indexes.
- To follow this docs and manage sotaai PyPI packages, you need
  to be an authorized contributor or have the required admin credentials.

## Requirements

- Make sure to have the required admin credentials, or have register
  your own account in PyPI.
- If you have your own PyPI account, make sure your account is added
  as a contributor of the stateoftheai PyPI packages
  (for each environment of Test PyPI and Real PyPI packages, you need
  an account).

## Auth

As a contributor, you need to configure your local machine to be able
to deploy new distributions to PyPI.

First, on your own PyPI account (in the desired environment)
create an Access Token ([follow this docs](https://pypi.org/help/#apitoken))

Then in your home directory create a `~/.pypirc` file with
the following content:

```
[distutils]
  index-servers =
    pypi
    testpypi

[pypi]
  username = __token__
  password = <token-value>

[testpypi]
  username = __token__
  password = <token-value>
```

Replace `<token-value>` with the respective's environment token.
If you will only use one environment, then remove the other one
from the file's content.

With this configuration on place, you will be able to run the `twine` command
to deploy new versions of the sotaai library.

## Build / Deployment

To build a new distribution from source code run:

```
python setup.py sdist bdist_wheel
```

To deploy the new destribution to PyPI:

```
python -m twine upload --repository testpypi dist/*
```

**important**: avoid `--repository testpypi` to deploy to PyPI production index.
For more information refer to [PyPI Docs](https://packaging.python.org/tutorials/packaging-projects/).

## Installation

Once the library is deployed to a PyPI index (whether test or production)
you can install it:

```
pip install --index-url https://test.pypi.org/simple/ --no-deps sotaai
```

**important**: avoid `--index-url https://test.pypi.org/simple/`
and `--no-deps` to install from PyPI real/production index.
