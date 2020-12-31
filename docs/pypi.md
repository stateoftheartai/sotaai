# PyPI

Documentation useful to manage the build and deployment of the library to [PyPI](https://pypi.org/)

## Overview

- Everything is managed under stateoftheart.ai official accounts in PyPI for both test/production PyPI indexes.
- To manage PyPI and thus being able to follow this docs, you need to be an authorized contributor or have the required global credentials.

## Build / Deployment

To build a new distribution from source code run:

```
python setup.py sdist bdist_wheel
```

To deploy a new destribution to PyPI:

```
python -m twine upload --repository testpypi dist/*
```

**important**: avoid `--repository testpypi` to deploy to PyPI production index. For more information refer to [PyPI Docs](https://packaging.python.org/tutorials/packaging-projects/).