# PyPI

Documentation useful to manage and run tests.

## Unittest

Unittests will test the current version installed in your
virtual environment of the `sotaai` library, thus,
first make sure you have installed the right version of it
i.e. the one you want to test:

- Install the version from your local code base
  `pip install -e .` in case you want to test
  your latest local changes.
- Install a specific version from PyPI
  (test or real indexes as desired)
  `pip install sotaai==x.y.z` or
  `pip install --index-url https://test.pypi.org/simple/ --no-deps sotaai`
  in case you want to test and already-deployed version.

To execute the unittest move to the `tests` directory:

```
cd tests
```

And execute the tests:

```
python -m unittest
```
