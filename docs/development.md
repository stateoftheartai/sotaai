# Library Development

Documentation useful to set-up the library development environment in your machine

# Requirements

- As of now, **Python 3.8.0**. If you have a different version we recommend you to use [pyenv](https://github.com/pyenv/pyenv) and thus you can have different versions of Python installed in your machine and switch among them.
- As of now, we recommend you to have **pip==19.2.3**.

# Development Dependencies

Create a new environment using `venv`. We recommend you to create the virutal environment directory inside the repository (it will be ignore it) and name it `.venv`:

```
python -m venv .venv
source .venv/bin/activate
```

Then, install development dependencies:

```
pip install -r requirements
```

Install pre-commit hooks:

```
pre-commit install
```

Make sure your local repository is working with our pre-commmit hooks:

```
pre-commit run --all-files
```

All hooks are to be in "Passed" status:

```
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...............................................................Passed
Check for added large files..............................................Passed
flake8...................................................................Passed
autopep8.................................................................Passed
cpplint..............................................(no files to check)Skipped
```

**Done**, now you are ready to start contributing and improving the `sotaai` library.
