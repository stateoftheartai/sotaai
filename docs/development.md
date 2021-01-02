# Library Development

Documentation useful to set-up the library development environment in your machine

# Requirements

- As of now, **Python 3.8.0**. If you have a different version we recommend you to use [pyenv](https://github.com/pyenv/pyenv) and thus you can have different versions of Python installed in your machine and switch among them.
- As of now, we recommend you to have **pip==19.2.3**.

# Local Installation

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

# Contribution Flow

Step 0: Make sure your repos user email match the email you use in Jira, in the repo run:

```
git config user.email = "myemailusedinjira@mail.com"
```

Step 1: Have or create an issue in JIRA for the feature, fix, or in general "task" you're going to work on.

Step 2: Create a branch with the JIRA's issue ID:

```
git checkout -b SOTA-1338
```

Step 3: work on that branch and constantly push changes to its respective remote. Tha branch update will be automatically shown in the JIRA's issue as long as:

- The branch will be linked to JIRA as long as the branch names matches the Jira's issue ID e.g. SOTA-1338
- For a commit to be linked, the commit message must follow the specs in [Jira Smart Commits](https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/). For example:

```
update: my commit message
- SOTA-1338 #progress
```

For this to work, your git user.email at repo level must match a valid Jira user.

to its respective remote (as much as possible), do as many commits as you want. For the commit nami
