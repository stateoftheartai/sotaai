# Library Development

Documentation useful to set-up the library development environment in your
machine.

## Requirements

- As of now, **Python 3.8.0**. If you have a different version we recommend you
  to use [pyenv](https://github.com/pyenv/pyenv) and thus you can have different
  versions of Python installed in your machine and switch among them.
- As of now, we recommend you to have **pip==19.2.3**.

## Local Installation

Create a new environment using `venv`. We recommend you to create the virutal
environment directory inside the repository (it will be ignore it) and name it
`.venv`:

```
python -m venv .venv
source .venv/bin/activate
```

Then, install development dependencies:

```
pip install -r requirements.txt
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

**Done**, now you are ready to start contributing and improving the `sotaai`
library.

## Visual Studio Code

In case you use Visual Studio Code as your IDE, this `.vscode/settings.json` might work for you:

```json
{
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "build": true,
    "dist": true,
    "**.egg-info": true
  },
  "python.pythonPath": ".venv/bin/python",
  "python.formatting.provider": "yapf",
  "python.linting.pylintEnabled": true,
  "python.linting.enabled": true,
  "python.linting.pylintArgs": [
    "--init-hook",
    "import sys; sys.path.insert(0, '../sotaai')"
  ]
}
```

This configuration is going to apply the default formatting and linting we use.

## Contribution Flow

**Step 1:** Make sure your local repo email match the email you use in Jira, in
the repo run:

```
git config user.email "myemailusedinjira@mail.com"
```

**Step 2:** Have or create an issue in JIRA for the feature your going to work
on. By feature we mean: a fix, feature, improve, refactor, to be added to the
current code base.

**Step 3:** Make sure to have `master` branch up to date, and then create a
branch out of `master` and name it after the Jira issue ID. This is going to be
refered as the "feature branch":

```
git checkout -b SOTA-1338
```

**Step 4:** work on the feature branch and constantly push changes to its
respective remote. For naming commits, follow [these
guidelines](https://github.com/stateoftheartai/sotaai/blob/master/docs/common/commits.md).
All updates are to be automatically reflected in Jira as long as:

- The feature branch name exactly matches the Jira's issue ID e.g. SOTA-1338
- Each of the feature branch commits meet the specs in [Jira Smart
  Commits](https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/)
  at least in one line of the title/body of the commit. For example:

To put the issue in progress:

```
update: add some updates to development.md
- SOTA-1338 #progress
```

To put the Jira issue in progress and add a comment to the Jira issue as well:

```
update: add some updates to development.md
- SOTA-1338 #progress #comment This is my comment
```

Valid transitions are: `#progress`, `#review`, `#test`, `#done`

**Step 5:** once the feature is finished create a Pull Request and assign it to
a Reviewer. The Pull Request will trigger testing jobs, if those succced and the
Reviewer approves the code, then the feature branch is to be merged into
`master` by the Reviewer. For this, all feature branch commits will be squashed
into one commit, this commit is to be rename it by the reviewer to be consistent
with the feature being merged, and finally merged. This way `master` branch will
have only one commit per feature, which was tested, and with a consistent name.

**Step 6:** when the Pull Request is accepted and thus the feature branch is
merged, this feature branch is automatically deleted from the remote, however
the developer must delete its local branch from its machine and also pull
`master` changes that might include the new feature:

```
git branch -D SOTA-1338
git checkout master
git pull origin master
```

**Step 7:** the developer can now create a new branch out of `master` and start
working in a new feature.
