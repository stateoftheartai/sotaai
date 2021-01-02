# Naming Commits

## Objective

Provide guidelines on how to name commits

## Guidelines

### General

**Do** always write commit messages in **ENGLISH**

**Do** always adhere your commit message to the following format:

```
<commit-type>: <commit-subject>
- ISSUE-ID <if any>
- <commit-detail-1>
- <commit-detail-2>
.
.
.
- <commit-detail-n>
```

**If supported, do** add [Jira Smart Commits](https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/) as part of a commit-detail. Not as the commit-subject.

**Do** use imperative mode for commit subject and details

**Examples**

```
Header toolbar functionality updated <== DO NOT

Fixed bug with firebase authentication <== DO NOT
```

```
Update header toolbar functionality <== DO

Fix bug when executing firebase authentication <== DO
```

**Examples**

```
update: add initial implementation of Keras
- SOTA-353
- Initial configuration of datasets and models
- Initial configuration of CI pipeline
```

```
refactor: Refactor abstract classes
- SOTA-123
- Rename utils for abstractions
- Add .abstract sufix to file names
```

To help writting commits in imperative mode. Notice that a properly formed commit subject or detail should always be able to complete the sentence: **If applied, this commit will** _<your commit subject or detail goes here_. For example:

- **If applied, this commit will** _update header toolbar functionality_
- **If applied, this commit will** _fix bug when executing firebase authentication_

### Commit Type

**Do** always add a **commit-type** to the commit message

- **Do** use `commit-type=update` when your change has to do with updates in code business logic, whether a perfomance improve, new features, etc.
  **Examples:** adding a new button; create a new cron task; add third-party login functionality; add a new librar method, etc.
- **Do** use `commit-type=fix` when your change is entirely related to a fix of a existing code base.
  **Examples:** fixing any kind of bug, defect, anomalous behavior, etc.
- **Do** use `commit-type=refactor` when your change is entirely related to a [refactor](https://pascal.computer.org/sev_display/search.action;jsessionid=2cd892e19975e94dc862f32b95c0) of a code base.
  **Examples:** change folder or files structures; rename variables; apply a standard, code convention, good practice, etc; delete or clean legacy code; add comments, etc.
- **Do** use `commit-type=merge` when this commit is a merge. **Do not** use the git's default merge commit message. Git default merge message is cumbersome, please **do** add a merge summarized message insted.
  **Examples:** all merge commits.
  **Important:** if you **do know** [how to use git rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing), then use rebase instead of merge. This helps keep a cleaner git history.

**Examples:**

```
update: update load_dataset to add some exceptions for Tensorflow

fix: fix bug while batching datasets

merge: merge latest changes into feature and solve conflicts
```

**Why:**

- Having a `<commit-type>` will allow us to rapidly identify the general objective of a commit.
- Having a `<commit-type>` will allow us to eventually have metrics out of the Git history.

### Commit Subject

**Do** start `<commit-subject>` with Uppercase.

**Do** add a descriptive, clear and succint `<commit-subject>` always starting with Uppercase

**Do** not exceed up to **50 characters** in the `<commit-subject>`

**Do** not end the `<commit-subject>` with a dot: `.`

**Examples:**

```
refactor: apply pep8 standards to unittests

fix: fix abstract classes layout

update: add twine authentication to easy deployment to PyPI
```

**Why:**

- A developer must easily identify the purpose of a commit

### Commit Details

**Do optionally** add details in the commit Body in case you want to explain in more detail the commit's purpose.

**Do** add commit details in a list format.

**Do** use commit details when you think it will be relevant for you or anyone in the team.

**Do** not use commit details to explain which files where updated, variables renamed, etc. If this is relevant for a certain developer then it can be obtained using ` git diff`.

**Do** add each commit detail in its own line starting with a dash: `-`

**Do** try not to exceed up to **70 chars** in each `<detail-content>`. Git automatically wrap text on 72 characters so exceeding might uglify the output when displaying the Git history.

**Do** start all `<commit-detail>`s with Uppercase.

**Do** not end each `<commit-detail>` with a dot: `.`

**Why:**

- Having useful details can help developers rapidly identify relevant information of a commit in general.
- Having body separated from details can allow you to check the git history in different ways. For instance, applying a `git log --oneline` will allow you to see all git Subjects in a more compact way (without loosing meaning) perhaps when looking for a certain commit or if you just want to see the git history in graph mode to know how the project is evolving.

**Important:** depending on wheter you use a Git Client or you use Git CLI directly take this into account:

- Some Git clients might prompt you for two fields: **commit subject** and **commit body**. If this is the case feel free to add each field just respecting these guidelines.
- If you use Git CLI or your client only prompts you for one field (the **commit message**), then take into consideration that Git by default will parse the body and the subject based on the first break line. Hence, make sure to **add a break line** just right after your commit subject. After this break line you can start writting the body.

**Examples**

```
update: use python callback to provide end-users with some way of customization

- Be more flexible to end-users to add custom behavior when implementing sotaai models
- Not all models provide this functionality, only those from Torch and Keras as of now.
```

```
fix: fix unauthorized notification when download datasets from Keras.

- Add ssl authentication library to support direct authentication without the need of user typing
```
