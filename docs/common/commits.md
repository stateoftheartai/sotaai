# Naming Commits

These guidelines provide specifications on how to name commits to achieve a
result like the following:

```
update: implement Github Actions to automate ci build/test

- Add github actions configuration initially to run unittests
- Refactor test directory to meet actions requirements
- SOTA-1233 #review
```

## In general...

**Do** always adhere your commit message to the following format:

```
<commit-type>: <commit-subject>
- <commit-detail-1>
- <commit-detail-2>
.
.
.
- SmartCommit <if any>
```

**Do** always write commit messages in **ENGLISH**

**Do** use imperative mode for commit subject and details

- **Do not:** `Keras authenitcation flow updated`
- **Do:** `Update keras authentication flow`

- **Do not:** `Fixed bug with tensorflow datasets configuration`
- **Do:** `Fix bug on tensorflow datasets configuration`

To help writting commits in imperative mode. Notice that a properly formed
commit subject or detail should always be able to complete the sentence: **If
applied, this commit will** _<your commit subject or detail goes here_. For
example:

- **If applied, this commit will** _update header toolbar functionality_
- **If applied, this commit will** _fix bug when executing firebase
  authentication_

**When possible, do** add [Smart
Commits](https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/)
as the last commit-detail.

## Regarding commit type...

**Do** always add a **commit-type** to the commit message

- **Do** use `commit-type=update` when your change has to do with updates in
  code business logic, whether a perfomance improve, new features, etc.
  **Examples:** adding a new button; create a new cron task; add third-party
  login functionality; add a new librar method, etc.
- **Do** use `commit-type=fix` when your change is entirely related to a fix of
  a existing code base. **Examples:** fixing any kind of bug, defect, anomalous
  behavior, etc.
- **Do** use `commit-type=refactor` when your change is entirely related to a
  [refactor](https://pascal.computer.org/sev_display/search.action;jsessionid=2cd892e19975e94dc862f32b95c0)
  of a code base. **Examples:** change folder or files structures; rename
  variables; apply a standard, code convention, good practice, etc; delete or
  clean legacy code; add comments, etc.
- **Do** use `commit-type=merge` when this commit is a merge. **Do not** use the
  git's default merge commit message. Git default merge message is cumbersome,
  please **do** add a merge summarized message insted. **Examples:** all merge
  commits. **Important:** if you **do know** [how to use git
  rebase](https://www.atlassian.com/git/tutorials/merging-vs-rebasing), then use
  rebase instead of merge. This helps keep a cleaner git history.

Examples of commits with commit types:

```
update: update load_dataset to add some exceptions for Tensorflow

fix: fix bug while batching datasets

merge: merge latest changes into feature and solve conflicts
```

## Regarding commit subject...

**Do** add a descriptive, clear and succint `<commit-subject>`

**Do not** exceed up to **50 characters** in the `<commit-subject>`

**Do not** end the `<commit-subject>` with a dot: `.`

Examples of commits with commit subjects:

```
refactor: apply pep8 standards to unittests

fix: fix abstract classes layout

update: add twine authentication to easy deployment to PyPI
```

## Regarding commit details...

**Do optionally** add details in the commit Body in case you want to explain in
more detail the commit's purpose.

**Do** add commit details in a list format.

**Do** use commit details when you think it will be relevant for you or anyone
in the team.

**Do not** use commit details to explain which files where updated, variables
renamed, etc. If this is relevant for a certain developer then it can be
obtained using ` git diff`.

**Do** add each commit detail in its own line starting with a dash: `-`

**Do not** exceed up to **70 chars** in each `<detail-content>`. Git
automatically wrap text on 72 characters so exceeding might uglify the output
when displaying the Git history.

**Do** start all `<commit-detail>`s with Uppercase.

**Do not** end each `<commit-detail>` with a dot: `.`

**Important:** depending on wheter you use a Git Client or you use Git CLI
directly take this into account:

- Some Git clients might prompt you for two fields: **commit subject** and
  **commit body**. If this is the case feel free to add each field just
  respecting these guidelines.
- If you use Git CLI or your client only prompts you for one field (the **commit
  message**), then take into consideration that Git by default will parse the
  body and the subject based on the first break line. Hence, make sure to **add
  a break line** just right after your commit subject. After this break line you
  can start writting the body.

Examples of commits with commit details:

```
update: use python callback to provide end-users with some way of customization

- Be more flexible to end-users to add custom behavior when implementing sotaai
  models
- Not all models provide this functionality, only those from Torch and Keras as
  of now
```

```
fix: fix unauthorized notification when download datasets from Keras.

- Add ssl authentication library to support direct authentication without the
  need of user typing
```
