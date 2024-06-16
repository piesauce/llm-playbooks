# LLM Playbooks


## Repository

### The `llm_playbooks` library

This library contains a collection of tools and utils for working with LLMs, that is used across our projects.


### `projects`

This folder contains individual subdirectories for each project related to LLM Playbooks. The content of each subdirectory is loosely organized, e.g.

* Notes for talks or writetups from papers, e.g. [`projects/presentations`](projects/presentations/)
* Collections Colab / Jupyter notebooks for LLM experiments, e.g. [`projects/getting_started`](projects/getting_started/)
* Focused Poetry projects including source code + notebooks, e.g. [`projects/hello_poetry`](projects/hello_poetry/)


## Setup

### Poetry

We will use [`poetry`](https://python-poetry.org/) extensively to manage dependencies. There are [several installation methods](https://python-poetry.org/docs/#installation), e.g. using [pipx](https://pipx.pypa.io/stable/installation/),

```shell
pipx install poetry
```

For an example of how to install the `llm_playbook` library alone, create any Python environment and simply run

```shell
poetry install
```

from this directory. For instructions on how to create Poetry projects for local experimentation, see [`projects/hello_poetry`](projects/hello_poetry/).


### Pre-commit

For contributing to LLM Playbooks, we highly suggest installing the provided [`pre-commit`](https://pre-commit.com/) hooks to help with code cleaning. Again, we suggest installing using [pipx](https://pipx.pypa.io/stable/installation/):

```shell
pipx install pre-commit
```

Then, install the hooks by running

```shell
pre-commit install
```

from this directory.
