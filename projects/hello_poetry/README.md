# Hello Poetry!

This is a template Poetry project that is intended to be replicated to help others start off with their own projects.

## Getting started with Poetry

### Setting up the project

There are a few things to be done when starting out a new project:

1. Copy and rename the folder and source code subfolder, e.g.

    ```shell
    cd ..  # projects folder
    cp -r hello_world my_project  # rename my_project as you like
    cd my_project  # project folder
    mv hello_world my_project  # source code folder
    ```

2. The main configuration of any Poetry projects happens in `pyproject.toml`, we first want to make sure the project metadata is correct, e.g. open up `my_project/pyproject.toml` and edit the following fields:

    ```toml
    [tool.poetry]
    name = "my_project"
    version = "0.1.0"
    description = "My Project"
    authors = ["John Doe <john.doe@example.com>"]
    readme = "README.md"
    ```

3. Next let's make sure we have an environment set up for the project. You can use any Python environment as long as the Python version is satisfies the requirements in `pyproject.toml`, but often a good idea is to create a local environment just for the project. For example, you can install a [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) for Python 3.11 in the local subdirectory `.venv` and activate it like this:

    ```shell
    virtualenv .venv -p 3.11 && source .venv/bin/activate
    ```

    (Alternatively) you can also do the same with [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

    ```shell
    conda create -n myenv python=3.11  # rename myenv as you like
    conda activate myenv
    ```

4. Then we will install the Poetry project in the activated environment:

    ```shell
    poetry lock
    poetry install
    ```

5. You should now be able to use the project library as well as any installed dependencies simply by importing them, e.g.

    ```python
    from my_project.hello import hello
    from llm_playbooks.hello import hello as llm_hello

    hello()  # Hello Poetry!
    llm_hello()  # Hello LLMs!
    ```

### Managing dependencies

Now you can add and remove dependencies and Poetry will keep track of whether they are valid, e.g.

```shell
poetry add pandas  # pandas can be any pip package
poetry remove pandas
```

You can also directly modify them in `pyproject.toml`, as long as you follow your modifications up with

```shell
poetry lock
poetry install
```

We are also able to add dependencies hosted on GitHub as well as local projects, e.g. the `llm_playbooks` project, which you can remove if you don't need.

Notice that in `pyproject.toml`, we make a distinction between dependencies that are needed for the project code to run, and those that are there for convenience, e.g. `ipython`. The latter are put into a separate `dev` group. To add such a dependency, e.g. run

```shell
poetry add ipython --group dev
```

Finally to activate the environment that the Poetry project was installed into, you can simply run

```shell
poetry shell
```
