# Welcome to the `fpsl` Contributing Guide

This guide will give you an overview of the contribution workflow from opening an issue and creating a PR. To get an overview of the project, read the [module overview][fpsl].

## Issues

### Create a new issue

If you spot a bug, want to request a new functionality, or have a question on how to use the module, please [search if an issue already exists](https://github.com/BereauLab/fokker-planck-score-learning/issues). If a related issue does not exist, feel free to [open a new issue](https://github.com/BereauLab/fokker-planck-score-learning/issues/new/choose).

### Solve an issue

If you want to contribute and do not how, feel free to scan through the [existing issues](https://github.com/BereauLab/fokker-planck-score-learning/issues).

## Create a new pull request
### Create a fork

If you want to request a change, you first have to [fork the repository](https://github.com/BereauLab/fokker-planck-score-learning/fork).

### Setup a development environment

It is recommended to use `uv` to set up the development environment. Run inside your forked repository
```bash
# for cpu
uv sync --group docs

# for cuda12
uv sync --extra cuda --group docs

# install pre-commit
uv run pre-commit install

```


### Make changes and run tests

Apply your changes and check if you followed the coding style (PEP8) by running
```bash
uv run ruff check
```
All errors pointing to `./build/` can be neglected.

If you add a new function/method/class please ensure that you add a test function, as well. Running the test simply by
```bash
uv run tox
```
Ensure that the coverage does not decrease.

### Open a pull request

Now you are ready to open a pull request and please do not forget to add a description.
