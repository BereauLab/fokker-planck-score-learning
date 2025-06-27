# Welcome to the `fpsl` Maintenance Guideline

This guide will give you an overview of how to publish a new version of fpsl. In the following we will refer to the new version as `v1.*.*`. This needs to be substituted to the current version, e.g. `v1.1.3`.

## Prepare New Release

1. the version number in `pyproject.toml` are bumped,
1. a new tag is created via `git tag v1.*.*` and pushed `git push --tags`, and
1. the changelog includes the new tag and all changes of the release.

## Build and Upload to PyPI

Todo: Describe how to build and upload to PyPI with using `uv` and Github action.