<div align="center">
  <img
    src="https://raw.githubusercontent.com/BereauLab/fokker-planck-score-learning/refs/heads/main/docs/logo.svg" width="200"
  />

  <p>
    <a href="https://arxiv.org/abs/2506.15653" alt="arXiv">
      <img src="https://img.shields.io/badge/arXiv-2506.15653-red" /></a>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide" >
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://github.com/BereauLab/fokker-planck-score-learning/actions/workflows/pytest.yml" alt="Pytest" >
        <img src="https://github.com/BereauLab/fokker-planck-score-learning/actions/workflows/pytest.yml/badge.svg?branch=main" /></a>
    <a href="https://pypi.org/project/fpsl" alt="PyPI" >
        <img src="https://img.shields.io/pypi/v/fpsl" /></a>
    <a href="https://pepy.tech/project/fpsl" alt="Downloads" >
        <img src="https://pepy.tech/badge/fpsl" /></a>
    <a href="https://img.shields.io/pypi/pyversions/fpsl" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/fpsl" /></a>
    <a href="https://github.com/BereauLab/fokker-planck-score-learning/blob/main/LICENSE" alt="PyPI - License" >
        <img src="https://img.shields.io/pypi/l/fpsl" /></a>
    <a href="https://bereaulab.github.io/fokker-planck-score-learning" alt="Doc" >
        <img src="https://img.shields.io/badge/mkdocs-Documentation-brightgreen" /></a>
  </p>

  <p>
    <a href="https://bereaulab.github.io/fokker-planck-score-learning">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#Installation">Installation</a>
  </p>
</div>

# Fokker-Planck Score Learning: Efficient Free-Energy Estimation Under Periodic Boundary Conditions

This package contains a proof-of-concept implementation of the Fokker-Planck score learning approach.

This package is published in:
> **Fokker-Planck Score Learning: Efficient Free-Energy Estimation Under Periodic Boundary Conditions**,  
> D. Nagel, and T. Bereau,  
> *arXiv* **2025**,  
> doi: [10.48550/arXiv.2506.15653](https://doi.org/10.48550/arXiv.2506.15653)

We kindly ask you to cite this article in case you use this software package for published works.

## Features
- TBA
- [Documentation](https://bereaulab.github.io/fokker-planck-score-learning) including tutorials
- Supports Python 3.10-3.13

## Getting started
### Installation
The package is called `fpsl` and will be soon available via [PyPI](https://pypi.org/project/fpsl). To install it, simply call:
```bash
python3 -m pip install fpsl
```
For now, you can install it from github. Download the repo and setup an env with with `fpsl` installed with `uv`. If you do not have `uv` you can get it [here](https://docs.astral.sh/uv/).
```
uv sync --extra cuda  # if you have an Nvidia GPU
```

### Usage

Add here a short example.

```python
import fpsl

ddm = fps.DrivenDDM(
    sigma_min=1e-3,
    symmetric=True,
    fourier_features=4,
    ...,
)
# load x position of MD trajectory and forces f
ddm.train(
    ...
)
...
```
