# Getting Started with `fpsl`

## Introduction
If you are interested in learning free energy landscapes from non-equilibrium data using score-based diffusion models, this tutorial will provide you with a step-by-step guide to use `fpsl` (Fokker-Planck Score Learning), a Python package designed specifically for this purpose. FPSL employs the steady-state solution of the Fokker-Planck equation as an ansatz in the denoising score learning scheme, making it particularly effective for periodic boundary conditions and force-conditioned generation.

Whether you are a beginner to score-based generative modeling or an experienced researcher in statistical mechanics, this tutorial will help you get up and running with FPSL and enable you to explore free energy landscapes of complex systems.

## Installation
Before you can use the FPSL package, you will need to install it on your machine. The package is available on PyPI, so you can install it using pip.

To install the package using pip, open a terminal or command prompt and enter the following command:

```bash
python -m pip install fpsl
```

Alternatively, you can install the development version directly from GitHub:

```bash
python -m pip install git+https://github.com/BereauLab/fokker-planck-score-learning.git
```

Once you have installed the package, you are ready to explore the tutorial!

## Tutorial

The tutorial is structured into the following sections:

- [**Theoretical Background:**](theory.md)
    This section provides a theoretical background on the FPSL approach, including the Fokker-Planck equation, score-based generative modeling, and the specific challenges of periodic boundary conditions.
- [**Interactive Tutorial Notebook:**](fpsl.ipynb)
    This comprehensive Jupyter notebook walks you through the complete FPSL workflow using toy model examples from the main paper. You'll learn how to: Set up and configure FPSL models, work with built-in potential energy datasets, train diffusion models on non-equilibrium data, generate samples and reconstruct free energy profiles, and analyze and visualize results.

## Getting Help
In case you encounter any issues or have questions while using FPSL, there are several resources available to help you:

- **Paper**: Read the original research article on [arXiv:2506.15653](https://arxiv.org/abs/2506.15653)
- **Issues**: Report bugs or request features on the [GitHub repository](https://github.com/BereauLab/fokker-planck-score-learning)

## Conclusion
FPSL represents a powerful approach to learning free energy landscapes from non-equilibrium data, particularly for systems with periodic boundary conditions. By the end of this tutorial, you will have a solid understanding of both the theoretical foundations and practical implementation of FPSL, enabling you to apply these techniques to your own research problems in statistical mechanics, molecular dynamics, and beyond.