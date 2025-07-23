"""Fokker-Planck Score Learning (FPSL): Score-based diffusion models for periodic data.

FPSL is a Python package for training and sampling from score-based denoising
diffusion models, with specialized support for periodic boundary conditions and
force-conditioned generation. The package implements the Fokker-Planck Score
Learning approach for learning score functions on toroidal (circular) domains.

This package contains the following main components:

- [**ddm:**][fpsl.ddm]
    This module implements the core [FPSL][fpsl.ddm.FPSL] class for learning the
    equilibrium free energy (PMF) from biased samples. It includes
    neural network architectures, noise scheduling, prior distributions, and
    force conditioning schedules for diffusion processes.
- [**datasets:**][fpsl.datasets]
    Collection of one-dimensional potential energy landscapes and biased-force
    variants for testing and benchmarking diffusion models.
- [**utils:**][fpsl.utils]
    Utility classes and functions including Gaussian mixture models, numerical
    integrators for stochastic differential equations, and base classes.

To get started, please have a look at the [tutorials](../tutorials).
"""

from importlib.metadata import version

from .ddm import FPSL

__version__ = version('fpsl')
