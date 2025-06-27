"""
This submodule provides a suite of one-dimensional potential energy landscapes and their
corresponding biased-force variants for use in score-based and diffusion-based modeling
experiments. Each dataset class encapsulates a specific analytic potential function and
supporting machinery to generate samples, compute energies, and (where applicable)
apply an external biasing force.

Classes:
    WPotential1D
        A symmetric double-well (W-shaped) potential in one dimension.
    BiasedForceWPotential1D
        The WPotential1D with an added constant biasing force term.
    ToyMembranePotential1D
        A simple membrane-like potential featuring a central barrier and flanking wells.
    BiasedForceToyMembranePotential1D
        The ToyMembranePotential1D augmented with an external biasing force.
    ToyMembrane2Potential1D
        An extended membrane potential with two barriers and three wells.
    BiasedForceToyMembrane2Potential1D
        The ToyMembrane2Potential1D with an additional biasing force.
    ToyMembrane3Potential1D
        A higher-order membrane potential featuring three barriers and four wells.
    BiasedForceToyMembrane3Potential1D
        The ToyMembrane3Potential1D augmented with an external biasing force.

All classes expose a consistent interface for:
    • Sampling data points from the potential's Boltzmann distribution.
    • Computing potential energies and (optional) biasing forces.
    • Integrating seamlessly with score-based learning workflows.

Usage example:

    dataset = WPotential1D(num_samples=10000, temperature=1.0)
    x, energy = dataset.sample()

"""

__all__ = [
    'WPotential1D',
    'BiasedForceWPotential1D',
    'ToyMembranePotential1D',
    'BiasedForceToyMembranePotential1D',
    'ToyMembrane2Potential1D',
    'BiasedForceToyMembrane2Potential1D',
    'ToyMembrane3Potential1D',
    'BiasedForceToyMembrane3Potential1D',
]

from .datasets import (
    BiasedForceWPotential1D,
    BiasedForceToyMembranePotential1D,
    BiasedForceToyMembrane2Potential1D,
    BiasedForceToyMembrane3Potential1D,
    WPotential1D,
    ToyMembranePotential1D,
    ToyMembrane2Potential1D,
    ToyMembrane3Potential1D,
)
