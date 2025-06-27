"""Dataset classes for sampling from 1D toy potentials.

This submodule provides interfaces and concrete implementations for
sampling from various 1D potentials using (biased) Euler-Maruyama
integrators.  Supported potentials include the W-potential and several
“toy membrane” variants.  Users can obtain sample trajectories or plot
the potential energy profile.

Classes
-------
DataSet
    Abstract base defining the dataset interface.
WPotential1D
    Dataset for the 1D W-potential.
BiasedForceWPotential1D
    WPotential1D with an additional bias force in sampling.
ToyMembranePotential1D
    Dataset for a 1D toy membrane potential.
BiasedForceToyMembranePotential1D
    Biased version of ToyMembranePotential1D.
ToyMembrane2Potential1D
    Dataset for a second toy membrane potential.
BiasedForceToyMembrane2Potential1D
    Biased version of ToyMembrane2Potential1D.
ToyMembrane3Potential1D
    Dataset for a third toy membrane potential.
BiasedForceToyMembrane3Potential1D
    Biased version of ToyMembrane3Potential1D.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import jax
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float, Int
from matplotlib import pyplot as plt

from fpsl.datasets.potentials import (
    toy_membrane_potential_1d,
    toy_membrane2_potential_1d,
    toy_membrane3_potential_1d,
    w_potential_1d,
)
from fpsl.utils.integrators import (
    BiasedForceEulerMaruyamaIntegrator,
    EulerMaruyamaIntegrator,
)
from fpsl.utils.typing import JaxKey


class DataSet(ABC):
    r"""Abstract base class defining dataset interface.

    All datasets must implement sampling and potential evaluation.

    Methods
    -------
    sample(key, **kwargs)
        Draw samples from the target distribution.
    potential(x, t)
        Evaluate the potential energy at position $x$ and time $t$.
    """

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def potential(self, *args, **kwargs):
        raise NotImplementedError


@beartype
@dataclass(kw_only=True)
class WPotential1D(DataSet):
    r"""Dataset for the 1D W-potential.

    This class integrates the 1D W-potential $U(x)$ using
    the overdamped Langevin dynamics (Euler-Maruyama).

    Parameters
    ----------
    x : array_like, shape (dim1,), default=linspace(0,1,100)
        Grid points for plotting the potential.
    gamma : callable
        Friction function $\gamma(x)$, defaults to constant 1.

    Methods
    -------
    potential(x, t)
        Returns $U(x)$ from w_potential_1d.
    plot_potential(x=None, title='Toy W-Potential')
        Plot $U(x)$ vs x.
    sample(key, dt, n_steps, n_samples, beta)
        Simulate and return samples modulo 1.

    Returns
    -------
    samples : ndarray, shape (n_samples, 1)
        Final positions of particles samples.
    """

    x: Float[ArrayLike, ' dim1'] = field(
        default_factory=lambda: jnp.linspace(0, 1, 100)
    )
    gamma: Callable[[Float[ArrayLike, ' dim1']], Float[ArrayLike, '']] = lambda x: 1.0

    def potential(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ''],
    ) -> Float[ArrayLike, '']:
        r"""Evaluate the W-potential at x.

        Parameters
        ----------
        x : array_like, shape (1,)
            Position in [0,1].
        t : float
            Time (ignored for static potential).

        Returns
        -------
        U : float
            Potential energy $U(x)$.
        """
        return w_potential_1d(x[0])

    def plot_potential(
        self,
        x: None | Float[ArrayLike, ' dim1'] = None,
        title: str = 'Toy W-Potential',
    ):
        """Plot the potential energy curve.

        Parameters
        ----------
        x : array_like, optional
            Grid points; defaults to self.x.
        title : str, optional
            Plot title.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Matplotlib Axes instance with the plot.
        """
        if x is None:
            x = self.x
        vectorized_potential = jnp.vectorize(
            lambda xv: self.potential(jnp.array([xv]), 0.0)
        )
        ax = plt.gca()
        ax.plot(x, vectorized_potential(x), 'k--', label='ref')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$U(x)$')
        ax.set_xlim(x[0], x[-1])
        ax.set_title(title)
        return ax

    def sample(
        self,
        key: JaxKey,
        dt: Float[ArrayLike, ''] = 1e-4,
        n_steps: Int[ArrayLike, ''] = int(1e5),
        n_samples: Int[ArrayLike, ''] = 2048,
        beta: Float[ArrayLike, ''] = 1.0,
    ) -> Float[ArrayLike, 'n_samples 1']:
        """Sample positions via Euler-Maruyama integration.

        Parameters
        ----------
        key : JaxKey
            PRNG key.
        dt : float
            Time step size.
        n_steps : int
            Number of heat-up steps.
        n_samples : int
            Number of independent trajectories.
        beta : float
            Inverse temperature.

        Returns
        -------
        samples : ndarray, shape (n_samples, 1)
            Final positions modulo 1.
        """
        integrator = EulerMaruyamaIntegrator(
            potential=self.potential,
            n_dims=1,
            dt=dt,
            beta=beta,
            n_heatup=n_steps,
            gamma=self.gamma,
        )
        key1, key2 = jax.random.split(key)
        trajs, _, _ = integrator.integrate(
            key=key1,
            X=jax.random.uniform(key2, (n_samples, 1)),
            n_steps=0,
        )
        return trajs[-1] % 1


@beartype
@dataclass(kw_only=True)
class BiasedForceWPotential1D(WPotential1D):
    r"""Biased-force dataset for the 1D W-potential.

    Extends WPotential1D by adding a bias force $b(x,t)$.

    Parameters
    ----------
    bias : callable
        Bias force function $b(x,t)$.

    Methods
    -------
    sample(key, dt, n_steps, n_samples, beta)
        Simulate biased dynamics and return samples.
    """

    bias: Callable[
        [Float[ArrayLike, ' dim1'], Float[ArrayLike, '']], Float[ArrayLike, ' dim1']
    ]

    def sample(
        self,
        key: JaxKey,
        dt: Float[ArrayLike, ''] = 1e-4,
        n_steps: Int[ArrayLike, ''] = int(1e5),
        n_samples: Int[ArrayLike, ''] = 2048,
        beta: Float[ArrayLike, ''] = 1.0,
    ) -> Float[ArrayLike, 'n_samples 1']:
        """Sample positions with bias force via Euler-Maruyama."""
        integrator = BiasedForceEulerMaruyamaIntegrator(
            bias_force=self.bias,
            potential=self.potential,
            n_dims=1,
            dt=dt,
            beta=beta,
            n_heatup=n_steps,
            gamma=self.gamma,
        )
        key1, key2 = jax.random.split(key)
        trajs, _, _ = integrator.integrate(
            key=key1,
            X=jax.random.uniform(key2, (n_samples, 1)),
            n_steps=0,
        )
        return trajs[-1] % 1


@beartype
@dataclass(kw_only=True)
class ToyMembranePotential1D(WPotential1D):
    r"""Dataset for a toy membrane potential in 1D.

    Overrides potential with $U(x)=\mathrm{toy\_membrane\_potential\_1d}(x)$.
    Inherits sampling and plotting behavior from WPotential1D.
    """

    def potential(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ''],
    ) -> Float[ArrayLike, '']:
        """Evaluate the toy membrane potential at x."""
        return toy_membrane_potential_1d(x[0])

    def plot_potential(
        self,
        x: None | Float[ArrayLike, ' dim1'] = None,
        title: str = 'Toy Membrane Potential',
    ):
        """Plot the toy membrane potential curve."""
        return super().plot_potential(x, title)


@beartype
@dataclass(kw_only=True)
class BiasedForceToyMembranePotential1D(
    ToyMembranePotential1D,
    BiasedForceWPotential1D,
):
    """Biased-force dataset for the toy membrane potential."""

    # inherits bias parameter and methods


@beartype
@dataclass(kw_only=True)
class ToyMembrane2Potential1D(WPotential1D):
    r"""Dataset for a second toy membrane potential in 1D.

    Overrides potential with $U(x)=\mathrm{toy\_membrane2\_potential\_1d}(x)$.
    """

    def potential(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ''],
    ) -> Float[ArrayLike, '']:
        """Evaluate the second toy membrane potential at x."""
        return toy_membrane2_potential_1d(x[0])

    def plot_potential(
        self,
        x: None | Float[ArrayLike, ' dim1'] = None,
        title: str = 'Toy Membrane 2 Potential',
    ):
        """Plot the second toy membrane potential curve."""
        return super().plot_potential(x, title)


@beartype
@dataclass(kw_only=True)
class BiasedForceToyMembrane2Potential1D(
    ToyMembrane2Potential1D,
    BiasedForceWPotential1D,
):
    """Biased-force dataset for the second toy membrane potential."""


@beartype
@dataclass(kw_only=True)
class ToyMembrane3Potential1D(WPotential1D):
    r"""Dataset for a third toy membrane potential in 1D.

    Overrides potential with $U(x)=\mathrm{toy\_membrane3\_potential\_1d}(x)$.
    """

    def potential(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ''],
    ) -> Float[ArrayLike, '']:
        """Evaluate the third toy membrane potential at x."""
        return toy_membrane3_potential_1d(x[0])

    def plot_potential(
        self,
        x: None | Float[ArrayLike, ' dim1'] = None,
        title: str = 'Toy Membrane 3 Potential',
    ):
        """Plot the third toy membrane potential curve."""
        return super().plot_potential(x, title)


@beartype
@dataclass(kw_only=True)
class BiasedForceToyMembrane3Potential1D(
    ToyMembrane3Potential1D,
    BiasedForceWPotential1D,
):
    """Biased-force dataset for the third toy membrane potential."""
