r"""Force schedule functions for driven DDPM diffusion models.

This module defines the base abstract class for force schedules and two
concrete implementations that map a normalized time variable $t \in [0,1]$
to a force scaling coefficient $\alpha_{\mathrm{force}}(t)$.

Classes
-------
ForceSchedule
    Abstract base class for force schedules.
LinearForceSchedule
    Fades in force linearly: $\alpha_{\mathrm{force}}(t) = 1 - t$.
ConstantForceSchedule
    Constant force: $\alpha_{\mathrm{force}}(t) = 1$.
"""

from abc import ABC, abstractmethod

from jax import numpy as jnp
from jaxtyping import ArrayLike, Float


class ForceSchedule(ABC):
    r"""Abstract base class for force schedules.

    Defines the interface for the time-dependent force scaling coefficient
    $\alpha_{\mathrm{force}}(t)$ in driven DDPM models.

    Methods
    -------
    alpha_force(t)
        Compute force scaling at time $t$.
    """

    @property
    @abstractmethod
    def _force_schedule(self) -> str:
        """Name identifier of the force schedule."""
        raise NotImplementedError

    @abstractmethod
    def alpha_force(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute the force scaling coefficient at time $t$.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps, where $t \in [0, 1]$.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Force scaling coefficients.
        """
        raise NotImplementedError


class LinearForceSchedule(ForceSchedule):
    r"""Linearly fading force schedule.

    Implements:

    $$
    \alpha_{\mathrm{force}}(t) = 1 - t
    $$

    Methods
    -------
    alpha_force(t)
        Returns $1 - t$ for each time $t$.
    """

    @property
    def _force_schedule(self) -> str:
        """Name identifier: 'linear'."""
        return 'linear'

    def alpha_force(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute linearly fading force scaling.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Force scaling coefficients equal to $1 - t$.
        """
        return 1 - t


class ConstantForceSchedule(ForceSchedule):
    r"""Constant force schedule.

    Implements:

    $$
    \alpha_{\mathrm{force}}(t) = 1
    $$

    Methods
    -------
    alpha_force(t)
        Returns 1 for each time $t$.
    """

    @property
    def _force_schedule(self) -> str:
        """Name identifier: 'constant'."""
        return 'constant'

    def alpha_force(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute constant force scaling.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Force scaling coefficients all equal to 1.
        """
        return jnp.ones_like(t)
