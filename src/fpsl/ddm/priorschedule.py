r"""Prior schedule functions for DDPM diffusion models.

This module defines a base abstract class for prior schedules and two
common concrete implementations:

- LinearPriorSchedule: $\alpha(t) = t$
- QuadraticPriorSchedule: $\alpha(t) = t^2$

Classes
-------
PriorSchedule
    Abstract base for defining $\alpha(t)$ schedules.
LinearPriorSchedule
    Simple linear schedule.
QuadraticPriorSchedule
    Simple quadratic schedule.

Notes
-----
It is assumed that the time steps `t` are normalized to the range [0, 1].
"""

from abc import ABC, abstractmethod
from jaxtyping import ArrayLike, Float


class PriorSchedule(ABC):
    r"""Abstract base class for prior schedules in DDPM models.

    A prior schedule defines the mixing coefficient $\alpha(t)$ that
    controls how the model incorporates the prior distribution over
    time steps in a diffusion process.

    Methods
    -------
    alpha(t)
        Compute the schedule coefficient $\alpha$ at time $t$.
    """

    @property
    @abstractmethod
    def _prior_schedule(self) -> str:
        """Name identifier of the schedule.

        Returns
        -------
        schedule : str
            Unique name of this prior schedule (e.g. 'linear').
        """
        raise NotImplementedError

    @abstractmethod
    def alpha(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute the schedule coefficient $\alpha$ at time $t$.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Time steps $t\in[0, 1]$.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Mixing coefficients corresponding to each time in $t$.
        """
        raise NotImplementedError


class LinearPriorSchedule(PriorSchedule):
    r"""Linearly increasing prior schedule.

    $\alpha(t) = t$

    Methods
    -------
    alpha(t)
        Compute the schedule coefficient $\alpha$ at time $t$.
    """

    @property
    def _prior_schedule(self) -> str:
        """Name of this schedule.

        Returns
        -------
        schedule : str
            'linear'
        """
        return 'linear'

    def alpha(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Linear schedule function.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Time steps $t\in[0, 1]$.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Equal to the input t.
        """
        return t


class QuadraticPriorSchedule(PriorSchedule):
    r"""Quadratically increasing prior schedule $\alpha(t) = t^2$.

    Methods
    -------
    alpha(t)
        Compute the schedule coefficient $\alpha$ at time $t$.
    """

    @property
    def _prior_schedule(self) -> str:
        """Name of this schedule.

        Returns
        -------
        schedule : str
            'quadratic'
        """
        return 'quadratic'

    def alpha(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Quadratic schedule function.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Time steps $t\in[0, 1]$.

        Returns
        -------
        alpha_t : array_like, shape (dim1,)
            Squares of the input t.
        """
        return t**2
