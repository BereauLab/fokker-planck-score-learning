r"""Time-dependent noise schedules for DDPM diffusion models.

This module defines an abstract base class for noise schedules and several
concrete implementations for mapping a normalized time variable $t \in [0,1]$
to the noise parameters $\beta(t)$, $\sigma(t)$, and $\gamma(t)$:

- QuadraticVarianceNoiseSchedule:
  $\sigma(t)\propto(\sqrt{\sigma_{\min}/\sigma_{\max}}+t)^2$.
- LinearVarianceNoiseSchedule:
  $\sigma(t)\propto(\sigma_{\min}/\sigma_{\max}+t)$.
- ExponentialVarianceNoiseSchedule:
  $\sigma(t)=\sigma_{\min}^{1-t}\,\sigma_{\max}^t$.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from fpsl.utils.baseclass import DefaultDataClass


@dataclass(kw_only=True)
class NoiseSchedule(DefaultDataClass, ABC):
    r"""Abstract base class for time-dependent noise schedules.

    Defines the noise schedule $\beta(t)$, the cummulative noise scale
    $\sigma(t)$, and the mean drift $\gamma(t)$ for a diffusion process
    given a normalized time $t \in [0,1]$.

    Attributes
    ----------
    sigma_min : float
        Minimum noise scale at $t=0$.
    sigma_max : float
        Maximum noise scale at $t=1$.

    Methods
    -------
    beta(t)
        Instantaneous noise rate $\beta(t)$.
    sigma(t)
        Noise scale $\sigma(t)$.
    gamma(t)
        Mean drift $\gamma(t)$.
    """

    sigma_min: float
    sigma_max: float

    @property
    @abstractmethod
    def _noise_schedule(self) -> str:
        """Identifier of the noise schedule (e.g. 'tanBeta')."""
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Instantaneous noise rate $\beta(t)$.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps $t \in [0,1]$.

        Returns
        -------
        beta_t : array_like, shape (dim1,)
            Noise rate at each time.
        """
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Noise scale $\sigma(t)$.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps $t \in [0,1]$.

        Returns
        -------
        sigma_t : array_like, shape (dim1,)
            Noise magnitude at each time.
        """
        raise NotImplementedError

    @abstractmethod
    def gamma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Mean drift $\gamma(t)$.

        Parameters
        ----------
        t : array_like, shape (dim1,)
            Normalized time steps $t \in [0,1]$.

        Returns
        -------
        gamma_t : array_like, shape (dim1,)
            Mean drift at each time.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class QuadraticVarianceNoiseSchedule(NoiseSchedule):
    r"""Quadratic variance-exploding noise schedule.

    Defines:

    $$
      \sigma(t) = \frac{(\sqrt{\sigma_{\min}/\sigma_{\max}} + t)^2}
                        {(\sqrt{\sigma_{\min}/\sigma_{\max}} + 1)^2}
                   \,\sigma_{\max},
      \quad \beta(t) = \frac{d}{dt}\sigma(t)^2.
    $$

    Attributes
    ----------
    sigma_min : float, default=0.07
        Starting noise scale.
    sigma_max : float, default=0.5
        Ending noise scale.
    """

    sigma_min: float = 0.07
    sigma_max: float = 0.5

    @property
    def _noise_schedule(self) -> str:
        """Name identifier: 'quadraticVariance'."""
        return 'quadraticVariance'

    def gamma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        """No drift; not implemented."""
        raise NotImplementedError

    def sigma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\sigma(t)$ as above."""
        factor = (jnp.sqrt(self.sigma_min / self.sigma_max) + t) ** 2
        norm = (jnp.sqrt(self.sigma_min / self.sigma_max) + 1) ** 2
        return factor / norm * self.sigma_max

    def beta(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\beta(t) = \frac{d}{dt}\sigma(t)^2$ via autograd."""
        return jnp.vectorize(jax.grad(lambda tt: self.sigma(tt) ** 2))(t)


@dataclass(kw_only=True)
class LinearVarianceNoiseSchedule(NoiseSchedule):
    r"""Linear variance–exploding noise schedule.

    Defines:

    $$
      \sigma(t) = \frac{(\sigma_{\min}/\sigma_{\max} + t)}
                       {(\sigma_{\min}/\sigma_{\max} + 1)}
                   \,\sigma_{\max},
      \quad \beta(t) = \frac{d}{dt}\sigma(t)^2.
    $$
    """

    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @property
    def _noise_schedule(self) -> str:
        """Name identifier: 'linearVariance'."""
        return 'linearVariance'

    def gamma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        """No drift; not implemented."""
        raise NotImplementedError

    def sigma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\sigma(t)$ as above."""
        factor = self.sigma_min / self.sigma_max + t
        norm = self.sigma_min / self.sigma_max + 1
        return factor / norm * self.sigma_max

    def beta(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\beta(t) = \frac{d}{dt}\sigma(t)^2$ via autograd."""
        return jnp.vectorize(jax.grad(lambda tt: self.sigma(tt) ** 2))(t)


@dataclass(kw_only=True)
class ExponetialVarianceNoiseSchedule(NoiseSchedule):
    r"""Exponential variance-exploding noise schedule.

    Defines:

    $$
      \sigma(t) = \sigma_{\min}^{1-t}\,\sigma_{\max}^t,
      \quad \beta(t) = \frac{d}{dt}\sigma(t)^2.
    $$
    """

    sigma_min: float = 0.05
    sigma_max: float = 0.5

    @property
    def _noise_schedule(self) -> str:
        """Name identifier: 'exponentialVariance'."""
        return 'exponentialVariance'

    def gamma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        """No drift; not implemented."""
        raise NotImplementedError

    def sigma(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\sigma(t) = \sigma_{\min}^{1-t}\,\sigma_{\max}^t$."""
        return self.sigma_min ** (1 - t) * self.sigma_max**t

    def beta(self, t: Float[ArrayLike, ' dim1']) -> Float[ArrayLike, ' dim1']:
        r"""Compute $\beta(t) = \frac{d}{dt}\sigma(t)^2$ via autograd."""
        return jnp.vectorize(jax.grad(lambda tt: self.sigma(tt) ** 2))(t)
