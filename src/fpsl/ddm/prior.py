"""Latent prior distributions for score-based diffusion models.

This module defines the abstract base class for latent prior distributions
and provides concrete implementations, such as the uniform prior on [0,1].

Classes
-------
LatentPrior
    Abstract base class for latent priors.
UniformPrior
    Uniform prior over [0,1] with periodic support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from fpsl.utils.baseclass import DefaultDataClass
from fpsl.utils.typing import JaxKey


@dataclass(kw_only=True)
class LatentPrior(DefaultDataClass, ABC):
    """Abstract base class for latent prior distributions.

    Attributes
    ----------
    is_periodic : bool
        If True, the support of the prior is treated as periodic.

    Methods
    -------
    _prior : str
        Name identifier of the prior (abstract property).
    prior_log_pdf(x)
        Log probability density at x.
    prior_pdf(x)
        Probability density at x.
    prior_sample(key, shape)
        Sample from the prior.
    prior_force(x)
        Force term (gradient of log-pdf) at x.
    prior_x_t(x, t, eps)
        Diffuse x with noise and wrap if periodic.
    """

    is_periodic: bool = False

    @property
    @abstractmethod
    def _prior(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def prior_log_pdf(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, '']:
        raise NotImplementedError

    @abstractmethod
    def prior_pdf(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, '']:
        raise NotImplementedError

    @abstractmethod
    def prior_sample(
        self,
        key: JaxKey,
        shape: tuple[int],
    ) -> Float[ArrayLike, ' dim1']:
        raise NotImplementedError

    @abstractmethod
    def prior_force(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, ' dim1']:
        raise NotImplementedError

    @abstractmethod
    def prior_x_t(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ' dim1'],
        eps: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, ' dim1']:
        raise NotImplementedError


@dataclass(kw_only=True)
class UniformPrior(LatentPrior):
    """Uniform prior over the unit interval [0, 1].

    A periodic prior with constant density and zero force.

    Attributes
    ----------
    is_periodic : bool
        Always True for the uniform prior.

    Methods
    -------
    prior_log_pdf(x)
        Returns zero array (log-density).
    prior_pdf(x)
        Returns one array (density).
    prior_sample(key, shape)
        Samples uniformly in [0,1].
    prior_force(x)
        Returns zero array (gradient of log-density).
    prior_x_t(x, t, eps)
        Applies diffusion step with noise and wraps modulo 1.
    """

    is_periodic: bool = True

    @property
    def _prior(self) -> str:
        return 'uniform'

    def prior_log_pdf(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, '']:
        """Log-probability density: zero everywhere on [0,1]."""
        return jnp.zeros_like(x)

    def prior_pdf(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, '']:
        """Probability density: one everywhere on [0,1]."""
        return jnp.ones_like(x)

    def prior_sample(
        self,
        key: JaxKey,
        shape: tuple[int],
    ) -> Float[ArrayLike, ' dim1']:
        """Sample uniformly from [0,1] with given JAX PRNG key."""
        return jax.random.uniform(key, shape)

    def prior_force(
        self,
        x: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, ' dim1']:
        """Force term (gradient of log-pdf): zero everywhere."""
        return jnp.zeros_like(x)

    def prior_x_t(
        self,
        x: Float[ArrayLike, ' dim1'],
        t: Float[ArrayLike, ' dim1'],
        eps: Float[ArrayLike, ' dim1'],
    ) -> Float[ArrayLike, ' dim1']:
        """Diffuse x with noise and wrap into [0,1].

        Parameters
        ----------
        x : array-like, shape (dim1,)
            Current latent variable.
        t : array-like, shape (dim1,)
            Time embedding for noise scaling.
        eps : array-like, shape (dim1,)
            Standard normal noise sample.

        Returns
        -------
        x_next : jnp.ndarray, shape (dim1,)
            Noisy update of x wrapped modulo 1.
        """
        return (x + self.sigma(t) * eps) % 1
