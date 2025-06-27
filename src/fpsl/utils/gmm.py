"""This submodule provides implementations of Gaussian Mixture Models (GMMs)
with identical covariance across components, including a periodic extension to
handle data on a bounded interval via component replication.

Classes
-------
GMM
    Defines a mixture of N Gaussians with shared scalar standard deviation.
    Offers methods to compute the probability density function (PDF) and its natural
    logarithm over input samples.
PeriodicGMM
    Inherits from GMM and adds support for periodic domains [0, bound]. It replicates
    mixture components across multiple copies of the domain to evaluate densities that
    respect periodic boundary conditions.
"""

from dataclasses import dataclass
from itertools import product

import jax
from beartype import beartype
from jaxtyping import ArrayLike, Float, Int
from jax import numpy as jnp

from fpsl.utils.baseclass import DefaultDataClass


@beartype
@dataclass
class GMM:
    """Gaussian Mixture Model of N Gaussians with identical covariance.

    Parameters
    ----------
    means : jnp.ndarray
        The means of the Gaussians.
    std : float
        A scalar representing the standard deviation of the Gaussians.

    """

    means: Float[ArrayLike, ' n_samples']
    std: Float[ArrayLike, '']

    def __post_init__(self):
        self.means = self.means.reshape(-1, 1)
        self.ndim: int = self.means.shape[1]

    def pdf(self, X: Float[ArrayLike, ' n_samples']) -> Float[ArrayLike, ' n_samples']:
        """
        Calculate the probability density function (PDF) of the Gaussian Mixture Model.

        Parameters
        ----------
        X : jnp.ndarray
            The input data.

        Returns
        -------
        float
            The PDF value.

        """
        return jax.scipy.stats.norm.pdf(
            X,
            loc=self.means,
            scale=self.std,
        ).mean(axis=0)

    def ln_pdf(
        self,
        X: Float[ArrayLike, ' n_samples'],
    ) -> Float[ArrayLike, ' n_samples']:
        """
        Calculate the natural logarithm of the probability density function (PDF) of the Gaussian Mixture Model.

        Parameters
        ----------
        X : jnp.ndarray
            The input data.

        Returns
        -------
        jnp.ndarray
            The natural logarithm of the PDF value.

        """
        return jnp.log(self.pdf(X))


@beartype
@dataclass
class PeriodicGMM(GMM, DefaultDataClass):
    """Gaussian Mixture Model of N Gaussians with identical covariance.

    Parameters
    ----------
    means : jnp.ndarray
        The means of the Gaussians.
    std : Union[jnp.ndarray, int]
        A scalar representing the standard deviation of the Gaussians.
    bound : float, default=1.0
        The data is periodic on [0, bound].
    copies : int, default=5
        Number of copies in each direction.

    """

    bound: Float[ArrayLike, ''] = 1.0
    copies: Int[ArrayLike, ''] = 5

    def __post_init__(self):
        super().__post_init__()
        self.means %= self.bound
        self.offset = (
            jnp.arange(
                -self.copies,
                self.copies + 1,
            )
            * self.bound
        )
        self.offsets = jnp.array(
            list(product(*[self.offset for _ in range(self.ndim)])),
        )
        self.n_copies = len(self.offsets)

    def pdf(self, X: Float[ArrayLike, ' n_samples']) -> Float[ArrayLike, ' n_samples']:
        """
        Calculate the probability density function (PDF) of the Gaussian Mixture Model.

        Parameters
        ----------
        X : jnp.ndarray
            The input data.

        Returns
        -------
        jnp.ndarray
            The PDF values.

        """
        return jnp.array([
            jax.scipy.stats.norm.pdf(
                X + offset,
                loc=self.means,
                scale=self.std,
            ).mean(axis=0)
            for offset in self.offsets
        ]).sum(axis=0)
