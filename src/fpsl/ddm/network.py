"""Neural network architectures for score-based diffusion models.

This module provides MLP-based score networks with optional Fourier feature
embeddings for periodic domains.
"""

from typing import Sequence

from flax import linen as nn
from jax import numpy as jnp


class MLP(nn.Module):
    """Simple feed-forward multilayer perceptron producing scalar outputs.

    Parameters
    ----------
    features : Sequence[int]
        Number of units in each hidden layer. A final Dense layer of size 1
        is appended automatically.

    Returns
    -------
    output : jnp.ndarray
        Array of shape (..., 1), the scalar output for each example.
    """

    features: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP."""
        for feature in self.features:
            x = nn.Dense(feature)(x)
            x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x


class ScoreMLP(nn.Module):
    """Time-dependent score network using a simple MLP.

    This network concatenates the state `x` and time embedding `t`, passes
    them through an MLP, and sums the final outputs to a scalar score.

    Parameters
    ----------
    features : Sequence[int]
        Hidden layer sizes for the underlying MLP.

    Returns
    -------
    score : jnp.ndarray
        Scalar score per example, shape (...,).
    """

    features: Sequence[int]

    def setup(self) -> None:
        self.mlp = MLP(features=self.features)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the score for each (x, t) pair."""
        h = jnp.concatenate((x, t), axis=-1)
        return self.mlp(h).sum(axis=-1)


class ScorePeriodicMLP(nn.Module):
    """Periodic score network with sine/cosine Fourier features.

    Extends `ScoreMLP` by first embedding `x` via Fourier features before
    concatenating with time.

    Parameters
    ----------
    features : Sequence[int]
        Hidden layer sizes for the underlying MLP.
    fourier_features_stop : int, optional
        Maximum frequency (inclusive) for Fourier embeddings. Default is 1.

    Returns
    -------
    score : jnp.ndarray
        Scalar score per example, shape (...,).
    """

    features: Sequence[int]
    fourier_features_stop: int = 1

    def setup(self) -> None:
        self.mlp = MLP(features=self.features)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute periodic score for each (x, t)."""
        ff = FourierFeatures(
            stop=self.fourier_features_stop,
            odd=True,
            even=True,
        )
        h = jnp.concatenate((ff(x), t), axis=-1)
        return self.mlp(h).sum(axis=-1)


class ScoreSymmetricPeriodicMLP(nn.Module):
    """Symmetric periodic score with cosine-only Fourier features.

    Similar to `ScorePeriodicMLP` but uses only cosine terms for even symmetry.

    Parameters
    ----------
    features : Sequence[int]
        Hidden layer sizes for the underlying MLP.
    fourier_features_stop : int, optional
        Maximum frequency (inclusive) for Fourier embeddings. Default is 1.

    Returns
    -------
    score : jnp.ndarray
        Scalar score per example, shape (...,).
    """

    features: Sequence[int]
    fourier_features_stop: int = 1

    def setup(self) -> None:
        self.mlp = MLP(features=self.features)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute symmetric periodic score for each (x, t)."""
        ff = FourierFeatures(
            stop=self.fourier_features_stop,
            odd=False,
            even=True,
        )
        h = jnp.concatenate((ff(x), t), axis=-1)
        return self.mlp(h).sum(axis=-1)


class FourierFeatures(nn.Module):
    """Generate sine/cosine Fourier features for periodic inputs.

    Embeds each dimension of `x` into multiple frequencies via sine and/or
    cosine.

    Parameters
    ----------
    start : int, optional
        Starting frequency (inclusive). Default is 1.
    stop : int, optional
        Stopping frequency (inclusive). Default is 8.
    step : int, optional
        Frequency step size. Default is 1.
    odd : bool, optional
        If True, include sine components. Default is True.
    even : bool, optional
        If True, include cosine components. Default is True.

    Raises
    ------
    ValueError
        If both `odd` and `even` are False.

    Returns
    -------
    features : jnp.ndarray
        Flattened array of shape (..., n_dims * n_freqs * n_components),
        where `n_components` is 1 or 2 depending on `odd`/`even`.
    """

    start: int = 1
    stop: int = 8
    step: int = 1
    odd: bool = True
    even: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute Fourier feature embedding for `x`."""
        freqs = jnp.arange(self.start, self.stop + 1, self.step, dtype=x.dtype)
        freqs = jnp.tile(freqs[None, :], (1, x.shape[-1]))
        x_freqs = ((freqs * jnp.repeat(x, len(freqs), axis=-1)) % 1) * 2 * jnp.pi

        if not (self.odd or self.even):
            raise ValueError('Either `odd` (sine) or `even` (cosine) must be True.')

        if self.odd and self.even:
            return jnp.concatenate(
                (jnp.sin(x_freqs), jnp.cos(x_freqs)),
                axis=-1,
            ).flatten()
        elif self.even:
            return jnp.cos(x_freqs).flatten()
        return jnp.sin(x_freqs).flatten()
