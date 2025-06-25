from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float

from fpsl.utils.baseclass import DefaultDataClass
from fpsl.utils.typing import JaxKey


class BrownianIntegrator(ABC):
    r"""Overdamped Langevin eq. integrator.

    Solving the following SDE

    $$
    \mathrm{d}x = -\phi(x, t)\mathrm{d}t + \sqrt{2\beta^{-1}}\mathrm{d}W_t
    $$
    """

    @abstractmethod
    def integrate(self):
        raise NotImplementedError


@beartype
@dataclass(kw_only=True)
class EulerMaruyamaIntegrator(DefaultDataClass, BrownianIntegrator):
    """Integrate Langevin equation."""

    potential: Callable[
        [Float[ArrayLike, ' n_dims'], Float[ArrayLike, '']],  # x, t
        Float[ArrayLike, ''],  # phi(x, t)
    ]
    n_dims: int
    dt: float
    beta: float
    n_heatup: int = 1000
    mgamma: Callable[
        [Float[ArrayLike, ' n_dims']],  # x
        Float[ArrayLike, ''],  # mGamma(x)
    ] = lambda x: 1.0

    def __post_init__(self):
        super().__post_init__()
        self.potential = beartype(self.potential)

    def integrate(
        self,
        key: JaxKey,
        X: Float[ArrayLike, 'n_samples n_dims'],
        n_steps: int = 1000,
    ) -> tuple[
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, ' n_t_steps'],
    ]:
        """Integrate Brownian dynamics using Euler Maruyama integrator."""

        @jax.jit
        def force(X: Float[ArrayLike, 'n_samples n_dims'], t: float):
            return -1 * jax.vmap(
                jax.grad(self.potential, argnums=0),
                in_axes=(0, None),
            )(X, t)

        return self._integrate(
            key=key,
            X=X,
            n_steps=n_steps,
            force=force,
        )

    def _integrate(
        self,
        key: JaxKey,
        X: Float[ArrayLike, 'n_samples n_dims'],
        n_steps: int,
        force: Callable[
            [Float[ArrayLike, ' n_dims'], Float[ArrayLike, '']],  # x, t
            Float[ArrayLike, ''],  # phi(x, t)
        ],
    ) -> tuple[
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, ' n_t_steps'],
    ]:
        xs = jnp.zeros((n_steps + 2, *X.shape))
        xs = xs.at[0].set(X)

        fs = jnp.zeros((n_steps + 2, *X.shape))

        ts = jnp.arange(-self.n_heatup, n_steps + 2) * self.dt
        ts = jnp.where(ts < 0, 0, ts)

        def step_fn(i, carry):
            xs, fs, key = carry
            key, _ = jax.random.split(key)
            idx_current = jnp.maximum(i - self.n_heatup, 0)
            idx_next = jnp.maximum(i + 1 - self.n_heatup, 0)

            fs = fs.at[idx_current].set(force(xs[idx_current], ts[i + 1]))
            next_x = (
                xs[idx_current]
                + self.dt * fs[idx_current] / self.mgamma(xs[idx_current])
                + jnp.sqrt(2 * self.dt / self.beta / self.mgamma(xs[idx_current]))
                * jax.random.normal(
                    key,
                    shape=X.shape,
                )
            )
            xs = xs.at[idx_next].set(next_x)
            return xs, fs, key

        final_xs, final_fs, _ = jax.lax.fori_loop(
            0, len(ts) - 1, lambda i, carry: step_fn(i, carry), (xs, fs, key)
        )

        # The final step is only for estimating the final force
        return final_xs[:-1], final_fs[:-1], ts[self.n_heatup : -1]


@beartype
@dataclass(kw_only=True)
class BiasedForceEulerMaruyamaIntegrator(EulerMaruyamaIntegrator):
    """Integrate Langevin equation."""

    bias_force: Callable[
        [Float[ArrayLike, ' n_dims'], Float[ArrayLike, '']],  # x, t
        Float[ArrayLike, ' n_dims'],  # forcess
    ] = lambda x, t: np.zeros_like(x)

    def __post_init__(self):
        super().__post_init__()
        self.bias_force = beartype(self.bias_force)

    def integrate(
        self,
        key: JaxKey,
        X: Float[ArrayLike, 'n_samples n_dims'],
        n_steps: int = 1000,
    ) -> tuple[
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, 'n_t_steps n_samples n_dims'],
        Float[ArrayLike, ' n_t_steps'],
    ]:
        """Integrate Brownian dynamics using Euler Maruyama integrator."""

        @jax.jit
        def force(X: Float[ArrayLike, 'n_samples n_dims'], t: float):
            return -1 * jax.vmap(
                jax.grad(self.potential, argnums=0),
                in_axes=(0, None),
            )(X, t) + jax.vmap(
                self.bias_force,
                in_axes=(0, None),
            )(X, t)

        return self._integrate(
            key=key,
            X=X,
            n_steps=n_steps,
            force=force,
        )
