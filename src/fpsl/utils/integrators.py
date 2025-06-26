from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import numpy as np
from beartype import beartype
from beartype.typing import Callable
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
    r"""
    Euler–Maruyama integrator for Langevin dynamics.

    This class implements the Euler–Maruyama numerical scheme to integrate the
    overdamped Langevin SDE:

    $$
    \mathrm{d}x = -\frac{\nabla \phi(x, t)}{\gamma(x)} \,\mathrm{d}t
    + \sqrt{\frac{2}{\beta\,\gamma(x)}} \,\mathrm{d}W_t
    $$

    where $\phi(x, t)$ is the potential energy, $\gamma(x)$ is the
    position-dependent friction coefficient, $\beta$ is the inverse
    temperature, and $W_t$ is a standard Wiener process.

    Parameters
    ----------
    potential : Callable[[ArrayLike, float], ArrayLike]
        Potential energy function $\phi(x, t)$. Accepts `x` of shape
        `(n_dims,)` and scalar `t`, returns scalar or array of shape `()`.
    n_dims : int
        Dimensionality of the state space.
    dt : float
        Time step size $\Delta t$.
    beta : float
        Inverse temperature parameter $\beta = 1/(k_B T)$.
    n_heatup : int, optional
        Number of initial “heat-up” steps before recording trajectories.
        Default is 1000.
    gamma : Callable[[ArrayLike], ArrayLike], optional
        Position-dependent friction coefficient function $\gamma(x)$.
        Accepts `x` of shape `(n_dims,)`, returns scalar or array of shape `()`.
        Default is constant 1.0.

    Attributes
    ----------
    potential : Callable
        (Scalar) Potential function $\phi(x, t)$.
    n_dims : int
        Dimensionality of the system.
    dt : float
        Integration time step $\Delta t$.
    beta : float
        Inverse temperature.
    n_heatup : int
        Number of pre-integration heat-up steps.
    gamma : Callable
        Position-dependent friction coefficient $\gamma(x)$.

    Methods
    -------
    integrate(key, X, n_steps=1000)
        Run the Euler–Maruyama integrator over `n_steps`, returning:
        - `xs`: array of shape `(n_t_steps, n_samples, n_dims)` of positions,
        - `fs`: array of same shape for forces,
        - `ts`: time points of shape `(n_t_steps,)`.
    """

    potential: Callable[
        [Float[ArrayLike, ' n_dims'], Float[ArrayLike, '']],  # x, t
        Float[ArrayLike, ''],  # phi(x, t)
    ]
    n_dims: int
    dt: float
    beta: float
    n_heatup: int = 1000
    gamma: Callable[
        [Float[ArrayLike, ' n_dims']],  # x
        Float[ArrayLike, ''],  # Gamma(x)
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
        r"""
        Integrate Brownian dynamics using the Euler–Maruyama scheme.

        Parameters
        ----------
        key : JaxKey
            PRNG key for JAX random number generation.
        X : array-like, shape (n_samples, n_dims)
            Initial positions of the particles.
        n_steps : int, optional
            Number of integration steps to perform. Default is 1000.

        Returns
        -------
        positions : ndarray, shape (n_t_steps, n_samples, n_dims)
            Trajectories of the particles at each time step.
        forces : ndarray, shape (n_t_steps, n_samples, n_dims)
            Deterministic forces $F = -\nabla U(X, t)$ evaluated along the trajectory.
        times : ndarray, shape (n_t_steps,)
            Time points corresponding to each integration step.

        Notes
        -----
        The integrator approximates the overdamped Langevin equation

        $$
            X_{t+1} = X_t + F(X_t, t)\,\Delta t + \sqrt{2\,\Delta t}\,\xi_t,
        $$

        where:

        - $F(X, t) = -\nabla U(X, t)$ is the conservative force,
        - $\Delta t$ is the time step size (total time divided by $n$ steps),
        - $\xi_t$ are independent standard normal random variables.
        """

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
                + self.dt * fs[idx_current] / self.gamma(xs[idx_current])
                + jnp.sqrt(2 * self.dt / self.beta / self.gamma(xs[idx_current]))
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
    r"""
    Euler–Maruyama integrator with an additional bias force.

    This class extends EulerMaruyamaIntegrator by incorporating
    a user-specified bias force term into the overdamped Langevin SDE:

    $$
    \mathrm{d}x = -\nabla \phi(x, t)\,\mathrm{d}t
    + b(x, t)\,\mathrm{d}t
    + \sqrt{\frac{2}{\beta\,\gamma(x)}}\,\mathrm{d}W_t,
    $$

    where
      - $\phi(x, t)$ is the potential energy,
      - $b(x, t)$ is the bias force,
      - $\gamma(x)$ is the (optional) position-dependent friction,
      - $\beta$ is the inverse temperature,
      - $W_t$ is a standard Wiener process.

    Parameters
    ----------
    bias_force : Callable[[ArrayLike, float], ArrayLike]
        User-defined bias force function $b(x, t)$. Accepts `x` of shape
        `(n_dims,)` and scalar `t`, returns an array of shape `(n_dims,)`.
        Default is zero bias.
    **kwargs
        All other keyword arguments are the same as for
        [fpsl.utils.integrators.EulerMaruyamaIntegrator][]:

        - `potential`: potential energy function,
        - `n_dims`: number of dimensions,
        - `dt`: time step size,
        - `beta`: inverse temperature,
        - `n_heatup`: number of initial heat-up steps,
        - `gamma`: friction coefficient function.
    """

    bias_force: Callable[
        [Float[ArrayLike, ' n_dims'], Float[ArrayLike, '']],  # x, t
        Float[ArrayLike, ' n_dims'],  # forces
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
        r"""
        Integrate Brownian dynamics using the Euler–Maruyama scheme with bias.

        Parameters
        ----------
        key : JaxKey
            PRNG key for JAX random number generation.
        X : array-like, shape (n_samples, n_dims)
            Initial positions of the particles.
        n_steps : int, optional
            Number of integration steps to perform. Default is 1000.

        Returns
        -------
        positions : ndarray, shape (n_t_steps, n_samples, n_dims)
            Trajectories of the particles at each time step, including heat-up.
        forces : ndarray, shape (n_t_steps, n_samples, n_dims)
            Total forces $F = -\nabla U(X,t) + b(X,t)$ evaluated along the trajectory.
        times : ndarray, shape (n_t_steps,)
            Time points corresponding to each integration step.

        """

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
