r"""
This module provides a collection of one-dimensional, periodic potential energy
functions. Each function is JIT-compiled for efficiency and runtime type-checked
using beartype.

Each function accepts an array-like input `x` of arbitrary shape and returns
a scalar or array of potential values corresponding to each element of `x`.

Defined functions:

- w_potential_1d(x):
    Periodic double-well potential
    $\phi(x) = [2\cos(2\pi x) + 1]\,\cos(2\pi x) - 0.628279.$

- toy_membrane_potential_1d(x):
    Toy membrane potential
    $\phi(x) = -\tfrac12\Bigl(\tfrac{(\cos(4\pi x)-1)^4}{4} + \cos(2\pi x)\Bigr) + 0.862700.$

- toy_membrane2_potential_1d(x):
    Parameterized periodic potential based on MD data
    $\phi(x) = \phi_0 + \sum_{i=1}^N \alpha_i\,\cos(2\pi i x),
    \quad \phi_0 = 0.77830946.$

- toy_membrane3_potential_1d(x):
    Alternative MD-based series potential
    $\phi(x) = \phi_0 + \sum_{i=1}^N \alpha_i\,\cos(2\pi i x),
    \quad \phi_0 = 0.26733318.$

"""

import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jaxtyping import ArrayLike, Float


@jit
@beartype
def w_potential_1d(x: Float[ArrayLike, '']) -> Float[ArrayLike, '']:
    r"""Periodic double well potential.

    $$
    \phi(x) = [2\cos(2\pi x) + 1]\cos(2\pi x)
    $$
    """
    return (2 * jnp.cos(2 * jnp.pi * x) ** 2 + jnp.cos(2 * jnp.pi * x) - 0.628279).sum()


@jit
@beartype
def toy_membrane_potential_1d(x: Float[ArrayLike, '']) -> Float[ArrayLike, '']:
    r"""Periodic toy membrane potential.

    $$
    \phi(x) = - \frac{1}{2} \left[\frac{(\cos(4\pi x) -1)^4}{4} +\cos(2\pi x)\right]
    $$
    """
    return (
        -1
        / 2
        * ((jnp.cos(4 * jnp.pi * x) - 1) ** 4 / 4 + jnp.cos(2 * jnp.pi * x)).sum()
        + 0.862700
    )


@jit
@beartype
def toy_membrane2_potential_1d(x: Float[ArrayLike, '']) -> Float[ArrayLike, '']:
    r"""Periodic toy membrane potential based on MD."""
    alphas = jnp.array([
        4.38756764e-01,
        5.51305175e-01,
        -9.66697633e-01,
        4.76220012e-01,
        3.51388663e-01,
        -6.57680452e-01,
        3.20680887e-01,
        1.19107859e-02,
        4.46960330e-04,
        -1.06695265e-01,
        1.18188225e-02,
        9.63761136e-02,
        -6.34292215e-02,
        -1.26926098e-02,
        -2.20742077e-03,
        3.11103798e-02,
        -1.16342846e-02,
        -2.06445083e-02,
        1.19867604e-02,
        2.14913208e-03,
    ])
    ns = jnp.arange(1, len(alphas) + 1)
    phi_0 = 0.77830946
    return phi_0 + (alphas * jnp.cos(2 * jnp.pi * x * ns)).sum()


@jit
@beartype
def toy_membrane3_potential_1d(x: Float[ArrayLike, '']) -> Float[ArrayLike, '']:
    r"""Periodic toy membrane potential based on MD."""
    alphas = jnp.array([
        -6.5813565e-01,
        6.3420063e-01,
        -5.1591349e-01,
        1.2018956e-01,
        2.8954777e-01,
        -3.8310459e-01,
        1.6142714e-01,
        3.7272871e-02,
        -4.2626891e-02,
        -4.5450684e-03,
        -6.4875120e-03,
        3.0953294e-02,
        -1.6555822e-02,
        -7.3198415e-03,
        9.5287291e-03,
        -1.1982573e-03,
        8.0206152e-04,
        -1.3514888e-03,
        1.1800211e-03,
        2.3124311e-03,
    ])
    ns = jnp.arange(1, len(alphas) + 1)
    phi_0 = 0.26733318
    return phi_0 + (alphas * jnp.cos(2 * jnp.pi * x * ns)).sum()
