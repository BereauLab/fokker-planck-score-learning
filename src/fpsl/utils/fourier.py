import jax
from jax import numpy as jnp


def get_fourier_func(ys, xs, n_order=51):
    """Fourier expansion of a function.

    Parameters
    ----------
    ys : jnp.ndarray
        The function values to be expanded.
    xs : jnp.ndarray
        The x values corresponding to the function values. Should be within [0, 1].
    n_order : int
        The order of the Fourier expansion. Default is 51.

    Returns
    -------
    jitted_taylor_func : callable[(float64)->float64]
        A JAX function that computes the Fourier expansion of the input function.

    """

    def alpha_and_beta_n(n, ys, xs):
        return (
            2 * jnp.trapezoid(ys * jnp.cos(2 * jnp.pi * n * xs), x=xs),
            2 * jnp.trapezoid(ys * jnp.sin(2 * jnp.pi * n * xs), x=xs),
        )

    offset = ys[:10].mean()
    ys = ys - offset

    ns = jnp.arange(1, n_order + 1)
    alphas, _ = jnp.array([alpha_and_beta_n(n, ys, xs) for n in ns]).T
    phi_0 = jnp.trapezoid(ys, x=xs) + offset

    def fourier_func(xs, phi_0, alphas):
        return (
            jnp.sum(
                jnp.array([
                    alpha * jnp.cos(2 * jnp.pi * i * xs)
                    for i, alpha in enumerate(alphas, 1)
                ]),
                axis=0,
            )
            + phi_0
        )

    @jax.jit
    def jitted_fourier_func(x):
        return fourier_func(x, phi_0, alphas).sum()

    return jitted_fourier_func
