import jax
from jax import numpy as jnp


def get_fourier_func(ys, xs, n_order=51, even=True, odd=False, idx_offset=10):
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

    offset = jnp.array([
        ys[:idx_offset].mean(),
        ys[-idx_offset:].mean(),
    ]).mean()
    ys = ys - offset

    ns = jnp.arange(1, n_order + 1)
    alphas, betas = jnp.array([alpha_and_beta_n(n, ys, xs) for n in ns]).T
    phi_0 = jnp.trapezoid(ys, x=xs) + offset

    def fourier_func(xs, phi_0, alphas, betas):
        f_even = jnp.sum(
            jnp.array([
                alpha * jnp.cos(2 * jnp.pi * i * xs)
                for i, alpha in enumerate(alphas, 1)
            ]),
            axis=0,
        )
        f_odd = jnp.sum(
            jnp.array([
                beta * jnp.sin(2 * jnp.pi * i * xs) for i, beta in enumerate(betas, 1)
            ]),
            axis=0,
        )
        return phi_0 + (f_even if even else 0) + (f_odd if odd else 0)

    @jax.jit
    def jitted_fourier_func(x):
        return fourier_func(x, phi_0, alphas, betas).sum()

    return jitted_fourier_func


def get_fourier_func2d(zs, xs, ys, n_order=21):
    """Fourier expansion of a function.

    Parameters
    ----------
    zs : jnp.ndarray
        The function values to be expanded.
    xs, ys : jnp.ndarray
        The x/y values corresponding to the function values. Should be within [0, 1].
    n_order : int
        The order of the Fourier expansion. Default is 51.

    Returns
    -------
    jitted_taylor_func : callable[(float64, float64)->float64]
        A JAX function that computes the Fourier expansion of the input function.

    """

    def kappa(n, m):
        return jnp.where(
            n == 0 and m == 0,
            1.0,
            jnp.where(n == 0 or m == 0, 2.0, 4.0),
        )

    base_funcs = [(fn, fm) for fn in (jnp.cos, jnp.sin) for fm in (jnp.cos, jnp.sin)]

    def abyd_nm(n, m, zs, xs, ys):
        """Alpha, beta, gamma, delta for order (n,m)."""
        return jnp.array([
            kappa(n, m)
            * jnp.trapezoid(
                jnp.trapezoid(
                    zs
                    * fn(2 * jnp.pi * n * xs).reshape(-1, 1)
                    * fm(2 * jnp.pi * m * ys).reshape(1, -1),
                    x=xs,
                    axis=0,
                ),
                x=ys,
                axis=0,
            )
            for fn, fm in base_funcs
        ])

    ns = jnp.arange(0, n_order + 1)
    ms = jnp.arange(0, n_order + 1)
    nms = jnp.array([(n, m) for n in ns for m in ms])
    abyds = jnp.array([abyd_nm(n, m, zs, xs, ys) for n, m in nms])
    n_orders = len(nms)

    def fourier_func(x, y):
        def inner_func(x, y, n, m, abyd):
            return jnp.sum(
                jnp.array([
                    weight * fn(2 * jnp.pi * n * x) * fm(2 * jnp.pi * m * y)
                    for weight, (fn, fm) in zip(abyd, base_funcs)
                ])
            )

        def body(idx, acc):
            return acc + inner_func(x, y, nms[idx][0], nms[idx][1], abyds[idx])

        return jax.lax.fori_loop(0, n_orders, body, 0.0)

    @jax.jit
    def jitted_fourier_func(x, y):
        return fourier_func(x, y)

    return jitted_fourier_func
