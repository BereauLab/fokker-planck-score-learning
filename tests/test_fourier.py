import jax.numpy as jnp

from fpsl.utils.fourier import get_fourier_func, get_fourier_func2d


def test_get_fourier_func_reconstructs_even_function():
    """Default (cosine-only) expansion must reconstruct an even periodic function."""
    xs = jnp.linspace(0, 1, 400, endpoint=True)
    ys = jnp.cos(2 * jnp.pi * xs)
    f = get_fourier_func(ys, xs, n_order=5)

    test_pts = jnp.array([0.0, 0.13, 0.37, 0.5, 0.81])
    out = jnp.array([f(x) for x in test_pts])
    assert jnp.allclose(out, jnp.cos(2 * jnp.pi * test_pts), atol=1e-3)


def test_get_fourier_func_odd_flag_reconstructs_odd_function():
    """With even=False, odd=True the expansion must reconstruct a sine signal."""
    xs = jnp.linspace(0, 1, 400, endpoint=True)
    ys = jnp.sin(2 * jnp.pi * xs)
    f = get_fourier_func(ys, xs, n_order=5, even=False, odd=True)

    test_pts = jnp.array([0.0, 0.13, 0.37, 0.5, 0.81])
    out = jnp.array([f(x) for x in test_pts])
    assert jnp.allclose(out, jnp.sin(2 * jnp.pi * test_pts), atol=1e-3)


def test_get_fourier_func_default_ignores_sine_component():
    """The default (cosine-only) expansion of a pure sine signal is ~flat/zero.

    Cosine terms cannot represent an odd signal, so with the default
    `even=True, odd=False` the reconstruction of a pure sine wave should
    collapse to (close to) zero rather than tracking the sine shape.
    """
    xs = jnp.linspace(0, 1, 400, endpoint=True)
    ys = jnp.sin(2 * jnp.pi * xs)
    f = get_fourier_func(ys, xs, n_order=5)

    test_pts = jnp.array([0.13, 0.37, 0.81])
    out = jnp.array([f(x) for x in test_pts])
    assert jnp.allclose(out, 0.0, atol=1e-3)


def test_get_fourier_func2d_reconstructs_separable_function():
    """2D expansion must reconstruct a simple separable periodic function."""
    xs = jnp.linspace(0, 1, 80, endpoint=True)
    ys = jnp.linspace(0, 1, 80, endpoint=True)
    x_grid, y_grid = jnp.meshgrid(xs, ys, indexing='ij')
    zs = jnp.cos(2 * jnp.pi * x_grid) * jnp.sin(2 * jnp.pi * y_grid)

    f = get_fourier_func2d(zs, xs, ys, n_order=3)

    test_pts = [(0.2, 0.65), (0.5, 0.5), (0.9, 0.1)]
    for tx, ty in test_pts:
        expected = jnp.cos(2 * jnp.pi * tx) * jnp.sin(2 * jnp.pi * ty)
        assert jnp.allclose(f(tx, ty), expected, atol=1e-3)
