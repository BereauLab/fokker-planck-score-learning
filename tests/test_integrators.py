import jax
import numpy as np

# python
import jax.numpy as jnp

from fpsl.utils.integrators import (
    EulerMaruyamaIntegrator,
    BiasedForceEulerMaruyamaIntegrator,
)


def test_euler_maruyama_zero_dt():
    # Setup
    key = jax.random.PRNGKey(0)
    X0 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    integrator = EulerMaruyamaIntegrator(
        potential=lambda x, t: 1.0,
        n_dims=2,
        dt=0.0,
        beta=1.0,
        n_heatup=0,
    )

    # Integrate for 5 steps
    xs, fs, ts = integrator.integrate(key, X0, n_steps=5)

    # Expected shapes
    assert xs.shape == (6, 2, 2)
    assert fs.shape == (6, 2, 2)
    assert ts.shape == (6,)

    # With dt=0, no drift, no noise → xs constant, fs zero, ts zero
    expected_xs = np.broadcast_to(np.expand_dims(np.array(X0), 0), xs.shape)
    assert np.array_equal(np.array(xs), expected_xs)
    assert np.allclose(np.array(fs), 0.0)
    assert np.allclose(np.array(ts), 0.0)


def test_biased_force_equals_euler_when_zero_bias():
    # Setup
    key = jax.random.PRNGKey(42)
    X0 = jnp.array([[0.5, -0.5]])
    common_kwargs = dict(
        potential=lambda x, t: 1.0,
        n_dims=2,
        dt=0.0,
        beta=1.0,
        n_heatup=0,
    )

    ext_force = 1.0
    base = EulerMaruyamaIntegrator(**common_kwargs)
    biased = BiasedForceEulerMaruyamaIntegrator(
        bias_force=lambda x, t: jnp.full_like(x, ext_force),
        **common_kwargs,
    )

    # Integrate both for 3 steps
    xs_e, fs_e, ts_e = base.integrate(key, X0, n_steps=3)
    xs_b, fs_b, ts_b = biased.integrate(key, X0, n_steps=3)

    # They must match exactly when bias_force is the default zero
    assert jnp.allclose(xs_b, xs_e)
    assert jnp.allclose(fs_b - ext_force, fs_e)
    assert jnp.allclose(ts_b, ts_e)
