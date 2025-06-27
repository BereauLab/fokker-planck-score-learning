import pytest
import jax
import jax.numpy as jnp
from fpsl.ddm.prior import UniformPrior


def test_uniform_prior_attributes():
    """UniformPrior should report periodic support and correct name."""
    up = UniformPrior()
    assert up.is_periodic is True
    assert up._prior == 'uniform'


def test_uniform_prior_pdf_and_log_pdf():
    """prior_pdf should be 1 everywhere, prior_log_pdf should be 0 everywhere."""
    up = UniformPrior()
    x = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    pdf = up.prior_pdf(x)
    log_pdf = up.prior_log_pdf(x)
    assert pdf.shape == x.shape
    assert jnp.all(pdf == 1.0)
    assert log_pdf.shape == x.shape
    assert jnp.all(log_pdf == 0.0)


def test_uniform_prior_force():
    """prior_force should return zero gradient for any x."""
    up = UniformPrior()
    x = jnp.array([0.0, 0.3, 0.9])
    force = up.prior_force(x)
    assert force.shape == x.shape
    assert jnp.all(force == 0.0)


def test_uniform_prior_sample_and_bounds():
    """prior_sample should generate samples in [0,1) of the requested shape."""
    up = UniformPrior()
    key = jax.random.PRNGKey(0)
    shape = (1000,)
    samples = up.prior_sample(key, shape)
    assert samples.shape == shape
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples < 1.0)


@pytest.mark.parametrize(
    'x, t, eps',
    [
        (
            jnp.array([-0.2, 0.5, 1.2]),
            jnp.array([0.0, 0.1, 0.5]),
            jnp.array([0.0, 1.0, -1.0]),
        ),
        (jnp.array([0.0, 0.25, 0.75]), jnp.zeros((3,)), jnp.ones((3,))),
    ],
)
def test_uniform_prior_x_t_wraps_into_unit_interval(x, t, eps):
    """
    prior_x_t should add noise scaled by sigma(t) and wrap results into [0,1).
    We only check that without noise the input is preserved.
    """
    up = UniformPrior()
    # add self.sigma function
    up.sigma = lambda t: jnp.zeros_like(t)
    x = x.reshape(-1, 1)
    xt = up.prior_x_t(x, t, eps)
    assert jnp.allclose(xt, x % 1)
