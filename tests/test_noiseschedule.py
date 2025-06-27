import pytest
import numpy as np
import jax.numpy as jnp

from fpsl.ddm.noiseschedule import (
    QuadraticVarianceNoiseSchedule,
    LinearVarianceNoiseSchedule,
    ExponetialVarianceNoiseSchedule,
)


@pytest.mark.parametrize(
    'cls, name',
    [
        (QuadraticVarianceNoiseSchedule, 'quadraticVariance'),
        (LinearVarianceNoiseSchedule, 'linearVariance'),
        (ExponetialVarianceNoiseSchedule, 'exponentialVariance'),
    ],
)
def test_schedule_name_and_gamma_not_implemented(cls, name):
    """Each schedule reports correct name and gamma() raises."""
    sched = cls()  # use default sigma_min/sigma_max
    assert sched._noise_schedule == name
    t = jnp.array([0.0, 0.5, 1.0])
    with pytest.raises(NotImplementedError):
        sched.gamma(t)


@pytest.mark.parametrize(
    'sched_cls',
    [
        QuadraticVarianceNoiseSchedule,
        LinearVarianceNoiseSchedule,
        ExponetialVarianceNoiseSchedule,
    ],
)
def test_sigma_endpoint_at_t_equals_one(sched_cls):
    """For t=1, sigma(1) should equal sigma_max."""
    sigma_min = 0.1
    sigma_max = 0.8
    sched = sched_cls(sigma_min=sigma_min, sigma_max=sigma_max)
    # scalar t
    t1 = jnp.array(1.0)
    sigma1 = sched.sigma(t1)
    assert np.allclose(float(sigma1), sigma_max, atol=1e-6)
    # array t
    t_arr = jnp.array([1.0, 1.0])
    sigma_arr = sched.sigma(t_arr)
    assert np.allclose(np.array(sigma_arr), sigma_max)


@pytest.mark.parametrize(
    'sched_cls',
    [
        QuadraticVarianceNoiseSchedule,
        LinearVarianceNoiseSchedule,
        ExponetialVarianceNoiseSchedule,
    ],
)
def test_sigma_monotonic_increasing(sched_cls):
    """sigma(t) should be non-decreasing on [0,1]."""
    sched = sched_cls(sigma_min=0.2, sigma_max=0.9)
    ts = jnp.linspace(0.0, 1.0, 11)
    sig = sched.sigma(ts)
    sig_np = np.array(sig)
    diffs = np.diff(sig_np)
    assert np.all(diffs >= -1e-8)  # allow tiny numerical noise


@pytest.mark.parametrize(
    'sched_cls',
    [
        QuadraticVarianceNoiseSchedule,
        LinearVarianceNoiseSchedule,
        ExponetialVarianceNoiseSchedule,
    ],
)
def test_beta_matches_derivative_of_sigma_squared(sched_cls):
    """beta(t) ≈ d/dt [sigma(t)**2] via finite differences."""
    sched = sched_cls(sigma_min=0.1, sigma_max=0.6)
    # pick some interior points
    t0 = 0.3
    h = 1e-3

    def sigma_sq(t):
        return float(sched.sigma(jnp.array(t)) ** 2)

    # finite difference
    num_grad = (sigma_sq(t0 + h) - sigma_sq(t0 - h)) / (2 * h)
    beta = float(sched.beta(jnp.array(t0)))
    assert np.allclose(beta, num_grad, rtol=1e-3, atol=1e-5)
