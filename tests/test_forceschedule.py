import pytest
import numpy as np
import jax.numpy as jnp

from fpsl.ddm.forceschedule import (
    ForceSchedule,
    LinearForceSchedule,
    ConstantForceSchedule,
)


def test_forceschedule_is_abstract():
    """ForceSchedule cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ForceSchedule()


@pytest.mark.parametrize('t_scalar', [0.0, 0.3, 1.0])
def test_linear_force_schedule_scalar(t_scalar):
    """LinearForceSchedule.alpha_force(t) == 1 - t for scalar t."""
    sched = LinearForceSchedule()
    assert sched._force_schedule == 'linear'
    out = sched.alpha_force(jnp.array(t_scalar))
    assert float(out) == pytest.approx(1.0 - t_scalar)


def test_linear_force_schedule_array():
    """LinearForceSchedule.alpha_force preserves shape for array input."""
    sched = LinearForceSchedule()
    t = jnp.array([0.0, 0.25, 0.75, 1.0])
    out = sched.alpha_force(t)
    assert isinstance(out, jnp.ndarray)
    np.testing.assert_allclose(np.array(out), 1.0 - np.array(t))


@pytest.mark.parametrize('t_scalar', [0.0, 0.5, 1.0])
def test_constant_force_schedule_scalar(t_scalar):
    """ConstantForceSchedule.alpha_force(t) == 1 for scalar t."""
    sched = ConstantForceSchedule()
    assert sched._force_schedule == 'constant'
    out = sched.alpha_force(jnp.array(t_scalar))
    assert float(out) == pytest.approx(1.0)


def test_constant_force_schedule_array():
    """ConstantForceSchedule.alpha_force returns ones array."""
    sched = ConstantForceSchedule()
    t = jnp.linspace(0.0, 1.0, 5)
    out = sched.alpha_force(t)
    assert isinstance(out, jnp.ndarray)
