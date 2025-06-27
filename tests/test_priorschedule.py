import pytest
import numpy as np
from fpsl.ddm.priorschedule import LinearPriorSchedule, QuadraticPriorSchedule


@pytest.mark.parametrize(
    't',
    [
        0.0,
        0.5,
        1.0,
        np.linspace(0, 1, 5),
    ],
)
def test_linear_prior_schedule_alpha_and_name(t):
    """LinearPriorSchedule.alpha(t) == t and name is 'linear'."""
    sched = LinearPriorSchedule()
    # check the identifier
    assert sched._prior_schedule == 'linear'
    # alpha must return exactly the input
    out = sched.alpha(t)
    # scalar input returns scalar, array returns array
    assert np.allclose(out, t)


@pytest.mark.parametrize(
    't',
    [
        0.0,
        0.5,
        1.0,
        np.array([0.0, 0.2, 0.8, 1.0]),
    ],
)
def test_quadratic_prior_schedule_alpha_and_name(t):
    """QuadraticPriorSchedule.alpha(t) == t**2 and name is 'quadratic'."""
    sched = QuadraticPriorSchedule()
    assert sched._prior_schedule == 'quadratic'
    out = sched.alpha(t)
    expected = np.asarray(t) ** 2
    assert np.allclose(out, expected)


def test_alpha_output_dtype_and_shape():
    """Alpha preserves the input shape and returns numpy arrays for array input."""
    t = np.random.rand(10)
    lin = LinearPriorSchedule().alpha(t)
    quad = QuadraticPriorSchedule().alpha(t)
    # both outputs should be numpy arrays of same shape as input
    assert isinstance(lin, np.ndarray)
    assert isinstance(quad, np.ndarray)
    assert lin.shape == t.shape
    assert quad.shape == t.shape
