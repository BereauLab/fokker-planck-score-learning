from contextlib import nullcontext

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from fpsl.ddm.network import (
    MLP,
    ScoreMLP,
    ScorePeriodicMLP,
    ScoreSymmetricPeriodicMLP,
    FourierFeatures,
    _BaseScorePeriodicMLP,
)


def test_mlp_output_shape():
    """MLP should map (..., n_in) → (..., 1)."""
    key = jax.random.PRNGKey(0)
    mlp = MLP(features=[8, 4])
    # batch of 3, input dim 5
    x = jnp.ones((3, 5))
    params = mlp.init(key, x)
    out = mlp.apply(params, x)
    print(out)
    assert out.shape == (3, 1)


def test_score_mlp_shape_and_sum():
    """ScoreMLP should return a scalar per example (sum over final dim)."""
    key = jax.random.PRNGKey(1)
    score = ScoreMLP(features=[6, 3])
    x = jnp.zeros((2, 4))
    t = jnp.zeros((2, 1))
    params = score.init(key, x, t)
    out = score.apply(params, x, t)
    # one scalar per example
    assert out.shape == (2,)
    # with zero weights/bias, output should be zero
    assert jnp.allclose(out, 0.0)


@pytest.mark.parametrize(
    'start, stop, step, kwargs, error',
    [
        (1, 3, 1, {'odd': True, 'even': False}, None),
        (1, 3, 1, {'odd': False, 'even': True}, None),
        (1, 3, 1, {'odd': True, 'even': True}, None),
        (1, 3, 1, {'odd': False, 'even': False}, ValueError),
    ],
)
def test_fourier_features(start, stop, step, kwargs, error):
    """FourierFeatures with odd-only on zero input must be all zeros."""
    with pytest.raises(error) if error is not None else nullcontext():
        ff = FourierFeatures(start=start, stop=stop, step=step, **kwargs)
        key = jax.random.PRNGKey(1)
        x = jnp.zeros((1,))
        params = ff.init(key, x)
        out = ff.apply(params, x)

        assert out.ndim == 1 and out.shape[0] == len(
            range(start, stop + 1, step)
        ) * np.sum(list(kwargs.values()))

        # with zero input all odd should be 0, all even should be 1
        if kwargs.get('odd', True):
            if kwargs.get('even', True):
                assert jnp.allclose(out[: len(out) // 2], 0.0)
                assert jnp.allclose(out[len(out) // 2 :], 1.0)
            else:
                assert jnp.allclose(out, 0.0)
        elif kwargs.get('even', True):
            # only even, so all should be 1
            assert jnp.allclose(out, 1.0)


@pytest.mark.parametrize('mlp', [ScorePeriodicMLP, ScoreSymmetricPeriodicMLP])
def test_score_periodic_symmetric(mlp):
    """ScorePeriodicMLP and ScoreSymmetricPeriodicMLP must output same shape."""
    key = jax.random.PRNGKey(2)
    n_dim = 5
    x = jnp.zeros((n_dim, 1))
    t = jnp.zeros(1)
    periodic = mlp(features=[5, 5], fourier_features_stop=2)
    # initialize network with random parameters
    p_params = periodic.init(key, x, t)
    out_p = periodic.apply(p_params, x, t)
    out_p_neg = periodic.apply(p_params, -x, t)
    # check that output is zero
    assert jnp.allclose(out_p, out_p_neg)


def test_base_score_periodic_mlp_odd_even_are_independently_configurable():
    """`_BaseScorePeriodicMLP` must honor its own odd/even flags (not hardcode them)."""
    key = jax.random.PRNGKey(3)
    x = jnp.full((1,), 0.1)
    t = jnp.zeros((1,))

    cosine_only = _BaseScorePeriodicMLP(
        features=[4], fourier_features_stop=2, odd=False, even=True
    )
    sine_and_cosine = _BaseScorePeriodicMLP(
        features=[4], fourier_features_stop=2, odd=True, even=True
    )

    cosine_only_params = cosine_only.init(key, x, t)
    sine_and_cosine_params = sine_and_cosine.init(key, x, t)

    # the sine-inclusive network embeds a wider Fourier feature vector (it adds
    # `n_freqs` sine terms on top of the `n_freqs` cosine terms + 1 for `t`),
    # so its first Dense layer must have a correspondingly larger kernel
    n_freqs = 2  # fourier_features_stop=2, start=1, step=1
    cosine_only_kernel = cosine_only_params['params']['mlp']['Dense_0']['kernel']
    sine_and_cosine_kernel = sine_and_cosine_params['params']['mlp']['Dense_0'][
        'kernel'
    ]
    assert cosine_only_kernel.shape[0] == n_freqs + 1
    assert sine_and_cosine_kernel.shape[0] == cosine_only_kernel.shape[0] + n_freqs


def test_score_periodic_mlp_is_not_cosine_only():
    """ScorePeriodicMLP must retain sine terms (odd=True), unlike the symmetric variant."""
    key = jax.random.PRNGKey(4)
    x = jnp.full((1,), 0.1)
    t = jnp.zeros((1,))

    periodic = ScorePeriodicMLP(features=[4], fourier_features_stop=2)
    symmetric = ScoreSymmetricPeriodicMLP(features=[4], fourier_features_stop=2)

    periodic_params = periodic.init(key, x, t)
    symmetric_params = symmetric.init(key, x, t)

    # a network with sine terms is not, in general, invariant to x -> 1 - x
    # (which flips the sign of sin(2*pi*n*x) while leaving cos(2*pi*n*x) fixed)
    out_periodic = periodic.apply(periodic_params, x, t)
    out_periodic_reflected = periodic.apply(periodic_params, 1 - x, t)
    assert not jnp.allclose(out_periodic, out_periodic_reflected)

    # the cosine-only network *is* invariant under that reflection
    out_symmetric = symmetric.apply(symmetric_params, x, t)
    out_symmetric_reflected = symmetric.apply(symmetric_params, 1 - x, t)
    assert jnp.allclose(out_symmetric, out_symmetric_reflected)
