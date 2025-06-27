import numpy as np
import jax
import jax.numpy as jnp
from fpsl.utils.gmm import GMM, PeriodicGMM


def test_gmm_pdf_single_gaussian():
    """GMM with one component should match scipy.stats.norm.pdf."""
    means = jnp.array([0.0])
    std = 1.0
    g = GMM(means=means, std=std)
    X = jnp.array([-1.0, 0.0, 1.0])
    pdf = g.pdf(X)
    expected = jax.scipy.stats.norm.pdf(X, loc=0.0, scale=std)
    assert pdf.shape == X.shape
    assert np.allclose(np.array(pdf), np.array(expected))


def test_gmm_pdf_two_components_average():
    """GMM with two components should average their PDFs."""
    means = jnp.array([0.0, 2.0])
    std = 0.5
    g = GMM(means=means, std=std)
    X = jnp.array([0.0, 2.0, 1.0])
    pdf = g.pdf(X)
    p0 = jax.scipy.stats.norm.pdf(X, loc=0.0, scale=std)
    p2 = jax.scipy.stats.norm.pdf(X, loc=2.0, scale=std)
    expected = 0.5 * (p0 + p2)
    assert pdf.shape == X.shape
    assert np.allclose(np.array(pdf), np.array(expected))


def test_gmm_ln_pdf_is_log_of_pdf():
    """ln_pdf should equal log(pdf)."""
    means = jnp.array([1.0, -1.0])
    std = 2.0
    g = GMM(means=means, std=std)
    X = jnp.array([0.0, 1.0, -1.0])
    pdf = g.pdf(X)
    ln_pdf = g.ln_pdf(X)
    assert ln_pdf.shape == X.shape
    assert np.allclose(np.array(ln_pdf), np.log(np.array(pdf)))


def test_periodicgmm_n_copies_and_offsets():
    """PeriodicGMM should generate (2*copies+1)^ndim offsets."""
    means = jnp.array([0.2, 0.8])
    copies = 2
    pg = PeriodicGMM(means=means, std=0.3, bound=1.0, copies=copies)
    expected_copies = (2 * copies + 1) ** pg.ndim
    assert pg.n_copies == expected_copies
    assert pg.offsets.shape == (expected_copies, pg.ndim)


def test_periodicgmm_pdf_invariance_under_periodic_shift():
    """PDF should be invariant under shifts by the period bound."""
    means = jnp.array([0.5])
    pg = PeriodicGMM(means=means, std=0.5, bound=1.0)
    X = jnp.array([0.1, 0.4, 0.9])
    pdf = pg.pdf(X)
    pdf_shift = pg.pdf(X + 1.0)
    assert np.allclose(np.array(pdf), np.array(pdf_shift))


def test_periodicgmm_pdf_exceeds_gmm_near_boundary():
    """PeriodicGMM.pdf should exceed GMM.pdf when boundary wrap matters."""
    means = jnp.array([0.0])
    std = 0.5
    g = GMM(means=means, std=std)
    pg = PeriodicGMM(means=means, std=std, bound=1.0)
    X = jnp.array([0.9])
    pdf_g = g.pdf(X)
    pdf_pg = pg.pdf(X)
    assert np.all(pdf_pg > pdf_g)
