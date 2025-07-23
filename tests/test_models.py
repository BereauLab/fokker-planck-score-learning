"""Test suite for the FPSL model class."""

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from fpsl.ddm.models import FPSL


class TestFPSLInitialization:
    """Test FPSL model initialization and properties."""

    def test_fpsl_basic_initialization(self):
        """Test basic FPSL instantiation with minimal parameters."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        assert model.mlp_network == (32, 32)
        assert model.key is key
        assert model.n_sample_steps == 100
        assert model.n_epochs == 100
        assert model.batch_size == 128
        assert model.wandb_log is False
        assert model.gamma_energy_regulariztion == 1e-5
        assert model.fourier_features == 1
        assert model.warmup_steps == 5
        assert model.box_size == 1.0
        assert not model.symmetric
        assert model.pbc_bins == 0

    def test_fpsl_custom_initialization(self):
        """Test FPSL with custom parameters."""
        key = jax.random.PRNGKey(123)

        def custom_diffusion(x):
            return 2.0 * jnp.ones_like(x)

        model = FPSL(
            mlp_network=(64, 64, 64),
            key=key,
            n_sample_steps=200,
            n_epochs=50,
            batch_size=64,
            wandb_log=True,
            gamma_energy_regulariztion=1e-4,
            fourier_features=2,
            warmup_steps=10,
            box_size=2.0,
            symmetric=True,
            diffusion=custom_diffusion,
            pbc_bins=10,
        )

        assert model.mlp_network == (64, 64, 64)
        assert model.n_sample_steps == 200
        assert model.n_epochs == 50
        assert model.batch_size == 64
        assert model.wandb_log is True
        assert model.gamma_energy_regulariztion == 1e-4
        assert model.fourier_features == 2
        assert model.warmup_steps == 10
        assert model.box_size == 2.0
        assert model.symmetric is True
        assert model.pbc_bins == 10
        assert model.diffusion(jnp.array([0.5])) == 2.0

    def test_fpsl_post_init(self):
        """Test that __post_init__ correctly initializes derived attributes."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        # Check that _avg_diffusion is computed
        assert hasattr(model, '_avg_diffusion')
        assert model._avg_diffusion == 1.0  # Default diffusion is constant 1.0

    def test_fpsl_inheritance_properties(self):
        """Test that FPSL inherits properties from parent classes."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        # Check inherited properties from UniformPrior
        assert model.is_periodic is True
        assert model._prior == 'uniform'

        # Check that noise schedule methods are available
        assert hasattr(model, 'sigma')
        assert hasattr(model, 'beta')

        # Check that prior schedule methods are available
        assert hasattr(model, 'alpha')

        # Check that force schedule methods are available
        assert hasattr(model, 'alpha_force')


class TestFPSLScoreModel:
    """Test score model creation and caching."""

    def test_score_model_creation_standard(self):
        """Test score model creation with standard (non-symmetric) MLP."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 16),
            key=key,
            symmetric=False,
        )

        score_model = model.score_model
        assert score_model is not None
        # Check that it's cached
        assert model.score_model is score_model

    def test_score_model_creation_symmetric(self):
        """Test score model creation with symmetric MLP."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 16),
            key=key,
            symmetric=True,
        )

        score_model = model.score_model
        assert score_model is not None


class TestFPSLDiffusion:
    """Test diffusion-related methods."""

    def test_estimate_avg_diffusion_constant(self):
        """Test average diffusion estimation with constant diffusion."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        avg_diff = model._estimate_avg_diffusion()
        assert jnp.isclose(avg_diff, 1.0)

    def test_estimate_avg_diffusion_variable(self):
        """Test average diffusion estimation with variable diffusion."""
        key = jax.random.PRNGKey(42)

        # Linear diffusion: D(x) = 1 + x
        def variable_diffusion(x):
            return 1.0 + x

        model = FPSL(
            mlp_network=(32, 32),
            key=key,
            diffusion=variable_diffusion,
        )

        avg_diff = model._estimate_avg_diffusion()
        # For linear function from 0 to 1: avg = 1 + 0.5 = 1.5
        assert jnp.isclose(avg_diff, 1.5, rtol=1e-3)

    def test_diffusion_t(self):
        """Test time-dependent diffusion coefficient."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        x = jnp.array([0.5])
        t = 0.5

        # Mock alpha method
        model.alpha = lambda t: t

        diff_t = model._diffusion_t(x, t)
        expected = (1 - t) * model.diffusion(x) / model._avg_diffusion + t
        assert jnp.allclose(diff_t, expected)

    def test_ln_diffusion_t(self):
        """Test logarithm of time-dependent diffusion."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 32),
            key=key,
        )

        x = jnp.array([0.5])
        t = 0.5

        # Mock alpha method
        model.alpha = lambda t: t

        ln_diff_t = model._ln_diffusion_t(x, t)
        diff_t = model._diffusion_t(x, t)
        expected = jnp.log(diff_t).sum()
        assert jnp.allclose(ln_diff_t, expected)


class TestFPSLScoreAndEnergy:
    """Test score and energy computation methods."""

    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model for testing."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Initialize model parameters manually for testing
        model.dim = 1
        model.params = model.score_model.init(
            key,
            t=jnp.ones(1),
            x=jnp.ones(1),
        )

        # Mock required methods
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t
        model.beta = lambda t: 1.0

        return model

    def test_score_computation_no_forces(self, trained_model):
        """Test score computation without force conditioning."""
        x = jnp.array([[0.5], [0.3], [0.7]])
        t = 0.5

        scores = trained_model.score(x, t, y=None)

        assert scores.shape == x.shape
        assert jnp.isfinite(scores).all()

    def test_score_computation_with_forces(self, trained_model):
        """Test score computation with force conditioning."""
        x = jnp.array([[0.5], [0.3], [0.7]])
        t = 0.5
        y = jnp.array([[1.0], [0.5], [-0.5]])

        scores = trained_model.score(x, t, y=y)

        assert scores.shape == x.shape
        assert jnp.isfinite(scores).all()

    def test_score_zero_sigma(self, trained_model):
        """Test score computation when sigma is zero."""
        x = jnp.array([[0.5], [0.3]])
        t = 0.0  # sigma(0) should be small/zero

        # Mock sigma to return exactly zero
        trained_model.sigma = lambda t: 0.0 if t == 0.0 else 0.1

        scores = trained_model.score(x, t)

        assert scores.shape == x.shape
        assert jnp.allclose(scores, 0.0)

    def test_energy_computation_no_forces(self, trained_model):
        """Test energy computation without force conditioning."""
        x = jnp.array([[0.5], [0.3], [0.7]])
        t = 0.5

        energies = trained_model.energy(x, t, y=None)

        assert energies.shape == (x.shape[0],)
        assert jnp.isfinite(energies).all()

    def test_energy_computation_with_forces(self, trained_model):
        """Test energy computation with force conditioning."""
        x = jnp.array([[0.5], [0.3], [0.7]])
        t = 0.5
        y = jnp.array([[1.0], [0.5], [-0.5]])

        energies = trained_model.energy(x, t, y=y)

        assert energies.shape == (x.shape[0],)
        assert jnp.isfinite(energies).all()

    def test_energy_zero_sigma(self, trained_model):
        """Test energy computation when sigma is zero."""
        x = jnp.array([[0.5], [0.3]])
        t = 0.0

        # Mock sigma to return exactly zero
        trained_model.sigma = lambda t: 0.0 if t == 0.0 else 0.1

        energies = trained_model.energy(x, t)

        # Accept either shape and flatten for comparison
        assert energies.shape == x.shape
        # Check if it's close to zero
        assert jnp.allclose(energies, 0.0)


class TestFPSLTraining:
    """Test training functionality."""

    def test_train_input_validation(self):
        """Test training input validation."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_epochs=1,
        )

        # Test 1D input (should raise error)
        X_1d = jnp.array([0.1, 0.2, 0.3])
        y = jnp.array([[1.0], [0.5], [-0.5]])
        lrs = jnp.array([1e-4, 1e-3])

        with pytest.raises(ValueError, match='X must be 2D array'):
            model.train(X_1d, y, lrs)

    def test_train_basic_functionality(self):
        """Test basic training functionality."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_epochs=5,
            batch_size=4,
            wandb_log=False,
            warmup_steps=1,  # Small warmup for fast testing
        )

        X = jax.random.uniform(key, (32, 1))
        y = jax.random.normal(key, (32, 1))
        lrs = jnp.array([1e-4, 1e-3])

        # Mock the required methods for training
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t

        loss_hist = model.train(X, y, lrs)

        assert 'train_loss' in loss_hist
        assert len(loss_hist['train_loss']) == 5
        assert hasattr(model, 'params')
        assert hasattr(model, 'dim')
        assert model.dim == 1

    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.finish')
    def test_train_with_wandb(self, mock_finish, mock_log, mock_init):
        """Test training with W&B logging."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_epochs=5,
            batch_size=4,
            wandb_log=True,
            warmup_steps=1,
        )

        X = jax.random.uniform(
            key, (24, 1)
        )  # Increased size to ensure positive decay_steps
        y = jax.random.normal(key, (24, 1))
        lrs = jnp.array([1e-4, 1e-3])

        # Mock required methods
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t

        model.train(X, y, lrs)

        mock_init.assert_called_once()
        mock_finish.assert_called_once()

    def test_train_with_validation(self):
        """Test training with validation data."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_epochs=5,
            batch_size=4,
            wandb_log=False,
            warmup_steps=1,
        )

        X = jax.random.uniform(key, (32, 1))
        y = jax.random.normal(key, (32, 1))
        X_val = jax.random.uniform(key, (4, 1))
        y_val = jax.random.normal(key, (4, 1))
        lrs = jnp.array([1e-4, 1e-3])

        # Mock required methods
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t

        loss_hist = model.train(X, y, lrs, X_val=X_val, y_val=y_val)

        assert 'val_loss' in loss_hist
        assert loss_hist['val_loss'] is not None
        assert len(loss_hist['val_loss']) == 5


class TestFPSLEvaluation:
    """Test model evaluation functionality."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for evaluation tests."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Set up required attributes
        model.dim = 1
        model.params = model.score_model.init(
            key,
            t=jnp.ones(1),
            x=jnp.ones(1),
        )

        # Mock required methods
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t

        return model

    def test_evaluate_basic(self, trained_model):
        """Test basic evaluation functionality."""
        X = jnp.array([[0.5], [0.3], [0.7]])
        y = jnp.array([[1.0], [0.5], [-0.5]])

        loss = trained_model.evaluate(X, y)

        assert isinstance(loss, float)
        assert jnp.isfinite(loss)

    def test_evaluate_without_forces(self, trained_model):
        """Test evaluation without force conditioning."""
        X = jnp.array([[0.5], [0.3], [0.7]])
        # Provide dummy forces since the loss function expects them
        y = jnp.zeros_like(X)

        loss = trained_model.evaluate(X, y)

        assert isinstance(loss, float)
        assert jnp.isfinite(loss)

    def test_evaluate_with_custom_key(self, trained_model):
        """Test evaluation with custom random key."""
        X = jnp.array([[0.5], [0.3]])
        y = jnp.array([[1.0], [0.5]])
        custom_key = jax.random.PRNGKey(999)

        loss = trained_model.evaluate(X, y, key=custom_key)

        assert isinstance(loss, float)
        assert jnp.isfinite(loss)


class TestFPSLSampling:
    """Test sampling functionality."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for sampling tests."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_sample_steps=10,
        )

        # Set up required attributes
        model.dim = 2  # 2D for more interesting tests
        model.params = model.score_model.init(
            key,
            t=jnp.ones(1),
            x=jnp.ones(2),
        )

        # Mock required methods
        model.prior_sample = lambda key, shape: jax.random.uniform(key, shape)
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.beta = lambda t: 1.0
        model.is_periodic = True

        return model

    def test_sample_basic(self, trained_model):
        """Test basic sampling functionality."""
        key = jax.random.PRNGKey(123)
        n_samples = 5

        samples = trained_model.sample(key, n_samples)

        assert samples.shape == (n_samples, trained_model.dim)
        assert jnp.isfinite(samples).all()
        # Check periodic bounds
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 1.0)

    def test_sample_custom_parameters(self, trained_model):
        """Test sampling with custom parameters."""
        key = jax.random.PRNGKey(123)
        n_samples = 3
        t_final = 0.1
        n_steps = 5

        samples = trained_model.sample(key, n_samples, t_final=t_final, n_steps=n_steps)

        assert samples.shape == (n_samples, trained_model.dim)
        assert jnp.isfinite(samples).all()

    def test_sample_different_sizes(self, trained_model):
        """Test sampling with different sample sizes."""
        key = jax.random.PRNGKey(123)

        for n_samples in [1, 10, 100]:
            samples = trained_model.sample(key, n_samples)
            assert samples.shape == (n_samples, trained_model.dim)


class TestFPSLUtilityMethods:
    """Test utility and helper methods."""

    def test_get_config(self):
        """Test configuration dictionary creation."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(32, 16),
            key=key,
            n_epochs=50,
            batch_size=64,
        )

        X = jnp.array([[0.5], [0.3]])
        y = jnp.array([[1.0], [0.5]])
        lrs = jnp.array([1e-4, 1e-3])
        n_epochs = 50

        config = model._get_config(lrs, key, n_epochs, X, y)

        assert isinstance(config, dict)
        assert 'learning_rates' in config
        assert 'mlp_network' in config
        assert 'n_epochs' in config
        assert 'batch_size' in config
        assert config['n_epochs'] == n_epochs
        assert config['batch_size'] == 64

    def test_create_loss_fn(self):
        """Test loss function creation."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Mock required methods
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.gamma_energy_regulariztion = 1e-5

        loss_fn = model._create_loss_fn()

        assert callable(loss_fn)

    def test_create_update_step(self):
        """Test update step function creation."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Mock optimizer
        mock_optim = MagicMock()

        # Mock required methods
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.gamma_energy_regulariztion = 1e-5

        update_step = model._create_update_step(mock_optim)

        assert callable(update_step)


class TestFPSLEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_sigma_handling(self):
        """Test handling of zero sigma values."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Mock params
        model.params = {}

        x = jnp.array([[0.5]])
        t = 0.0

        # Mock sigma to return zero
        model.sigma = lambda t: 0.0

        # Both score and energy should handle zero sigma gracefully
        scores = model.score(x, t)
        energies = model.energy(x, t)

        assert jnp.allclose(scores, 0.0)
        assert jnp.allclose(energies, 0.0)

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary condition handling."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            pbc_bins=5,
        )

        # This mainly tests that PBC parameters are set correctly
        assert model.pbc_bins == 5

    def test_different_diffusion_functions(self):
        """Test various diffusion function types."""
        key = jax.random.PRNGKey(42)

        # Constant diffusion
        model1 = FPSL(mlp_network=(8,), key=key)
        assert model1._avg_diffusion == 1.0

        # Linear diffusion
        def linear_diff(x):
            return 1.0 + x

        model2 = FPSL(mlp_network=(8,), key=key, diffusion=linear_diff)
        assert model2._avg_diffusion > 1.0

        # Quadratic diffusion
        def quad_diff(x):
            return 1.0 + x**2

        model3 = FPSL(mlp_network=(8,), key=key, diffusion=quad_diff)
        assert model3._avg_diffusion > 1.0


class TestFPSLIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_mini(self):
        """Test a complete mini workflow: create, train, evaluate, sample."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
            n_epochs=5,
            batch_size=5,
            n_sample_steps=5,
            wandb_log=False,
            warmup_steps=1,
        )

        # Generate larger dataset to ensure positive decay_steps
        key1, key2, key3 = jax.random.split(key, 3)
        X = jax.random.uniform(key1, (40, 1))  # Increased from 16 to 40
        y = jax.random.normal(key2, (40, 1))
        lrs = jnp.array([1e-4, 1e-3])

        # Mock required methods for functionality
        model.prior_x_t = lambda x, t, eps: x + 0.1 * eps * t
        model.prior_sample = lambda key, shape: jax.random.uniform(key, shape)
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t
        model.beta = lambda t: 1.0
        model.is_periodic = True

        # 1. Train
        loss_hist = model.train(X, y, lrs)
        assert 'train_loss' in loss_hist

        # 2. Evaluate
        eval_loss = model.evaluate(X, y)
        assert isinstance(eval_loss, float)

        # 3. Sample
        samples = model.sample(key3, n_samples=3)
        assert samples.shape == (3, 1)

        # 4. Compute score and energy
        scores = model.score(X[:2], t=0.5)
        energies = model.energy(X[:2], t=0.5)
        assert scores.shape == (2, 1)
        assert energies.shape == (2,)

    def test_model_consistency(self):
        """Test consistency between related methods."""
        key = jax.random.PRNGKey(42)
        model = FPSL(
            mlp_network=(8, 8),
            key=key,
        )

        # Initialize model
        model.dim = 1
        model.params = model.score_model.init(
            key,
            t=jnp.ones(1),
            x=jnp.ones(1),
        )

        # Mock methods
        model.sigma = lambda t: 0.1 + 0.9 * t
        model.alpha = lambda t: t
        model.alpha_force = lambda t: t

        x = jnp.array([[0.5]])
        t = jnp.array([0.5])

        # Test consistency between public and private methods
        score_public = model.score(x, t)
        score_private = model._score_eq(model.params, x[0], t)

        # They should be related by sigma scaling
        # Handle the shapes properly - score_private might be scalar or 1D
        expected_public = -score_private / model.sigma(t)

        # Ensure both are 1D arrays for comparison
        if score_private.ndim == 0:  # scalar
            expected_public = jnp.array([expected_public])
        elif score_private.ndim > 1:
            expected_public = expected_public.flatten()

        # Compare flattened versions
        assert jnp.allclose(score_public.flatten(), expected_public.flatten())


# Parametrized tests for different configurations
@pytest.mark.parametrize('symmetric', [True, False])
@pytest.mark.parametrize('fourier_features', [1, 2, 3])
def test_fpsl_configurations(symmetric, fourier_features):
    """Test FPSL with different architectural configurations."""
    key = jax.random.PRNGKey(42)
    model = FPSL(
        mlp_network=(16, 8),
        key=key,
        symmetric=symmetric,
        fourier_features=fourier_features,
    )

    assert model.symmetric == symmetric
    assert model.fourier_features == fourier_features

    # Test that score model can be created
    score_model = model.score_model
    assert score_model is not None


@pytest.mark.parametrize('pbc_bins', [0, 5, 10])
def test_fpsl_pbc_configurations(pbc_bins):
    """Test FPSL with different PBC bin configurations."""
    key = jax.random.PRNGKey(42)
    model = FPSL(
        mlp_network=(8, 8),
        key=key,
        pbc_bins=pbc_bins,
    )

    assert model.pbc_bins == pbc_bins


if __name__ == '__main__':
    pytest.main([__file__])
