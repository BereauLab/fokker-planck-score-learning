"""Simply tests for PyPI."""

import jax.random as jr
from fpsl.ddm.models import FPSL


if __name__ == '__main__':
    # Create model
    key = jr.PRNGKey(42)
    model = FPSL(
        mlp_network=(8, 8),
        key=key,
        n_epochs=2,
        batch_size=32,
        warmup_steps=1,
    )

    # Train on data
    X = jr.uniform(key, (256, 1))  # periodic data
    y = jr.normal(key, (256, 1))  # force data
    lrs = [1e-6, 1e-4]  # Learning rate range
    loss_hist = model.train(X, y, lrs)

    assert len(loss_hist['train_loss'] == 2)
