from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import jax
import jax_dataloader as jdl
import numpy as np
import optax
import wandb
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from tqdm import tqdm

from fpsl.ddm.prior import UniformPrior
from fpsl.ddm.noiseschedule import (
    ExponetialVarianceNoiseSchedule,
)
from fpsl.ddm.priorschedule import LinearPriorSchedule
from fpsl.ddm.network import (
    ScorePeriodicMLP,
    ScoreSymmetricPeriodicMLP,
)
from fpsl.ddm.forceschedule import (
    LinearForceSchedule,
)
from fpsl.utils.baseclass import DefaultDataClass
from fpsl.utils.typing import JaxKey

# enable NaN debugging
jax.config.update('jax_debug_nans', True)


@dataclass(kw_only=True)
class DrivenDDM(
    LinearForceSchedule,
    LinearPriorSchedule,
    UniformPrior,
    ExponetialVarianceNoiseSchedule,
    DefaultDataClass,
):
    """Energy-based denoising diffusion model for periodic data on [0, 1]."""

    mlp_network: tuple[int]
    key: JaxKey
    n_sample_steps: int = 100
    n_epochs: int = 100
    batch_size: int = 128
    wandb_log: int = False
    gamma_energy_regulariztion: float = 1e-5
    fourier_features: int = 1
    warmup_steps: int = 5
    box_size: float = 1.0
    symmetric: bool = False
    diffusion: Callable[[Float[ArrayLike, ' n_features']], Float[ArrayLike, '']] = (
        lambda x: 1.0
    )
    pbc_bins: int = 0

    def _estimate_avg_diffusion(self) -> float:
        xs = jnp.linspace(0, 1, 1000)
        return jax.vmap(self.diffusion)(xs).mean()

    def __post_init__(self) -> None:
        super().__post_init__()
        self._avg_diffusion = self._estimate_avg_diffusion()

    def _diffusion_t(self, x, t):
        return (
            (1 - self.alpha(t)) * self.diffusion(x) / self._avg_diffusion
        ) + self.alpha(t)

    def _ln_diffusion_t(self, x, t):
        return jnp.log(self._diffusion_t(x, t)).sum()

    def score(
        self,
        x: Float[ArrayLike, 'n_samples n_features'],
        t: float,
        y: None | Float[ArrayLike, ''] = None,
    ) -> Float[ArrayLike, 'n_samples n_features']:
        if self.sigma(t) == 0:  # catch division by zero
            return np.zeros_like(x)

        score_times_minus_sigma = jax.vmap(
            self._score_eq, in_axes=(None, 0, 0),
        )(self.params, x, jnp.full((len(x), 1), t)) if y is None else jax.vmap(
            self._score, in_axes=(None, 0, 0, 0),
        )(self.params, x, jnp.full((len(x), 1), t), jnp.full((len(x), 1), y))  # fmt: skip

        return -score_times_minus_sigma / self.sigma(t)

    def _score(
        self,
        params: optax.Params,
        x: Float[ArrayLike, ' n_features'],
        t: float,
        y: None | Float[ArrayLike, ' n_features'] = None,
    ) -> Float[ArrayLike, '']:
        r"""Diffusion score.

        $$
        s_\theta = \nabla_x \ln p_t
        $$
        """
        return self._score_and_energy(params, x, t, y)[0]

    def _score_eq(
        self,
        params: optax.Params,
        x: Float[ArrayLike, ' n_features'],
        t: float,
    ) -> Float[ArrayLike, '']:
        r"""Diffusion score.

        $$
        s_\theta = \nabla_x \ln p_t
        $$
        """
        return -jax.grad(
            self._energy_eq,
            argnums=1,
        )(params, x, t)

    def _energy(
        self,
        params: optax.Params,
        x: Float[ArrayLike, ' n_features'],
        t: float,
        y: Float[ArrayLike, ' n_features'],
    ) -> Float[ArrayLike, '']:
        r"""Energy, aka. negative log-likelihood.

        $$
        \begin{aligned}
        \nabla_x E_\theta &= - s_\theta\\
        \Rightarrow -\sigma_t E_\theta &= -\ln p + C\\
        \end{aligned}
        $$
        """
        work = y * x

        if self.pbc_bins == 0:
            pbc_correction = 0
        else:
            xs = jnp.linspace(x.sum(), x.sum() + 1, self.pbc_bins)
            dx = xs[1] - xs[0]
            energies = jax.vmap(
                self._energy_eq,
                in_axes=(None, 0, 0),
            )(params, xs.reshape(-1, 1), jnp.full((len(xs), 1), t))
            U_eff = -energies / self.sigma(t) - self.alpha_force(t) * y * xs
            # mimic trapezoid weights
            w = jnp.ones_like(xs)
            w = w.at[0].set(0.5).at[-1].set(0.5)
            pbc_correction = jax.scipy.special.logsumexp(
                U_eff,
                b=w,
                axis=0,
            ) + jnp.log(dx)

        # use _energy_eq here
        return jnp.sum(
            (1 - self.alpha(t)) * self.score_model.apply(params, x, t)
            - self.sigma(t)
            * (
                self._ln_diffusion_t(x, t) - self.alpha_force(t) * work - pbc_correction
            ),
        )

    def _energy_eq(
        self,
        params: optax.Params,
        x: Float[ArrayLike, ' n_features'],
        t: float,
    ) -> Float[ArrayLike, '']:
        r"""Energy, aka. negative log-likelihood.

        $$
        \begin{aligned}
        \nabla_x E_\theta &= - s_\theta\\
        \Rightarrow E_\theta &= -\ln p + C\\
        \end{aligned}
        $$
        """
        return jnp.sum(
            (1 - self.alpha(t)) * self.score_model.apply(params, x, t),
        )

    def energy(
        self,
        x: Float[ArrayLike, 'n_samples n_features'],
        t: float,
        y: None | Float[ArrayLike, ''] = None,
    ) -> Float[ArrayLike, ' n_samples']:
        # catch division by zero
        if isinstance(t, float) and self.sigma(t) == 0:
            return np.zeros_like(x)

        energy_times_minus_sigma = (
            jax.vmap(
                self._energy_eq,
                in_axes=(None, 0, 0),
            )(self.params, x, jnp.full((len(x), 1), t))
            if y is None
            else jax.vmap(
                self._energy,
                in_axes=(None, 0, 0, 0),
            )(self.params, x, jnp.full((len(x), 1), t), jnp.full((len(x), 1), y))
        )

        return -energy_times_minus_sigma / self.sigma(t) + jax.vmap(
            self._ln_diffusion_t,
        )(x, jnp.full((len(x), 1), t))

    def _score_and_energy(
        self,
        params: optax.Params,
        x: Float[ArrayLike, ' n_features'],
        t: Float[ArrayLike, ''],
        y: Float[ArrayLike, ' n_features'],
    ) -> Float[ArrayLike, '']:
        r"""Diffusion score and energy.

        $$
        \begin{aligned}
        s_\theta &= \nabla_x \ln p_t\\
        E_\theta &= -\ln p + C
        \end{aligned}
        $$
        """
        energy, negative_score = jax.value_and_grad(
            self._energy,
            argnums=1,
        )(params, x, t, y)

        return -negative_score, energy

    @cached_property
    def score_model(self) -> float:
        mlp = ScoreSymmetricPeriodicMLP if self.symmetric else ScorePeriodicMLP
        return mlp(
            self.mlp_network,
            fourier_features_stop=self.fourier_features,
        )

    def _create_loss_fn(self):
        def loss_fn(
            params: optax.Params,
            key: JaxKey,
            X: Float[ArrayLike, 'n_samples n_features'],
            y: None | Float[ArrayLike, ' n_features'] = None,
        ):
            key1, key2 = jax.random.split(key)
            t = jax.random.uniform(key1, (len(X), 1))
            eps = jax.random.normal(key2, X.shape)
            x_t = self.prior_x_t(x=X, t=t, eps=eps)

            score_times_minus_sigma_pred = jax.vmap(
                self._score,
                in_axes=(None, 0, 0, 0),
            )(params, x_t, t, y)

            dt_energy = jax.vmap(
                jax.grad(
                    lambda x, t: -self._energy_eq(
                        params,
                        x,
                        t,
                    )
                    / self.sigma(t).mean(),
                    argnums=1,
                )
            )(x_t, t)

            return jnp.mean(
                jnp.min(
                    jnp.array([
                        jnp.abs(score_times_minus_sigma_pred - eps) % 1,
                        jnp.abs(score_times_minus_sigma_pred - eps),
                    ]),
                    axis=0,
                )
                ** 2,
            ) + self.gamma_energy_regulariztion * (jnp.mean(dt_energy**2))

        return loss_fn

    def _create_update_step(self, optim):
        loss_fn = self._create_loss_fn()

        @jax.jit
        def update_step(
            key: JaxKey,
            params: optax.Params,
            opt_state: optax.OptState,
            X: Float[ArrayLike, 'n_samples n_features'],
            y: None | Float[ArrayLike, ' n_features'] = None,
        ):
            loss_and_grad_fn = jax.value_and_grad(
                lambda p: loss_fn(params=p, key=key, X=X, y=y),
            )
            loss, grad = loss_and_grad_fn(params)
            updates, opt_state = optim.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        return update_step

    def _get_config(
        self,
        lrs: Float[ArrayLike, '2'],
        key: JaxKey,
        n_epochs: int,
        X: Float[ArrayLike, 'n_samples n_features'],
        y: None | Float[ArrayLike, ' n_features'] = None,
    ) -> dict:
        return {
            'learning_rates': lrs,
            'key': key,
            'n_epochs': n_epochs,
            'warmup_steps': self.warmup_steps,
            'n_samples': X.shape[0],
            'mlp_network': self.mlp_network,
            'noise_schedule': self._noise_schedule,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'batch_size': self.batch_size,
            'n_sample_steps': self.n_sample_steps,
            'prior': self._prior,
            'prior_schedule': self._prior_schedule,
            'class': self.__class__.__name__,
            'gamma_energy_regulariztion': self.gamma_energy_regulariztion,
            'fourier_features': self.fourier_features,
            'box_size': self.box_size,
            'symmetric': self.symmetric,
            'force_schedule': self._force_schedule,
            'forces': jnp.unique(y).tolist(),
            'pbc_bins': self.pbc_bins,
        }

    def train(
        self,
        X: Float[ArrayLike, 'n_samples n_features'],
        y: Float[ArrayLike, ' n_features'],
        lrs: Float[ArrayLike, '2'],
        key: None | JaxKey = None,
        n_epochs: None | int = None,
        X_val: None | Float[ArrayLike, 'n_val n_features'] = None,
        y_val: None | Float[ArrayLike, ' n_features'] = None,
        project: str = 'entropy-prod-diffusion',
        wandb_kwargs: dict = {},
    ):
        if X.ndim == 1:
            raise ValueError('X must be 2D array.')

        # if self.wandb_log and dataset is None:
        #    raise ValueError('Please provide a dataset for logging.')

        if key is None:
            key = self.key

        if n_epochs is None:
            n_epochs = self.n_epochs

        self.dim: int = X.shape[-1]

        # start a new wandb run to track this script
        if self.wandb_log:
            wandb.init(
                project=project,
                config=self._get_config(
                    lrs=lrs,
                    key=key,
                    n_epochs=n_epochs,
                    X=X,
                    y=y,
                    wandb_kwargs=wandb_kwargs,
                ),
            )

        # main logic
        loss_hist = self._train(
            X=X,
            lrs=lrs,
            key=key,
            n_epochs=n_epochs,
            y=y,
            X_val=X_val,
            y_val=y_val,
        )

        if self.wandb_log:
            wandb.finish()
        return loss_hist

    def _train(
        self,
        X: Float[ArrayLike, 'n_samples n_features'],
        y: Float[ArrayLike, ' n_features'],
        lrs: Float[ArrayLike, '2'],
        key: JaxKey,
        n_epochs: int,
        X_val: None | Float[ArrayLike, 'n_val n_features'],
        y_val: None | Float[ArrayLike, ' n_features'],
    ):
        self.params: optax.Params = self.score_model.init(
            key,
            t=jnp.ones(1),
            x=jnp.ones(self.dim),
        )
        n_batches = len(X) // self.batch_size
        schedule = optax.schedules.warmup_cosine_decay_schedule(
            warmup_steps=self.warmup_steps * n_batches,
            init_value=np.min(lrs),
            peak_value=np.max(lrs),
            decay_steps=n_epochs * n_batches,
            end_value=np.min(lrs),
        )
        optim = optax.adamw(learning_rate=schedule)
        opt_state: optax.OptState = optim.init(self.params)

        update_step = self._create_update_step(optim)
        loss_fn = self._create_loss_fn()

        ds = jdl.ArrayDataset(X, y)

        loss_hist = np.zeros(n_epochs)
        val_loss_hist = np.zeros(n_epochs) if X_val is not None else None
        for idx in (pbar := tqdm(range(n_epochs), leave=not self.wandb_log)):
            train_batches = jdl.DataLoader(
                ds,
                'jax',
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            total_loss = 0
            for batches in train_batches:
                key, _ = jax.random.split(key)
                loss, self.params, opt_state = update_step(
                    key,
                    self.params,
                    opt_state,
                    *batches,
                )
                total_loss += loss * self.batch_size / X.shape[0]
            loss_hist[idx] = total_loss
            if X_val is not None:
                # compute validation loss once per epoch
                val_loss = loss_fn()(params=self.params, key=key, X=X_val, y=y_val)
                val_loss_hist[idx] = float(val_loss)
            loss_min = loss_hist[: idx + 1].min()

            pbar.set_description(
                f'loss={total_loss:.4g}/{loss_min:.4g}',
            )
            if self.wandb_log:
                # Log the training loss
                wandb.log(
                    {'Loss': loss_hist[idx]},
                    step=idx + 1,
                )

        if self.wandb_log:
            key, _ = jax.random.split(key)
            xs = jnp.linspace(0, 1, 200)
            e_pred = self.energy(xs.reshape(-1, 1), t=0.0)

            # Log the plot
            data = [[x, y] for (x, y) in zip(xs, e_pred - e_pred.mean())]
            table = wandb.Table(data=data, columns=['x', 'U'])
            wandb.log(
                {
                    'U_pred_id': wandb.plot.line(
                        table, 'x', 'U', title='Free Energy Landscape'
                    ),
                },
                step=idx + 1,
            )

        # return both train and val loss history
        return {'train_loss': loss_hist, 'val_loss': val_loss_hist}

    def evaluate(
        self,
        X: Float[ArrayLike, 'n_samples n_features'],
        y: None | Float[ArrayLike, ' n_features'] = None,
        key: None | JaxKey = None,
    ) -> float:
        """Compute the same loss used in training on held-out data."""
        if key is None:
            key = self.key
        loss_fn = self._create_loss_fn()
        return float(loss_fn(params=self.params, key=key, X=X, y=y))

    def sample(
        self,
        key: JaxKey,
        n_samples: int,
        t_final: float = 0,
        n_steps: None | int = None,
    ) -> Float[ArrayLike, 'n_samples n_dims']:
        x_init = self.prior_sample(key, (n_samples, self.dim))
        if n_steps is None:
            n_steps = self.n_sample_steps
        dt = (1 - t_final) / n_steps
        t_array = jnp.linspace(1, t_final, n_steps + 1)

        def body_fn(i, val):
            x, key = val
            key, subkey = jax.random.split(key)
            t_curr = t_array[i]
            eps = jax.random.normal(subkey, x.shape)

            score_times_minus_sigma = jax.vmap(
                self._score_eq,
                in_axes=(None, 0, 0),
            )(self.params, x, jnp.full((len(x), 1), t_curr))
            score = -score_times_minus_sigma / self.sigma(t_curr)

            x_new = (
                x
                + self.beta(t_curr) * score * dt
                + jnp.sqrt(self.beta(t_curr)) * eps * jnp.sqrt(dt)
            )
            if self.is_periodic:
                x_new = x_new % 1
            return (x_new, key)

        final_x, _ = jax.lax.fori_loop(
            0,
            n_steps + 1,
            body_fn,
            (x_init, key),
        )
        return final_x
