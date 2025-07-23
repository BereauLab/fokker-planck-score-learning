r"""Denoising Diffusion Models (DDM) for score-based generative modeling.

This submodule provides a comprehensive suite of components for building and
training score-based denoising diffusion models, with a focus on periodic data
and force-conditioned sampling.

The submodule is structured into the following submodules:

- [**models:**][fpsl.ddm.models]
    Core FPSL (Fokker-Planck Score Learning) diffusion model implementation
    for learning score functions and generating samples.
- [**network:**][fpsl.ddm.network]
    Neural network architectures including MLPs with Fourier feature embeddings
    for score function approximation on periodic domains.
- [**noiseschedule:**][fpsl.ddm.noiseschedule]
    Time-dependent noise scheduling functions that control the variance and
    diffusion coefficients during the forward/reverse processes.
- [**prior:**][fpsl.ddm.prior]
    Latent prior distribution definitions, including uniform priors with
    periodic boundary conditions for circular/toroidal data.
- [**priorschedule:**][fpsl.ddm.priorschedule]
    Interpolation schedules between data and prior distributions that control
    the mixing coefficient $\alpha(t)$ during diffusion.
- [**forceschedule:**][fpsl.ddm.forceschedule]
    Force conditioning schedules that control how external forces influence
    the diffusion process through time-dependent scaling factors.
"""

__all__ = [
    'FPSL',
]

from .models import FPSL
