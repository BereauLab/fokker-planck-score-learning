# Theoretical Background

## Overview

Fokker-Planck Score Learning (FPSL) is a novel approach that combines the analytical solution of the Fokker-Planck equation for periodic systems with score-based diffusion models to reconstruct free energy landscapes from non-equilibrium data. This method is particularly powerful for systems with periodic boundary conditions, where conventional free energy estimation methods often struggle.

## Steady-State Solution of the Fokker-Planck Equation

### Non-Equilibrium Steady States in Periodic Systems

Consider a Brownian particle in a periodic potential $U(x)$ with period $L$, subject to a constant external driving force $f$. The system is governed by the overdamped Langevin equation:

$$\frac{dx}{dt} = -\beta D \nabla U(x) + f + \sqrt{2D}\xi(t)$$

where $\beta = (k_B T)^{-1}$ is the inverse temperature, $D$ is the diffusion coefficient, and $\xi(t)$ represents Gaussian white noise.

The effective potential under the driving force becomes:

$$U_{\text{eff}}(x) = U(x) - fx$$

### The Fokker-Planck Steady-State Solution

For a periodic system with period $L$, the non-equilibrium steady-state (NESS) distribution has the remarkable analytical form:

$$p^{\text{s}}(x) \propto \frac{1}{D(x)} e^{-\beta U_{\text{eff}}(x)} \int_{x}^{x+L} dy \, e^{\beta U_{\text{eff}}(y)}$$

This expression consists of two key components:

1. **Local Boltzmann factor**: the standard equilibrium weighting, $e^{-\beta U_{\text{eff}}(x)}$
2. **Periodic correction integral**: accounts for the periodic boundary conditions and ensures proper normalization, $\int_{x}^{x+L} dy \, e^{\beta U_{\text{eff}}(y)}$

For the derivation of this expression, we refer to our paper on [arXiv:2506.15653](https://arxiv.org/abs/2506.15653).


## Diffusion Models on Periodic Domains

Denoising diffusion models learn to generate samples from a target distribution by reversing a gradual noising process. Since we consider periodic systems, using a uniform prior is essential to respect the periodic topology. This leads to the simplified forward process:

$$dx_\tau = \sqrt{2\alpha_\tau} dW_\tau$$

where $\tau \in [0,1]$ is the diffusion time, $\alpha_\tau$ is a noise schedule. With the corresponding reverse process defined as:

$$dx_\tau = -2\alpha_\tau \nabla \ln p_\tau(x_\tau) d\tau + \sqrt{2\alpha_\tau} d\bar{W}_\tau$$

The key to diffusion models is learning the **score function**:

$$s(x_\tau, \tau) = \nabla \ln p_\tau(x_\tau)$$

This score guides the reverse diffusion process that transforms noise back into data samples.

## Fokker-Planck Score Learning: The Core Idea

### Using Physical Insights as Inductive Bias

The central innovation of FPSL is to use the analytical NESS solution as an **ansatz** for the score function in the diffusion model. Instead of learning the steady-state, we learn the equilibrium distribution from the steady-state samples.

### The FPSL Score Function

The score function in FPSL combines the standard energy-based score with a periodic correction term:

$$
\begin{aligned}
s^\theta(x_\tau, \tau, L)
&= \nabla \ln p^{\text{ss}}(x_\tau, \tau, L)\\
&= - \beta \nabla U^\theta_{\text{eff}}(x_\tau, \tau) - \nabla \ln D(x) + \Delta s^\theta(x_\tau, \tau, L)
\end{aligned}$$

where the periodic correction is:

$$\Delta s^\theta(x_\tau, \tau, L) = \nabla \ln \int_{x_\tau}^{x_\tau + L} dy \, e^{\beta U^\theta_{\text{eff}}(y, \tau)}$$

In the main paper, we show that in our case, this correction is negligible, if we enforce the network to learn a periodic potential.

For a more detailed discussion on the periodic correction and its implications, please refer to the main paper.