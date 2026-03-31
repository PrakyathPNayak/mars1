# Neural Ordinary Differential Equations

**Authors:** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
**Year:** 2018 | **Venue:** NeurIPS (Best Paper)
**Links:** https://arxiv.org/abs/1806.07366

---

## Abstract Summary
This paper introduces Neural Ordinary Differential Equations (Neural ODEs), a fundamentally new family of deep learning models that parameterize the continuous dynamics of hidden states using ordinary differential equations. Instead of specifying a discrete sequence of hidden layers, a Neural ODE defines the derivative of the hidden state as a neural network `dh/dt = f(h(t), t, θ)`, and the output is obtained by solving this ODE using a black-box numerical integrator (e.g., Dormand-Prince RK45). This replaces the discrete layer-by-layer forward pass with a continuous-depth transformation.

A key innovation is the adjoint sensitivity method for memory-efficient training. Standard backpropagation through an ODE solver would require storing all intermediate solver states, consuming memory proportional to the number of solver steps. The adjoint method instead solves a second ODE backward in time to compute gradients, reducing memory cost to O(1) regardless of the number of forward solver steps. This makes it feasible to train deep continuous-depth models without the memory bottleneck of very deep residual networks.

The paper demonstrates three applications: (1) continuous-depth residual networks that match standard ResNets with constant memory cost, (2) continuous normalizing flows (CNFs) that enable exact log-likelihood computation without the architectural restrictions of discrete normalizing flows, and (3) a latent ODE model for irregularly-sampled time series that naturally handles variable time gaps between observations. The `torchdiffeq` library provides a PyTorch implementation of ODE solvers with adjoint-method backpropagation.

## Core Contributions
- **Neural ODE formulation:** Defined hidden state dynamics as `dh/dt = f(h(t), t, θ)` where `f` is a neural network, replacing discrete residual layers with a continuous-depth analog
- **Adjoint sensitivity method:** Enabled O(1) memory training by computing gradients via a backward-in-time ODE solve, eliminating the need to store intermediate states
- **Continuous normalizing flows:** Showed that continuous-time flows allow unrestricted architectures (no invertibility constraints) while providing exact log-likelihood computation via the instantaneous change of variables formula
- **Latent ODE for time series:** Introduced a model that naturally handles irregularly-sampled time series by running an ODE in latent space between observations
- **Adaptive computation:** The ODE solver automatically determines the number of function evaluations based on the complexity of the dynamics, providing an adaptive depth mechanism
- **torchdiffeq library:** Released a production-quality PyTorch library implementing multiple ODE solvers (Euler, RK4, Dormand-Prince, Adams) with adjoint backpropagation support

## Methodology Deep-Dive
The core insight connects residual networks to ODE dynamics. A residual block computes `h_{t+1} = h_t + f(h_t, θ_t)`, which is an Euler discretization of `dh/dt = f(h(t), t, θ)` with step size 1. Neural ODEs take this analogy literally by defining the continuous dynamics and solving them with a proper numerical integrator. Given initial state `h(0) = x`, the output is `h(T) = h(0) + ∫₀ᵀ f(h(t), t, θ) dt`, computed by calling `odeint(f, h(0), [0, T])`.

The adjoint sensitivity method is the key enabler for scalable training. Define the adjoint state `a(t) = dL/dh(t)`. The authors show that `a(t)` satisfies its own ODE: `da/dt = -a(t)^T (∂f/∂h)`. Starting from `a(T) = dL/dh(T)` (computed from the loss), we solve this ODE backward from `T` to `0` to get gradients with respect to all parameters. Crucially, we also need `h(t)` during the backward pass, which is obtained by solving the forward ODE backward (leveraging reversibility). The parameter gradients are accumulated as `dL/dθ = -∫ₜ⁰ a(t)^T (∂f/∂θ) dt`. All three quantities (adjoint, state, parameter gradients) can be computed in a single backward ODE solve.

For continuous normalizing flows, the paper leverages the instantaneous change of variables formula: `∂ log p(h(t))/∂t = -tr(∂f/∂h(t))`. This avoids the Jacobian determinant computation required by discrete normalizing flows and removes the architectural constraint of requiring invertible transformations. The trace computation is estimated efficiently using Hutchinson's trace estimator: `tr(A) = E[ε^T A ε]` where `ε` is a random vector.

The latent ODE model for time series defines a latent state that evolves continuously according to learned dynamics. An RNN encoder processes observed (irregularly-timed) data points to produce an approximate posterior over the initial latent state. The generative model then evolves this state forward using the ODE dynamics, decoding observations at the required time points. This naturally handles missing data and irregular sampling without imputation or binning.

Numerical stability is addressed through adaptive step-size solvers (Dormand-Prince), which automatically take smaller steps when dynamics are stiff and larger steps when dynamics are smooth. The number of function evaluations (NFE) serves as a proxy for model complexity and computational cost, typically ranging from 20-100 for practical models.

## Key Results & Numbers
- MNIST classification: Neural ODE achieves comparable test error (0.42%) to a 6-layer ResNet (0.41%) while using constant O(1) memory vs. O(L) for ResNets
- NFE (number of function evaluations) during forward pass: ~70-100 on MNIST, adapting to input complexity
- Continuous normalizing flows on miniboost and 2D densities: log-likelihood improvements over FFJORD baseline
- Latent ODE on PhysioNet (irregularly-sampled clinical time series): achieves lower MSE than GRU-based baselines, especially with >50% missing data
- Training time: ~2-4x slower than equivalent discrete models due to ODE solver overhead, but memory savings enable much deeper effective models
- Backward pass (adjoint): 1.5x the cost of forward pass, comparable to standard backprop ratios

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
Neural ODEs are not directly used in the Mini Cheetah RL pipeline. The standard PPO-based approach with MLP or recurrent policies does not require continuous-depth dynamics modeling. However, the concept of continuous-time dynamics could theoretically be applied to model the quadruped's state transitions more accurately than discrete time-step simulations, particularly for bridging sim-to-real gaps where the real robot operates in continuous time. This remains a speculative application and is not part of the current project scope.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This is THE foundational paper for the Neural ODE Gait Phase component in Project B. The Neural ODE Gait Phase module uses `torchdiffeq` to model continuous gait phase dynamics `dφ/dt = g(φ(t), s(t), θ)`, where `φ` is the gait phase variable and `s(t)` encodes proprioceptive state. This allows the gait phase to evolve continuously rather than being discretized into fixed bins, enabling smooth phase transitions and natural handling of variable walking speeds. The adjoint method enables memory-efficient training of this component within the larger hierarchical policy. The latent ODE formulation is especially relevant since gait phase is inherently a latent variable inferred from noisy proprioceptive observations. Key implementation detail: the `odeint_adjoint` function from `torchdiffeq` is used during training, while the faster (non-adjoint) `odeint` may be used at inference for speed.

## What to Borrow / Implement
- **Use `torchdiffeq.odeint_adjoint`** for memory-efficient training of the Neural ODE Gait Phase module; switch to `torchdiffeq.odeint` for deployment inference
- **Adaptive solver selection:** Start with `dopri5` (Dormand-Prince) for training stability; consider `euler` or `rk4` with fixed step size for real-time deployment on Cassie
- **NFE monitoring:** Track number of function evaluations during training as a diagnostic for dynamics complexity; if NFE grows excessively, add regularization (`kinetic_energy` regularization from FFJORD)
- **Latent ODE pattern for gait phase:** Model gait phase as a latent state with continuous ODE dynamics, decoded into observable gait events (heel strike, toe off)
- **Tolerance tuning:** Use `atol=1e-5, rtol=1e-5` for training, relax to `atol=1e-3, rtol=1e-3` for real-time inference to reduce solver steps

## Limitations & Open Questions
- ODE solver overhead makes Neural ODEs 2-4x slower than equivalent discrete networks; real-time deployment on Cassie's onboard compute requires careful solver choice and tolerance tuning
- The adjoint method can introduce numerical errors when the forward dynamics are chaotic or highly stiff, potentially affecting gait phase gradient computation
- Neural ODEs assume continuous dynamics, but contact events in bipedal walking (heel strike, toe off) are inherently discontinuous — the model must learn to approximate these through smooth dynamics
- Integration with the rest of the hierarchical policy (MC-GAT, Transformers, LCBF) requires careful gradient flow management to prevent vanishing/exploding gradients through the ODE solve
