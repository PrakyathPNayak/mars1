# DROPO: Sim-to-Real Transfer with Offline Domain Randomization

**Authors:** Gabriele Tiboni, Karol Arndt, Ville Kyrki
**Year:** 2023 | **Venue:** Robotics and Autonomous Systems
**Links:** [Paper](https://www.sciencedirect.com/science/article/pii/S0921889023000714)

---

## Abstract Summary
DROPO (Domain Randomization Off-Policy Optimization) introduces a principled approach to sim-to-real transfer by replacing hand-tuned uniform domain randomization with a likelihood-based optimization of randomization distributions. Rather than randomly sampling simulation parameters from broad uniform ranges, DROPO leverages a small number of real-world trajectories to estimate the posterior distribution over simulation parameters that best explains observed real-world behavior.

The method formulates domain randomization as an inference problem: given real-world state transitions, what distribution over simulator parameters maximizes the likelihood of reproducing those transitions? DROPO uses a Bayesian optimization approach with Gaussian process surrogate models to efficiently search the space of randomization distribution parameters (means and variances of each simulation parameter). This produces tightly calibrated randomization ranges that focus training on the most task-relevant parameter variations.

Experiments on manipulation and locomotion tasks demonstrate that DROPO significantly outperforms both uniform domain randomization and system identification approaches. The method requires only 5-10 real-world trajectories (collected with a random or simple policy), making it practical for hardware-constrained settings where extensive real-world data collection is infeasible.

## Core Contributions
- Formulation of domain randomization as a probabilistic inference problem with likelihood-based optimization over randomization distributions
- Offline method requiring only a small set of real-world trajectories (5-10) to calibrate randomization parameters
- Bayesian optimization with Gaussian process surrogates for efficient search over high-dimensional randomization parameter spaces
- Demonstrates that optimized randomization distributions significantly outperform uniform randomization on sim-to-real locomotion and manipulation
- Theoretical analysis showing that tighter, data-driven randomization distributions reduce policy conservatism while maintaining robustness
- Compatible with any RL algorithm (PPO, SAC, TD3) as it operates on the environment parameter level
- Open-source implementation facilitating adoption

## Methodology Deep-Dive
DROPO operates in three phases. Phase 1 involves collecting a small dataset of real-world trajectories D_real = {τ₁, ..., τ_N} using either a random policy or a simple hand-crafted controller. Each trajectory consists of state-action-next_state tuples (s_t, a_t, s_{t+1}). Only 5-10 trajectories of 100-500 steps each are needed, totaling minutes of real-world interaction.

Phase 2 is the core optimization. DROPO parameterizes the randomization distribution as a product of independent Gaussians: φ = {(μᵢ, σᵢ²)} for each simulation parameter i (friction, mass, damping, motor gains, etc.). The objective is to maximize the log-likelihood of real-world transitions under the simulator with parameters drawn from the distribution: max_φ E_{ξ~p(ξ|φ)} [Σ_t log p(s_{t+1} | s_t, a_t, ξ)]. Since this objective is not differentiable through the simulator, DROPO employs Bayesian optimization with a Gaussian process surrogate. The GP models the log-likelihood surface over φ, and an acquisition function (Expected Improvement) selects the next φ to evaluate. Each evaluation involves sampling K parameter vectors from p(ξ|φ), running the simulator forward for each trajectory in D_real, and computing the average log-likelihood.

Phase 3 uses the optimized distribution p*(ξ|φ*) for domain randomization during RL training. At the start of each episode, simulation parameters are sampled from p* rather than broad uniform distributions. This focuses training on the most relevant parameter variations, reducing the policy's need to be overly conservative while still maintaining robustness to real-world conditions.

The log-likelihood computation uses a multivariate Gaussian observation model: p(s_{t+1} | s_t, a_t, ξ) = N(s_{t+1}; f(s_t, a_t, ξ), Σ_obs), where f is the simulator transition function and Σ_obs is a diagonal noise covariance estimated from the data. This formulation gracefully handles simulator inaccuracies by absorbing them into the noise term.

A key design choice is the independence assumption between simulation parameters in the randomization distribution. While this limits expressiveness, it dramatically reduces the dimensionality of the optimization problem (2D per parameter instead of D² for full covariance), making Bayesian optimization tractable for 10-20 simulation parameters.

## Key Results & Numbers
- Locomotion tasks: DROPO policies achieve 40-60% higher reward on real hardware compared to uniform domain randomization
- Manipulation tasks: 85% success rate vs 55% for uniform DR and 70% for system identification
- Only 5-10 real-world trajectories needed (vs hundreds for system identification methods)
- Optimized randomization distributions are 3-5x narrower than typical uniform ranges, reducing training conservatism
- Bayesian optimization converges in 100-200 evaluations (each taking ~30 seconds of simulation)
- Total calibration time: ~1-2 hours of computation after 5-10 minutes of real-world data collection
- Policies trained with DROPO show 25% less energy consumption on real hardware due to reduced conservatism

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
DROPO is highly relevant to Project A's sim-to-real pipeline for the Mini Cheetah. Currently, domain randomization ranges are likely hand-tuned or based on engineering intuition. DROPO offers a systematic, data-driven alternative that could significantly improve transfer quality. Collecting 5-10 trajectories on the Mini Cheetah hardware (even with a simple PD controller) would provide sufficient data to optimize randomization distributions for friction, mass, motor gains, joint damping, and ground compliance.

The method's compatibility with PPO means it can be directly integrated into Project A's existing training pipeline without changing the RL algorithm. The reduced conservatism from tighter randomization distributions could lead to more dynamic and energy-efficient gaits on the real Mini Cheetah. The MuJoCo simulator used in Project A is well-suited for DROPO's likelihood computation since MuJoCo's physics are differentiable and deterministic given parameters.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
For Project B's Cassie platform, DROPO addresses a critical challenge in the Adversarial Curriculum training pipeline. Rather than uniformly randomizing all simulation parameters during adversarial environment generation, DROPO can provide data-driven priors that ensure the adversarial curriculum focuses on realistic parameter variations. This is particularly important for Cassie's complex dynamics where overly broad randomization can lead to physically implausible training scenarios.

DROPO's offline nature is valuable since collecting real-world Cassie data is expensive and time-consuming. The method's ability to work with just 5-10 trajectories makes it practical for the Cassie platform. The optimized randomization distributions can be used at every level of the hierarchy: the Planner level can use them for high-level planning robustness, the Controller level for joint-tracking robustness, and the Safety level (LCBF) can benefit from tighter parameter bounds for more precise CBF constraint formulation.

## What to Borrow / Implement
- Implement DROPO's likelihood-based optimization to replace hand-tuned domain randomization ranges in both projects
- Collect a small dataset of real-world trajectories from Mini Cheetah with a simple controller for DROPO calibration
- Use DROPO-optimized distributions as the base distribution for Project B's Adversarial Curriculum, with adversarial perturbations applied on top
- Integrate Bayesian optimization pipeline (GPyTorch or BoTorch) for automatic randomization parameter tuning
- Apply DROPO's noise covariance estimation to quantify simulator fidelity gaps for prioritizing simulator improvements

## Limitations & Open Questions
- Independence assumption between simulation parameters may miss correlated real-world effects (e.g., friction-mass coupling on slopes)
- Gaussian assumption on randomization distributions may be inadequate for multi-modal real-world parameter distributions
- Method requires a reasonably accurate base simulator; if simulator physics are fundamentally wrong, optimized randomization cannot compensate
- Scalability to very high-dimensional parameter spaces (>50 parameters) untested; Bayesian optimization may struggle
