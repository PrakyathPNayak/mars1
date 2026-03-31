# A Review of Differentiable Simulators

**Authors:** Newbury et al.
**Year:** 2024 | **Venue:** IEEE Access
**Links:** [arXiv:2407.05560](https://arxiv.org/abs/2407.05560)

---

## Abstract Summary
Newbury et al. provide a comprehensive survey of differentiable simulation for robotics, covering the mathematical foundations, computational implementations, and practical applications across manipulation, locomotion, and co-design. The review synthesizes over 150 papers to identify key design choices, trade-offs, and open challenges in building simulators that provide gradient information through physics-based dynamics.

The paper categorizes differentiable simulators along multiple axes: gradient computation method (analytical, automatic differentiation, finite difference), contact model (penalty-based, impulse-based, complementarity), supported physics (rigid body, soft body, fluid), and software framework (JAX, PyTorch, custom). For each category, the authors analyze gradient accuracy, computational cost, scalability, and suitability for different downstream tasks (policy optimization, system identification, trajectory optimization).

A central theme is the tension between physical accuracy and gradient quality. Hard contact models (complementarity-based) are more physically accurate but produce discontinuous gradients at contact mode transitions. Soft contact models (penalty-based) provide smooth gradients but introduce physical artifacts. The review identifies this as the fundamental open problem and surveys recent approaches to bridge this gap, including randomized smoothing, contact-implicit formulations, and hybrid methods.

## Core Contributions
- Comprehensive taxonomy of differentiable simulators covering 150+ papers across robotics applications
- Systematic comparison of gradient computation methods: analytical, automatic differentiation, and finite difference
- Analysis of contact model trade-offs: physical accuracy vs. gradient smoothness
- Survey of open-source tools: Brax, DiffTaichi, Nimble, Drake, Warp, MJX, Dojo
- Identification of key challenges: gradient accuracy through contacts, scalability, sim-to-real gap
- Practical guidelines for selecting differentiable simulation tools based on application requirements
- Discussion of integration strategies with model-free RL for hybrid approaches

## Methodology Deep-Dive
The review organizes differentiable simulators by their approach to the core computational challenge: computing ∂s_{t+1}/∂(s_t, a_t) where s_t is the simulation state and a_t is the action. Three paradigms are identified. **Analytical differentiation** derives gradient expressions by hand from the equations of motion, yielding exact and computationally efficient gradients but requiring significant mathematical effort for each new system. **Automatic differentiation (AD)** applies chain rule mechanically through the simulation code using frameworks like JAX or PyTorch, providing correctness guarantees with minimal manual effort but potentially high memory consumption for long rollouts. **Finite differences** approximate gradients via perturbation, requiring no code modification but scaling poorly with parameter dimensionality (O(n) forward passes for n parameters) and suffering from numerical precision issues.

Contact handling receives the deepest analysis, as it is the primary obstacle to useful differentiable simulation in robotics. The review identifies four contact approaches. **Penalty-based** methods model contacts as springs (F = kδ), providing C∞-smooth gradients but non-physical interpenetration. **Impulse-based** methods compute instantaneous velocity changes at contact, yielding physically accurate dynamics but discontinuous gradients at collision times. **Complementarity-based** methods formulate contact as a linear complementarity problem (LCP), providing exact solution structure but non-differentiable at contact mode boundaries. **Randomized smoothing** approaches average over stochastic perturbations to produce smooth expected gradients, maintaining physical accuracy at the cost of increased variance and computational cost.

For locomotion applications specifically, the review identifies penalty-based and randomized smoothing as the most practical approaches. Penalty-based methods (used by Brax, MJX) are simple to implement and efficient but require careful stiffness tuning. Randomized smoothing (used by recent works from CMU and MIT) provides physically accurate expected gradients but requires averaging over many samples, partially negating the efficiency advantage over model-free RL.

The review also covers integration strategies: (1) pure differentiable simulation for policy optimization via BPTT, (2) differentiable simulation as a learned world model for model-based RL, (3) hybrid approaches using differentiable simulation for warm-starting model-free RL, and (4) differentiable simulation for system identification and domain adaptation. Each strategy has distinct computational profiles and applicability conditions.

The analysis of open-source tools compares Brax (Google, JAX-based, penalty contacts, GPU-parallel), DiffTaichi (MIT, custom AD, multi-physics), Nimble (Stanford, analytical gradients, complementarity contacts), Drake (TRI, C++ with Python bindings), Warp (NVIDIA, GPU-optimized, penalty contacts), MJX (DeepMind, JAX-based MuJoCo port), and Dojo (MIT, complementarity with smoothing). Performance benchmarks show Brax and MJX scaling most effectively to thousands of parallel environments.

## Key Results & Numbers
- Analytical gradients: 2–5× faster than AD for fixed system topologies, but rigid to modify
- AD via JAX: scales to 8192 parallel environments with sub-linear memory growth using checkpointing
- Penalty-based contacts: gradient SNR 10–100× higher than impulse-based near contact events
- Randomized smoothing: requires 32–128 samples for low-variance gradients, partially offsetting efficiency gains
- BPTT through 1000-step rollouts: feasible with checkpointing, 2–3× wall-clock overhead vs. forward-only
- Differentiable simulation policy optimization: 10–100× more sample-efficient than PPO on locomotion tasks
- Sim-to-real gap: penalty-based models require 20–50% more domain randomization than complementarity-based
- Tool maturity: Brax and MJX rated highest for locomotion applications; Nimble for manipulation

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
This review serves as a practical guide for selecting and configuring differentiable simulation tools for the Mini Cheetah project. The comparison of MJX (MuJoCo's JAX backend) against other tools is directly relevant, as the Mini Cheetah already uses MuJoCo for simulation. The review's finding that penalty-based contacts (used by MJX) provide the best gradient quality for locomotion validates this choice while cautioning about the need for stiffness tuning.

The hybrid approach—using differentiable simulation for warm-starting PPO—could accelerate Mini Cheetah training without abandoning the proven PPO pipeline. The review provides concrete guidance on checkpointing strategies and learning rate schedules for BPTT that would be essential for implementation.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The review's analysis of contact gradient computation methods is essential for implementing Cassie's differentiable capture point (DCP) module. The comparison between penalty-based and randomized smoothing approaches directly informs the DCP's contact model choice: penalty-based for efficiency or randomized smoothing for physical accuracy. The review's finding that randomized smoothing requires 32–128 samples suggests a computational budget for the DCP module.

The integration strategies section is relevant to Cassie's hierarchical architecture: the DCP could use differentiable simulation internally while the higher-level Planner and Primitives layers use model-free RL. The review validates this hybrid approach and provides guidelines for managing the interface between differentiable and non-differentiable components.

## What to Borrow / Implement
- Use MJX as the differentiable simulation backend for Mini Cheetah, with penalty-based contacts
- Adopt the checkpointing strategy (every 50–100 steps) for memory-efficient BPTT
- Consider the hybrid approach: differentiable simulation for warm-starting, then PPO for fine-tuning
- Use gradient SNR as a diagnostic metric for contact model quality during development
- Evaluate randomized smoothing for Cassie's DCP if penalty-based gradients prove insufficient

## Limitations & Open Questions
- Survey focuses on simulation-only results; comprehensive sim-to-real comparisons across tools are lacking
- Rapidly evolving field means tool recommendations may become outdated quickly
- Limited coverage of soft-body and deformable terrain interactions relevant to outdoor deployment
- Computational overhead of differentiable simulation vs. model-free RL is highly hardware-dependent and not consistently benchmarked
