# Learning Quadruped Locomotion Using Differentiable Simulation

**Authors:** Song et al. (University of Zurich & MIT)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv:2403.14864](https://arxiv.org/abs/2403.14864)

---

## Abstract Summary
This paper presents the first successful real-world deployment of a quadruped locomotion policy trained entirely through differentiable simulation. Unlike conventional RL approaches (e.g., PPO) that rely on zeroth-order stochastic gradient estimates, differentiable simulation provides exact analytical gradients of the return with respect to policy parameters, enabling dramatically faster convergence. The authors address the fundamental challenge that rigid-body contact dynamics are inherently non-differentiable by decomposing the simulation into continuous, differentiable sub-domains.

The method learns diverse gaits—trot, pace, bound, and gallop—in minutes of wall-clock training time on a single GPU, compared to the hours or days required by model-free RL. The trained policies transfer zero-shot from simulation to a real quadruped robot without any fine-tuning or additional domain randomization beyond what the differentiable simulator provides. This represents a paradigm shift in locomotion learning: from sample-inefficient trial-and-error to gradient-efficient optimization.

The key technical innovation lies in handling contact discontinuities. The authors propose a smooth contact model that replaces hard contact constraints with compliant penalty-based forces, enabling gradient flow through contact events while maintaining physically realistic behavior. Combined with automatic differentiation through the simulation rollout, this yields end-to-end differentiable trajectories suitable for first-order optimization.

## Core Contributions
- First real-world deployment of quadruped locomotion trained entirely via differentiable simulation
- Novel decomposition of non-differentiable contact dynamics into continuous, differentiable sub-problems
- Achieves training convergence in minutes vs. hours for PPO, representing 50–100× speedup in sample efficiency
- Demonstrates zero-shot sim-to-real transfer without additional domain randomization
- Learns multiple gait patterns (trot, pace, bound, gallop) from a unified optimization framework
- Provides ablation studies comparing gradient quality and convergence against PPO and evolutionary strategies
- Open-source implementation enabling reproducibility

## Methodology Deep-Dive
The differentiable simulator is built on a compliant contact model where contact forces are computed as smooth functions of penetration depth and relative velocity. Specifically, contact normal forces follow a spring-damper model: F_n = k_n * max(0, δ)^α - d_n * v_n, where δ is penetration depth, v_n is normal velocity, and k_n, d_n, α are tunable parameters. Friction forces use a smooth Coulomb approximation with a regularized tangent function to avoid the non-differentiable transition between static and kinetic friction. This smoothing introduces a small sim-to-real gap but enables gradient computation.

The policy is parameterized as a feedforward neural network (typically 2–3 hidden layers, 128–256 units, ELU activations) mapping proprioceptive observations (joint positions, velocities, body orientation, angular velocity) to target joint positions. A PD controller at 500 Hz converts target positions to torques. The entire forward pass—observation → policy → PD → simulation step—is differentiable, enabling backpropagation through time (BPTT) over the full episode (typically 2–4 seconds, 1000–2000 simulation steps).

Gradient computation uses reverse-mode automatic differentiation through the unrolled simulation. To manage memory consumption, the authors employ checkpointing: only every k-th simulation state is stored, with intermediate states recomputed during the backward pass. This reduces memory from O(T) to O(T/k) at the cost of ~2× computation. For a 2-second episode at 500 Hz (1000 steps), checkpointing every 50 steps reduces memory by 50×.

The reward function includes terms for forward velocity tracking, orientation stability (penalizing roll/pitch deviation), energy efficiency (minimizing torque squared), smoothness (penalizing action rate-of-change), and gait-specific phase constraints. Because gradients are exact, the optimizer (Adam, learning rate 1e-3 to 1e-4) converges in 200–500 iterations, each processing a batch of 64–256 parallel rollouts.

For sim-to-real transfer, the authors apply light domain randomization to friction coefficients (±20%), body mass (±10%), and motor strength (±15%). Importantly, the amount of randomization required is substantially less than for PPO-trained policies, because the differentiable simulator already provides a more accurate learning signal.

## Key Results & Numbers
- Training time: 3–10 minutes on a single NVIDIA A100 GPU vs. 2–8 hours for PPO
- Sample efficiency: 50–100× fewer environment interactions than PPO to reach equivalent performance
- Zero-shot sim-to-real transfer success rate: >90% for trot, >85% for pace and bound gaits
- Forward velocity tracking error: <5% for trot gait at target speeds of 0.5–1.5 m/s
- Energy efficiency: 15–25% lower cost-of-transport compared to PPO-trained policies
- Gait diversity: Successfully learns trot, pace, bound, and gallop from identical network architecture
- Gradient accuracy: Mean cosine similarity >0.85 between analytical and finite-difference gradients

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Critical**
This paper is among the most directly relevant to the Mini Cheetah project. The Mini Cheetah's 12-DoF configuration, MuJoCo simulation environment, and PPO training pipeline could be augmented or replaced by differentiable simulation for dramatically faster iteration cycles. The 50–100× training speedup would transform the development workflow: experiments that currently take hours could complete in minutes, enabling rapid hyperparameter sweeps and reward function exploration.

The zero-shot sim-to-real transfer results are particularly compelling. The Mini Cheetah project currently relies on extensive domain randomization and curriculum learning to bridge the sim-to-real gap; differentiable simulation's exact gradients may reduce this burden significantly. The gait diversity demonstrated (trot, pace, bound, gallop) directly maps to the Mini Cheetah's locomotion repertoire goals. The compliant contact model could be integrated into MuJoCo via the MJX differentiable backend.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Differentiable simulation could accelerate training of Cassie's low-level Controller module, which currently uses PPO for joint-level torque generation. The gradient-based optimization would be especially valuable for the differentiable capture point computation in Cassie's architecture, where accurate gradients through contact dynamics are essential for balance control.

The memory-efficient checkpointing strategy is directly applicable to Cassie's longer training horizons (bipedal balance requires longer episodes). The smooth contact model could improve gradient quality for Cassie's foot-ground interactions, which are critical for the CBF-QP safety layer's constraint computations. However, the hierarchical nature of Cassie's architecture means differentiable simulation would primarily benefit the lower levels.

## What to Borrow / Implement
- Integrate MuJoCo MJX differentiable backend for Mini Cheetah training with exact gradients
- Adopt the compliant contact model parameters as starting points for smooth gradient computation
- Use the checkpointing strategy to manage memory during BPTT over long locomotion episodes
- Apply the reduced domain randomization approach (±10–20% vs. ±50%) enabled by exact gradients
- Benchmark differentiable simulation training time against current PPO pipeline on identical hardware

## Limitations & Open Questions
- Compliant contact model introduces a small but non-zero sim-to-real gap compared to hard contacts
- BPTT through long horizons can suffer from vanishing/exploding gradients despite smooth contacts
- Limited to locomotion on relatively flat terrain; complex terrain with many contact mode switches may degrade gradient quality
- Requires custom differentiable simulator implementation rather than drop-in replacement for existing MuJoCo pipelines
