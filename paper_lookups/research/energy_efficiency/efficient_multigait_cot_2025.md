# Efficient Learning of Robust Multigait Quadruped Locomotion for Minimizing the Cost of Transport

**Authors:** (FITEE 2025)
**Year:** 2025 | **Venue:** Frontiers of IT & Electronic Engineering
**Links:** https://www.fitee.zjujournals.com/en/article/doi/10.1631/FITEE.2401070/

---

## Abstract Summary
This paper presents a comprehensive study on training reinforcement learning policies for multi-gait quadruped locomotion with the explicit objective of minimizing Cost of Transport (CoT). The work addresses the challenge of learning a single policy capable of walk, trot, and gallop gaits while achieving energy efficiency that approaches natural animal performance across a wide speed range. The authors propose a structured reward function with gait-specific auxiliary terms and a progressive training curriculum.

The key contribution is a unified training framework that produces a single neural network policy capable of executing multiple gaits, where the gait selection is driven by energy optimality rather than explicit commands. The policy is robust to perturbations including external pushes, payload variations, and terrain irregularities, maintaining energy-efficient locomotion under disturbances. Extensive simulation benchmarks compare the learned CoT against biological reference data from quadruped animals of similar scale.

The work also introduces an efficient training methodology that reduces sample complexity by leveraging gait-phase references as soft constraints during early training, which are gradually relaxed to allow the policy to discover optimal behaviors. This curriculum over constraint relaxation significantly accelerates convergence compared to learning from scratch.

## Core Contributions
- A unified single-policy framework for walk/trot/gallop that minimizes CoT across a continuous speed range (0–3.5 m/s)
- Structured reward decomposition with gait-phase auxiliary terms that serve as soft constraints during training and are progressively relaxed
- Training curriculum over constraint relaxation: gait references are strong early in training and fade to zero, allowing the policy to discover energy-optimal behaviors beyond the reference motions
- Comprehensive CoT analysis comparing learned policies against biological quadruped data, showing the policy approaches animal-level efficiency at each gait
- Robustness evaluation under perturbations (lateral pushes up to 60 N, payload ±3 kg, terrain noise ±3 cm) with minimal CoT degradation
- Ablation study demonstrating the importance of each reward component and the constraint relaxation curriculum
- Efficient training achieving convergence in ~4 hours with 2048 parallel environments

## Methodology Deep-Dive
The reward function is decomposed into four groups: (1) task rewards covering velocity tracking and orientation maintenance, (2) energy terms penalizing mechanical power and joint acceleration, (3) gait-phase auxiliary terms rewarding footfall patterns consistent with walk/trot/gallop reference trajectories, and (4) regularization terms penishing joint velocity limits, torque limits, and action oscillation.

The gait-phase auxiliary terms are the methodological centerpiece. For each gait, reference foot contact sequences are defined: walk uses a lateral-sequence pattern (LH→LF→RH→RF) with ~65% duty factor; trot uses diagonal pairing (LF+RH, RF+LH) with ~50% duty factor; gallop uses a rotary or transverse pattern with ~35% duty factor. These references generate per-timestep reward signals based on the agreement between the robot's actual foot contacts and the reference contact schedule at the current phase.

Crucially, the weight of the gait-phase auxiliary terms follows a curriculum schedule w_gait(t) = w_0 · max(0, 1 - t/T_relax), where T_relax is typically set to 60% of total training iterations. During early training (t < T_relax), the gait references guide the policy toward structured locomotion patterns, dramatically reducing exploration time. After T_relax, the gait references are fully removed, and the policy is free to optimize purely for energy efficiency and task performance. This often results in the policy discovering gait modifications that are more efficient than the initial references.

The observation space includes joint positions (12), joint velocities (12), base angular velocity (3), projected gravity (3), commanded velocity (3), and a clock signal encoding the current gait phase (2 sinusoidal signals). The clock signal's frequency is velocity-dependent, linking commanded speed to expected stride frequency. The action space outputs 12 target joint positions processed through PD controllers.

Training uses PPO with clipping parameter ε=0.2, GAE λ=0.95, and a learning rate of 3×10⁻⁴ with cosine annealing. Domain randomization includes friction (μ ∈ [0.3, 1.5]), restitution (0.0–0.5), added mass (0–4 kg), center-of-mass offset (±2 cm per axis), motor strength (±15%), PD gain randomization (±20%), communication delay (0–20 ms), and terrain height field noise (±3 cm). A total of 2048 parallel environments run in Isaac Gym on a single RTX 4090.

## Key Results & Numbers
- CoT at walking speed (0.5 m/s): 0.65, compared to biological reference of ~0.55 for similar-mass quadrupeds (18% gap)
- CoT at trotting speed (1.5 m/s): 0.42, compared to biological reference of ~0.35 (20% gap)
- CoT at galloping speed (3.0 m/s): 0.38, compared to biological reference of ~0.30 (27% gap)
- Autonomous gait transitions occur at approximately 0.7 m/s (walk→trot) and 2.2 m/s (trot→gallop)
- Robustness: maintains locomotion under 60 N lateral pushes with CoT increase < 15%
- Payload robustness: CoT increases by < 10% with 3 kg added payload (on ~12 kg robot)
- Training convergence in ~4 hours on RTX 4090 with 2048 environments, 50% faster than training without constraint relaxation curriculum
- Velocity tracking RMSE < 0.06 m/s across full speed range

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper provides a comprehensive blueprint for implementing multi-gait, energy-efficient locomotion on Mini Cheetah. The three-gait (walk/trot/gallop) framework maps directly to Mini Cheetah's operational speed range. The constraint relaxation curriculum—starting with gait references and gradually removing them—is an elegant approach for Mini Cheetah's curriculum learning pipeline, combining the structure of reference-based training with the flexibility of unconstrained optimization.

The CoT analysis framework, comparing learned policies against biological baselines, provides a principled evaluation methodology for Mini Cheetah. The domain randomization parameters (friction ranges, payload, motor strength, PD gains) are calibrated for similarly-sized quadrupeds and can serve as initial values for Mini Cheetah's sim-to-real transfer pipeline. The demonstrated 4-hour training time makes rapid iteration feasible.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
While the multi-gait framework is quadruped-specific, the CoT analysis methodology and constraint relaxation curriculum are transferable to Cassie. The idea of using gait-phase references as soft constraints during early training, then relaxing them, could improve Cassie's Primitives-level skill learning—providing initial structure for walking and running primitives while allowing the policy to optimize beyond the references.

The energy decomposition analysis (identifying which joints contribute most to CoT) is applicable to Cassie's bipedal configuration and could inform reward design for the Controller level.

## What to Borrow / Implement
- Implement the gait-phase auxiliary reward terms with constraint relaxation curriculum for Mini Cheetah's walk/trot/gallop training
- Use the velocity-dependent clock signal encoding as an observation feature for Mini Cheetah's policy network
- Adopt the CoT analysis framework with biological reference comparisons for evaluating Mini Cheetah's energy efficiency
- Apply the domain randomization parameter ranges as starting values for Mini Cheetah's sim-to-real configuration
- Borrow the training convergence acceleration technique (gait references as soft constraints) for faster Mini Cheetah policy iteration

## Limitations & Open Questions
- Biological CoT baselines are still 18–27% more efficient than the learned policies, suggesting room for improvement in reward design or policy architecture
- Gallop gait stability is noted as challenging, with occasional stumbles during rapid speed changes; gallop-specific stability constraints may be needed
- The velocity-dependent clock signal assumes a fixed stride-frequency-to-speed relationship; this may not hold on rough terrain where the robot needs to adapt stride length and frequency independently
- The work is simulation-only; real-world deployment and sim-to-real transfer are not demonstrated
