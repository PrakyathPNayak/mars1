# Efficient Learning of Robust Multigait Quadruped Locomotion for Energy Optimization

**Authors:** Various
**Year:** 2025 | **Venue:** Frontiers of IT & EE (Springer)
**Links:** https://link.springer.com/article/10.1631/FITEE.2401070

---

## Abstract Summary
Proposes a training-synthesizing framework to integrate diverse gait-conditioned policies into a unified, energy-optimal locomotion policy. Enables seamless, low-CoT transitions between gaits while maintaining energy efficiency across speeds and commands. The approach decouples gait discovery from gait unification, producing a single deployable policy that autonomously selects the most efficient gait for each operating condition.

## Core Contributions
- Introduces a two-stage training-synthesizing framework: first train individual gait-conditioned policies, then distill into a unified multigait policy
- Achieves seamless gait transitions (walk↔trot↔bound↔gallop) without explicit transition controllers
- Demonstrates measurable reduction in cost of transport (CoT) compared to single-gait policies
- Unified policy autonomously selects the most energy-efficient gait based on velocity commands
- Validates on real quadruped hardware with energy consumption measurements
- Shows that gait diversity improves robustness through complementary stability properties
- Provides energy-aware reward formulation that balances performance and efficiency

## Methodology Deep-Dive
The framework operates in two stages. Stage 1 trains individual gait-conditioned policies using gait-specific reward functions. Each gait (walk, trot, bound, gallop) is trained as a separate PPO policy with a gait reference signal that encourages the characteristic foot contact pattern. For example, the trot reference enforces diagonal foot pairing, while the bound reference enforces front-back synchronization. Each policy is optimized for velocity tracking within its natural speed range while penalizing energy consumption (joint torque × joint velocity). This stage produces 4 expert policies, each proficient in its respective gait.

Stage 2 synthesizes the individual experts into a unified policy through a distillation process. The unified policy receives the same observations as the experts plus a continuous speed command. A learned gait selector module outputs mixing weights over the expert policies, and the unified policy is trained to match the expert outputs while additionally optimizing for minimal energy consumption during gait transitions. The gait selector is trained end-to-end, learning to activate the most energy-efficient expert for each speed range and to produce smooth interpolations during transitions.

The energy optimization is implemented through a composite reward that includes a primary velocity tracking term, a secondary energy penalty (mechanical power = τ · ω, summed across joints), and a transition smoothness term (penalizing abrupt changes in joint acceleration during gait switches). The relative weighting of these terms is automatically adjusted through a curriculum: early training emphasizes velocity tracking, while later stages increase the energy penalty weight to refine efficiency without sacrificing task performance.

A key technical contribution is the gait phase representation used for conditioning. Rather than discrete gait labels, the paper uses a continuous phase variable that encodes the foot contact pattern as a circular coordinate. This allows smooth interpolation between gaits in the phase space, enabling the unified policy to discover intermediate gaits that may be more energy-efficient than any of the four canonical gaits at certain speeds.

Hardware validation measures actual energy consumption using current and voltage sensors on the robot's motors. The paper compares the unified multigait policy against single-gait baselines and a naive policy that switches between gaits at predefined speed thresholds. Results show the unified policy achieves lower CoT across the full speed range and produces noticeably smoother transitions.

## Key Results & Numbers
- Unified policy reduces CoT by 15-25% compared to single-gait policies across the speed range
- Seamless gait transitions in < 0.5 seconds with no stability loss
- Walk-to-trot transition occurs naturally at ~1.0 m/s, trot-to-bound at ~2.0 m/s (speed-dependent)
- Energy consumption validated on real hardware matches simulation predictions within 8%
- Robustness to external pushes improved by 20% due to gait diversity (can switch to more stable gait)
- Training time: ~4 hours for Stage 1 (4 gaits in parallel), ~2 hours for Stage 2 distillation
- Unified policy adds < 5% inference overhead compared to single-gait policy

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Mini Cheetah is designed for multigait locomotion (walk, trot, bound, pronk) and energy efficiency is critical for extended autonomous operation. The training-synthesizing framework can be directly applied: train individual gait policies for Mini Cheetah's gait repertoire, then distill into a unified energy-optimal policy. The energy-aware reward formulation (τ · ω penalty) is straightforward to implement in the existing PPO training setup. The continuous gait phase representation is particularly relevant for Mini Cheetah's 12 DoF system, where it can encode the characteristic coordination patterns of each gait. The hardware-validated energy measurements provide a benchmark methodology for evaluating Mini Cheetah's efficiency.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
While Cassie is bipedal and has fewer gait options than a quadruped, the gait transition optimization is relevant to the Neural ODE Gait Phase module in Project B's hierarchy. The continuous phase representation for encoding gait patterns aligns conceptually with the Neural ODE's continuous-time gait generation. The energy optimization framework could be applied to Cassie's walking and running gaits, optimizing the transition speed where switching from walking to running becomes energy-optimal. The distillation approach (experts → unified policy) provides a template for how motion primitives at the Primitives level could be unified. However, the quadruped-specific gait diversity (4+ gaits) is less directly applicable to bipedal locomotion.

## What to Borrow / Implement
- Implement the two-stage training-synthesizing framework for Mini Cheetah multigait learning
- Use the energy-aware reward formulation (velocity tracking + τ·ω penalty + smoothness) in PPO training
- Adopt the continuous gait phase representation for encoding foot contact patterns
- Apply the distillation approach to unify separately trained motion primitives in Project B
- Use current/voltage measurement methodology for real-world energy efficiency validation
- Implement the automatic reward weight curriculum (tracking → efficiency progression)

## Limitations & Open Questions
- Four canonical gaits may not span the full space of energy-efficient locomotion patterns
- Distillation may lose some performance at the extremes of each gait's natural speed range
- The approach assumes a known set of target gaits; discovering novel energy-optimal gaits is not addressed
- Gait transitions are optimized for steady-state switching; rapid alternation under perturbation is less studied
- How to extend the framework to include terrain-dependent gait selection (not just speed-dependent)
- The energy model used in reward shaping may not capture all real-world energy losses (e.g., electronics, cooling)
