# Adaptive Energy Regularization for Autonomous Gait Transition and Efficient Locomotion

**Authors:** Various (UC Berkeley)
**Year:** 2024 | **Venue:** arXiv 2024
**Links:** https://arxiv.org/abs/2403.20001

---

## Abstract Summary
Introduces energy-centric reward terms in RL that enable robots to autonomously select the most efficient gait for given speeds. The distance-averaged energy consumption reward induces natural gait transitions (walking↔trotting) and measurably reduces cost of transport on real quadruped robots. Unlike manually scheduled gait transitions, this approach allows the policy to discover optimal transition points through energy minimization alone.

## Core Contributions
- Introduces the distance-averaged energy consumption (DAEC) reward that naturally induces gait transitions
- Demonstrates autonomous gait emergence without gait references, contact schedules, or manual transition rules
- Shows that energy regularization alone is sufficient to produce walking-to-trotting transitions at biologically appropriate speeds
- Achieves measurable reduction in cost of transport (CoT) on real quadruped hardware
- Provides theoretical analysis connecting energy-optimal locomotion to natural gait transition phenomena observed in animals
- Validates that energy-optimal policies are also more robust than energy-unaware policies
- Introduces adaptive scaling of the energy reward to prevent efficiency-performance trade-off collapse

## Methodology Deep-Dive
The core innovation is the distance-averaged energy consumption (DAEC) reward, defined as the ratio of total mechanical energy consumed to distance traveled: DAEC = ΣE / d, where ΣE is the cumulative mechanical energy (|τ · ω| integrated over time for all joints) and d is the forward distance covered. This metric directly corresponds to the cost of transport (CoT), a standard measure of locomotion efficiency in biomechanics. By penalizing high DAEC, the RL policy is incentivized to find the most energy-efficient locomotion strategy for any given velocity command.

A critical insight from the paper is that different gaits are energy-optimal at different speeds, mirroring observations in animal locomotion. Walking is most efficient at low speeds due to inverted-pendulum dynamics, while trotting becomes more efficient at higher speeds due to spring-mass dynamics. By using DAEC as a reward term, the policy naturally discovers this relationship and transitions between gaits at the energetically optimal speed. No gait references, contact schedules, or phase clocks are required—the gait pattern emerges purely from energy optimization.

The adaptive scaling mechanism addresses a key challenge: if the energy penalty is too strong, the policy converges to standing still (zero energy); if too weak, it has no effect on gait selection. The paper introduces an adaptive scaling factor that normalizes the energy penalty by the current velocity magnitude, ensuring that the energy incentive is proportional to the locomotion task difficulty. Additionally, a minimum velocity constraint prevents the degenerate standing-still solution. The scaling factor is automatically adjusted during training based on the running average of velocity tracking performance.

The training pipeline uses PPO with a composite reward: primary velocity tracking (high weight), DAEC penalty (adaptively scaled), joint acceleration smoothness penalty, and body orientation stability reward. The policy architecture is a standard MLP with proprioceptive inputs (joint positions, velocities, body orientation, angular velocity, velocity command). No gait-specific modules or conditioning inputs are used—the policy is a single unified network that implicitly learns gait selection.

Hardware validation is performed on a real quadruped robot with onboard power measurement. The robot is commanded to traverse a velocity sweep from 0 to 2 m/s, and energy consumption is recorded. The energy-regularized policy is compared against baselines: a velocity-only policy (no energy term), a hand-tuned gait-switching policy, and a policy with a fixed energy penalty (no adaptive scaling).

## Key Results & Numbers
- Autonomous walking-to-trotting transition emerges at ~0.8-1.0 m/s, matching biological predictions
- 18-22% reduction in CoT compared to velocity-only policies across the full speed range
- DAEC-regularized policy matches or beats hand-tuned gait-switching policy efficiency
- Adaptive scaling outperforms fixed energy penalty by 12% in CoT reduction
- No gait references, phase clocks, or contact schedules required
- Policy robustness (push recovery) improved by 15% compared to velocity-only baseline
- Real-world CoT measurements match simulation predictions within 10%
- Training converges in ~3 hours on a single GPU with 4096 parallel environments

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The DAEC reward is directly implementable in Mini Cheetah's PPO training pipeline with minimal modification. Since Mini Cheetah already uses joint torque and velocity measurements, computing DAEC is straightforward. The autonomous gait emergence eliminates the need for manually designing gait schedules or phase clocks for Mini Cheetah, simplifying the reward function while potentially improving performance. The adaptive scaling mechanism is particularly valuable for Mini Cheetah's wide operating speed range (0-3 m/s), where different gaits are optimal at different speeds. The finding that energy-optimal policies are also more robust aligns with the project's dual goals of efficiency and robustness. The 500 Hz PD control loop provides sufficient resolution for accurate energy computation.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
While Cassie is bipedal and the walking-trotting transition is quadruped-specific, the energy regularization principle applies to Cassie's walking-running transition. The DAEC reward could be incorporated into the Controller level of Project B's hierarchy to optimize low-level energy efficiency. The adaptive scaling mechanism is relevant for the Primitives level, where different motion primitives may have different energy profiles. However, the simplicity of the approach (single unified policy) contrasts with Project B's hierarchical architecture, and direct application would need adaptation. The Neural ODE Gait Phase module could potentially replace the implicit gait discovery with a more structured energy-optimal phase generation.

## What to Borrow / Implement
- Implement the DAEC reward term in Mini Cheetah's PPO reward function: r_energy = -λ · |τ·ω| / max(v_forward, ε)
- Add adaptive scaling of the energy penalty based on velocity tracking performance
- Remove or reduce reliance on explicit gait phase clocks in favor of energy-driven gait emergence
- Use the velocity sweep evaluation protocol to measure real-world CoT on Mini Cheetah
- Apply energy regularization at the Controller level of Project B's hierarchy for Cassie
- Compare emerged gaits against biological gait transition data for validation

## Limitations & Open Questions
- Only walking-trotting transition is demonstrated; extension to bound/gallop requires additional validation
- The minimum velocity constraint is a somewhat ad-hoc fix for the standing-still degenerate solution
- Energy computation from torque × velocity may not capture all real-world energy costs (electronics, cooling fans)
- The approach assumes flat terrain; how energy-optimal gaits change on slopes and rough terrain is unexplored
- Adaptive scaling introduces additional hyperparameters (running average window, scaling bounds)
- Whether the emerged gaits are truly globally energy-optimal or only locally optimal is not proven
- Extension to bipedal locomotion (walking-running transition) is mentioned but not validated
