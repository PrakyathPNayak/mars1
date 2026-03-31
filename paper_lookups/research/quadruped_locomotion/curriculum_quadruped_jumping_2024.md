# Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-Free Design

**Authors:** Vassil Atanassov et al. (Oxford / TU Delft)
**Year:** 2024 | **Venue:** IEEE RA-L / arXiv
**Links:** [arXiv](https://arxiv.org/abs/2401.16337)

---

## Abstract Summary
This paper presents a reference-free curriculum learning approach for training quadruped robots to perform dynamic jumping without motion capture data, trajectory optimization, or any form of reference motion. The robot learns entirely from scratch using reinforcement learning with a carefully designed curriculum that progressively increases the difficulty of jumping tasks. Starting from simple standing and hopping motions, the curriculum escalates through increasing jump distances, heights, and precision requirements.

The approach achieves a record 90cm jumping distance on real quadruped hardware, surpassing previous RL-based and optimization-based methods. A key contribution is the robust landing recovery policy that enables the robot to reliably stabilize after impact, even from imperfect jumps. The method is demonstrated on real hardware deployed on grass terrain, showing robustness to uneven surfaces and unpredictable ground conditions.

The paper argues that reference-free design is preferable for extreme agile skills like jumping because reference motions from motion capture or trajectory optimization may not represent dynamically optimal strategies. By allowing the RL agent to discover its own jumping strategy through the curriculum, the system finds solutions that fully exploit the robot's dynamic capabilities rather than imitating sub-optimal references.

## Core Contributions
- Reference-free curriculum learning for quadrupedal jumping that discovers optimal jumping strategies without any motion priors or trajectory references
- Progressive difficulty curriculum spanning standing → hopping → short jumps → long jumps → precision jumps, with automatic difficulty adjustment based on success rate
- Record 90cm jumping distance on real quadruped hardware using only reinforcement learning
- Robust landing recovery policy that stabilizes the robot after impact, trained jointly with the jumping policy using a phase-based reward structure
- Real-world deployment on grass terrain demonstrating robustness to surface irregularities and unmodeled ground compliance
- Systematic analysis of curriculum design choices (progression speed, difficulty metrics, reward shaping) and their impact on final performance
- Ablation studies showing that reference-free design outperforms reference-tracking approaches for dynamic jumping

## Methodology Deep-Dive
The curriculum is structured as a sequence of increasingly difficult jumping tasks, parameterized by target distance d_target and target height h_target. The curriculum manager tracks the agent's success rate σ (fraction of jumps within tolerance of the target) over a rolling window of 100 episodes. When σ exceeds a promotion threshold τ_up = 0.8, the curriculum advances to the next difficulty level; when σ falls below a demotion threshold τ_down = 0.3, the curriculum regresses. This automatic progression ensures the agent always trains near its current skill frontier.

The reward function is decomposed into phase-specific components. The jump is divided into three phases: (1) Preparation phase — the robot positions itself for launch, rewarded for maintaining a stable squat posture with appropriate CoM height and leg extension angles; (2) Flight phase — initiated by detecting all feet leaving the ground, rewarded for tracking the target ballistic trajectory with bonuses for body orientation (level torso) and angular velocity minimization; (3) Landing phase — triggered by first ground contact after flight, rewarded for rapid stabilization measured by base velocity convergence to zero and orientation convergence to upright within T_recovery = 1.0 seconds.

The preparation phase reward is: r_prep = w₁·r_posture + w₂·r_alignment, where r_posture rewards crouching to an optimal launch angle and r_alignment rewards facing the target direction. The flight phase reward is: r_flight = w₃·exp(−||p_com − p_target||²/σ²) + w₄·exp(−||ω_base||²), combining CoM trajectory tracking with angular velocity minimization. The landing reward is: r_land = w₅·(1 − ||v_base||/v_max)·(1 − ||θ_base − θ_upright||/θ_max) + w₆·I(stable), where I(stable) is an indicator that the robot achieves zero-velocity upright stance within T_recovery.

The policy architecture uses a two-headed MLP with shared feature layers [512, 256] and separate action heads for the jumping and recovery behaviors. The jumping head outputs desired joint positions for all 12 joints during preparation and flight phases, while the recovery head outputs PD gain modulations and reference positions for landing stabilization. Phase detection is handled by a learned phase classifier that estimates the current jump phase from proprioceptive history. The entire system is trained end-to-end with PPO in Isaac Gym.

Domain randomization is extensive: mass (±20%), friction (0.2–1.5), motor strength (±15%), latency (0–20ms), ground compliance (0.5–2.0 relative stiffness), and terrain slope (±5°). The grass deployment specifically benefits from the ground compliance randomization, which simulates the energy absorption and uneven support of natural terrain.

## Key Results & Numbers
- 90cm forward jumping distance — a record for RL-trained quadrupedal jumping without references
- 35cm vertical jump height achieved on flat ground
- Landing recovery success rate: 94% from well-executed jumps, 78% from partially failed jumps
- Curriculum training converges in ~8 hours on a single GPU with Isaac Gym (4096 parallel environments)
- Real-world deployment on grass terrain: 85cm average jump distance (94% of simulation performance)
- Reference-free approach outperformed reference-tracking baseline by 15% in maximum jump distance
- Curriculum progression through 8 difficulty levels with automatic advancement
- Sim-to-real gap: <7% performance degradation on flat ground, <12% on grass

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Jumping is a critical agile skill for Mini Cheetah, and this reference-free curriculum approach is directly applicable. Mini Cheetah's lightweight design (9kg) and powerful actuators make it an ideal platform for dynamic jumping. The curriculum structure can be adopted directly: standing → hopping → short jumps → long jumps, with automatic progression based on success rate. The phase-decomposed reward (preparation, flight, landing) provides a clean reward engineering template. The landing recovery policy is especially valuable — Mini Cheetah needs robust recovery from all agile maneuvers, not just jumps. The grass deployment results demonstrate real-world feasibility with unmodeled terrain effects.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Low**
Jumping is less relevant to Cassie's primary locomotion objectives (walking, running, terrain navigation). However, the curriculum learning methodology — progressive difficulty with automatic advancement/regression — could be applied to Cassie's training for other skills (walking speed curriculum, terrain difficulty curriculum). The phase-decomposed reward structure is also transferable to bipedal locomotion phases (stance, swing, double support). The reference-free philosophy aligns with unsupervised skill discovery approaches used in Project B.

## What to Borrow / Implement
- Adopt the automatic curriculum progression with success-rate-based promotion (τ_up=0.8) and demotion (τ_down=0.3) thresholds for Mini Cheetah jumping training
- Implement the three-phase reward decomposition (preparation, flight, landing) with phase detection from proprioceptive state
- Use the landing recovery policy architecture (separate recovery head with PD gain modulation) for post-jump stabilization
- Apply the extensive domain randomization protocol (especially ground compliance randomization) for grass/outdoor deployment
- Consider the reference-free curriculum approach for other agile Mini Cheetah skills (flipping, rapid turning) beyond jumping

## Limitations & Open Questions
- The curriculum structure (number of levels, progression thresholds) requires manual design and may not generalize across different robot platforms without re-tuning
- Landing recovery from extreme failure cases (e.g., upside-down landing) is not addressed — the 78% recovery rate from failed jumps leaves room for improvement
- The method focuses on forward jumping; lateral and diagonal jumps are not explored and may require separate curricula
- Energy efficiency of discovered jumping strategies is not analyzed — the RL agent may find solutions that are dynamically effective but energetically wasteful for battery-powered deployment
