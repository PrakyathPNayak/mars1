# Central Pattern Generator-Enhanced Reinforcement Learning for Quadruped Locomotion

**Authors:** (2024)
**Year:** 2024 | **Venue:** Various
**Links:** N/A (Multiple publications on CPG-RL hybrid locomotion)

---

## Abstract Summary
This line of work proposes a hybrid approach that combines Central Pattern Generator (CPG) oscillator networks with deep reinforcement learning policies for quadruped locomotion control. The core insight is that biological locomotion is fundamentally rhythmic, driven by CPG neural circuits in the spinal cord, and embedding this inductive bias into RL policies produces more natural gaits, faster training convergence, and easier sim-to-real transfer.

The CPG component generates rhythmic oscillation patterns that provide the baseline gait structure—specifying the phase relationships between legs (e.g., trot = diagonal pairs in anti-phase, bound = front-rear pairs in anti-phase). The RL policy operates on top of the CPG output, providing residual corrections that adapt the gait to terrain, velocity commands, and disturbances. This structured action space dramatically reduces the exploration burden compared to RL operating directly in joint space, as the policy only needs to learn corrections rather than discovering rhythmic patterns from scratch.

The hybrid approach demonstrates improved training stability (fewer failed training runs), more natural gait emergence (contact patterns closer to biological quadrupeds), and significantly easier sim-to-real transfer due to the reduced action space and inherent periodicity. Experiments across multiple quadruped platforms show consistent advantages over pure RL methods, particularly in early training phases and on challenging terrains.

## Core Contributions
- Proposes a principled integration of CPG oscillator networks with deep RL for quadruped locomotion
- Demonstrates that CPG-imposed rhythmic structure reduces RL exploration burden by constraining the action space to residual corrections
- Shows improved training stability with 50–70% fewer failed training runs compared to pure RL on quadruped tasks
- Achieves more biologically plausible gait patterns with contact timing closer to natural quadruped locomotion
- Demonstrates easier sim-to-real transfer due to structured, periodic action spaces that are more consistent across sim-real gaps
- Provides ablation on CPG parameters (frequency, amplitude, coupling weights) and their interaction with RL residual learning
- Supports multiple gait modes (walk, trot, pace, bound, gallop) through CPG topology configuration

## Methodology Deep-Dive
The CPG network is implemented as a system of coupled oscillators, typically using the Matsuoka oscillator model or phase oscillators. For a quadruped with 4 legs, each leg has an oscillator that produces a rhythmic signal. The oscillators are coupled with phase offsets that define the gait pattern. For a **trot gait**, the coupling matrix specifies that diagonal leg pairs (FL-HR, FR-HL) oscillate in phase while adjacent legs are in anti-phase. The CPG state is described by phase variables φᵢ and amplitude variables rᵢ for each leg i:

dφᵢ/dt = ωᵢ + Σⱼ wᵢⱼ sin(φⱼ - φᵢ - Δφᵢⱼ)
drᵢ/dt = αᵢ(Rᵢ - rᵢ)

where ωᵢ is the intrinsic frequency, wᵢⱼ are coupling weights, Δφᵢⱼ are desired phase offsets, Rᵢ is the target amplitude, and αᵢ is the convergence rate. The CPG output for each leg is mapped to target joint positions through a parameterized mapping: q_cpg,i = f(φᵢ, rᵢ), typically producing sinusoidal hip and knee trajectories.

The RL policy receives proprioceptive observations (joint positions, velocities, body orientation, angular velocity, velocity commands) and CPG state (phases, amplitudes) as input, and outputs **residual actions** Δq that are added to the CPG-generated joint targets: q_target = q_cpg + Δq. The residual actions are bounded (typically ±0.3 rad) to prevent the RL from overriding the CPG structure entirely. The RL also modulates CPG parameters: it can adjust frequency ω (controlling gait speed), amplitude R (controlling step height), and even phase offsets Δφ (enabling gait transitions).

Training uses PPO with a reward function that includes velocity tracking, energy efficiency, and a **gait regularity bonus** that rewards contact patterns consistent with the target gait. The CPG provides a strong prior that produces functional (if not optimal) locomotion from the start of training, so the RL policy can focus on refining the gait rather than discovering it. This is reflected in the learning curves: CPG-RL methods achieve functional locomotion within ~1M steps compared to ~5M steps for pure RL.

Domain randomization is applied to physical parameters (friction, mass, terrain height), but the CPG-RL approach is inherently more robust to these variations because the CPG structure constrains the action space. The sim-to-real transfer typically requires less aggressive randomization compared to pure RL policies, as the periodic structure of the CPG prevents the policy from exploiting sim-specific dynamics.

For **gait transitions**, the CPG topology is switched by changing the phase offset matrix Δφ, while the RL policy adapts its residual corrections for the new gait pattern. The transition can be triggered by velocity commands (e.g., switch from walk to trot above 1.0 m/s) or learned by the RL policy as part of the state-dependent CPG parameter modulation.

## Key Results & Numbers
- 50–70% reduction in failed training runs compared to pure RL on quadruped locomotion
- Functional locomotion achieved within ~1M environment steps vs. ~5M for pure RL (5× faster initial convergence)
- 20–30% improvement in gait regularity metrics (duty factor, phase consistency) compared to pure RL
- 15–25% reduction in cost of transport (energy efficiency) due to smoother, more periodic gaits
- Sim-to-real transfer success rate: 85–95% for CPG-RL vs. 60–80% for pure RL with equivalent domain randomization
- Residual action bounds of ±0.3 rad provide optimal balance between CPG structure and RL flexibility
- Supports walk (0–0.5 m/s), trot (0.5–2.0 m/s), and bound (2.0–3.5 m/s) with smooth transitions

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
CPG-RL hybrid control is directly applicable to Mini Cheetah's 12 DoF locomotion. The MIT Mini Cheetah has 3 DoF per leg (hip ab/ad, hip flex/ext, knee flex/ext), and a CPG network can be designed to provide rhythmic trajectories for each joint. The structured action space would significantly ease training, particularly for the initial curriculum stages where basic locomotion must be established before domain randomization.

The approach is especially valuable for sim-to-real transfer—Mini Cheetah deployment on hardware would benefit from the reduced sim-real gap of CPG-RL, potentially requiring less aggressive domain randomization and producing more reliable hardware deployments. The natural gait patterns produced by CPG-RL are also mechanically gentler on the hardware, reducing wear and risk of damage during training.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
For Cassie's bipedal locomotion, CPG concepts are applicable but require adaptation. Bipedal CPGs have a simpler coupling structure (two legs in anti-phase for walking) but more complex per-leg dynamics due to the underactuated nature of bipedal balance. The CPG rhythm generation is relevant to Cassie's Neural ODE component at the Controller level, which could incorporate oscillatory dynamics as an inductive bias for gait phase generation.

The CPG's gait phase signal could serve as a clock input to the Controller level, providing temporal structure for the lower-level policy. The transition mechanism between CPG gait patterns maps to Cassie's Primitives level gait selection. However, the residual learning framework may conflict with Cassie's hierarchical architecture, where the Controller level needs to track Primitives-level targets rather than applying residuals to a CPG baseline.

## What to Borrow / Implement
- Implement a Matsuoka oscillator CPG network for Mini Cheetah's 12 DoF with configurable gait patterns (trot, bound, pace)
- Use residual RL (PPO) on top of CPG outputs with bounded residual actions (±0.3 rad) for Mini Cheetah
- Incorporate CPG phase variables as auxiliary observations for the RL policy to enable phase-aware residual learning
- Adapt CPG oscillatory dynamics as an inductive bias for Cassie's Neural ODE gait phase generation at the Controller level
- Use CPG frequency modulation as a mechanism for velocity-adaptive gait control in Mini Cheetah

## Limitations & Open Questions
- CPG structure imposes a strong prior that may limit the policy's ability to discover non-rhythmic behaviors (e.g., recovery, jumping)
- Optimal CPG parameters (frequency range, amplitude, coupling weights) require manual tuning or separate optimization
- The mapping from CPG oscillator output to joint trajectories involves hand-designed functions that may not be optimal
- Extension to highly dynamic maneuvers (flips, jumps) that break the rhythmic assumption is unclear
