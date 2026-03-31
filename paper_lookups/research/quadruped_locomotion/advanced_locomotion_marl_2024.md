# Learning Advanced Locomotion for Quadrupedal Robots: A Multi-Agent RL Approach

**Authors:** (MDPI Robotics 2024)
**Year:** 2024 | **Venue:** MDPI Robotics
**Links:** [MDPI](https://www.mdpi.com/2218-6581/13/6/86)

---

## Abstract Summary
This paper formulates quadrupedal locomotion as a multi-agent reinforcement learning (MARL) problem where each leg operates as an independent agent. Rather than training a single centralized policy that maps the full robot state to all 12 joint actions simultaneously, the method assigns each leg a local agent with its own observation space, action space, and reward signal. Coordination emerges through shared global information and inter-agent communication channels rather than explicit centralized control.

The multi-agent decomposition dramatically simplifies the learning problem for extreme agile behaviors. Each leg agent only needs to learn 3-DoF control within its local workspace, with coordination handled by a lightweight communication protocol. This decomposition enables learning of previously intractable behaviors including backflips, rapid 360° body rotations, and aggressive recovery from extreme perturbations. The per-leg Riemannian motion policies ensure that actions respect the geometric constraints of the leg workspace, improving both learning efficiency and motion quality.

The approach also improves reward engineering: instead of designing complex reward functions for the whole-body behavior, the method uses simple per-leg rewards (foot clearance, ground reaction force targets, swing trajectory smoothness) that combine emergently to produce coordinated whole-body locomotion. The emergent coordination often discovers strategies that a human-designed reward would not encode.

## Core Contributions
- Formulation of quadrupedal locomotion as a cooperative multi-agent RL problem with each leg as an independent agent
- Riemannian motion policies that parameterize leg actions on the manifold of kinematically feasible joint configurations, ensuring smooth and physically valid motions
- Lightweight inter-agent communication mechanism enabling coordination without centralized control — each leg agent broadcasts a low-dimensional embedding to others
- Demonstrated learning of extreme agile behaviors (backflips, rapid rotation recovery) that are intractable for single-policy approaches
- Simplified reward engineering through per-leg decomposition — complex whole-body behaviors emerge from simple local leg rewards
- Improved learning efficiency: 3× faster convergence than centralized PPO for trotting, 5× faster for backflips
- Analysis of emergent inter-leg coordination patterns and their correspondence to natural animal gait patterns

## Methodology Deep-Dive
Each of the four leg agents i ∈ {FL, FR, RL, RR} receives a local observation o_i = [q_i, q̇_i, τ_i, f_i, p_com, v_com, ω_base, R_base, m_j≠i] where q_i, q̇_i, τ_i are the leg's own joint positions, velocities, and torques; f_i is the ground reaction force at the foot; p_com, v_com, ω_base, R_base are shared global body state; and m_j≠i are communication messages from the other three legs. Each agent outputs actions a_i ∈ R³ (desired joint positions for hip, thigh, calf) for its own leg.

The communication mechanism works as follows: each leg agent has an encoder network that maps its local observation to a low-dimensional message m_i ∈ R^8. These messages are broadcast to all other agents at each timestep, providing information about each leg's current state and intentions without requiring a centralized controller. The communication encoder is trained end-to-end alongside the policy through the MAPPO (Multi-Agent PPO) objective.

The Riemannian motion policy parameterization constrains actions to the manifold of kinematically feasible leg configurations. Rather than outputting raw joint angle targets, the policy outputs a velocity vector on the tangent space of the current configuration: a_i = Exp_{q_i}(v_i), where Exp is the Riemannian exponential map and v_i is the predicted tangent vector. This ensures that actions always produce smooth, kinematically valid motions and eliminates the need for clipping or projection. The manifold is defined by the leg's kinematic constraints (joint limits, singularity avoidance regions, self-collision boundaries) and is pre-computed as a discretized mesh.

Per-leg rewards are composed of: (1) r_swing = −||foot_z − h_clearance||² during swing phase, rewarding appropriate foot clearance height; (2) r_stance = −||GRF_i − GRF_target||² during stance phase, rewarding target ground reaction force distribution; (3) r_smooth = −||a_i,t − a_i,t−1||² penalizing jerky actions; (4) r_energy = −||τ_i · q̇_i|| penalizing energy consumption. The global coordination reward is added as: r_global = r_velocity_tracking + r_orientation + r_stability, where r_velocity_tracking rewards following a commanded base velocity, r_orientation penalizes roll/pitch deviations, and r_stability rewards maintaining the center of pressure within the support polygon.

For extreme agile behaviors, the method adds behavior-specific reward components. The backflip reward includes: r_flip = w_rot · angular_progress + w_height · max_com_height − w_impact · landing_force, where angular_progress measures the accumulated pitch rotation, max_com_height rewards achieving sufficient height for the rotation, and landing_force penalizes hard impacts. The training curriculum starts from standing, progresses through jumping, and finally adds rotation incentives.

Training uses the MAPPO algorithm with shared parameters across all four leg agents (parameter sharing) but agent-specific input encoding layers to handle morphological differences (front vs. rear legs have different kinematic properties). The centralized critic receives all four agents' observations for variance reduction during training, while execution is fully decentralized.

## Key Results & Numbers
- Backflip success rate: 78% (vs. 12% for centralized PPO with same training budget)
- Trotting velocity tracking error: 0.05 m/s RMSE (comparable to centralized baseline at 0.06 m/s)
- Training convergence: 3× faster for standard gaits, 5× faster for agile maneuvers vs. centralized PPO
- Recovery from 30° lateral push: 0.8s average recovery time (vs. 1.2s for centralized policy)
- Communication message dimensionality: 8 floats per leg, total inter-agent bandwidth of 96 floats/step
- Emergent gait patterns included walk, trot, pace, bound, and gallop — discovered through MARL coordination without explicit gait specification
- Energy efficiency: 15% improvement over centralized policy for trotting due to better inter-leg coordination
- Parameter count: 4× smaller per-agent network (64K params) vs. centralized network (512K params) for equivalent performance

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The per-leg MARL decomposition is directly applicable to Mini Cheetah's 12-DoF system (3 joints per leg). Mini Cheetah's high-bandwidth actuators and dynamic capabilities make it an ideal platform for the extreme agile behaviors this method enables (backflips, rapid recovery). The simplified per-leg reward design would reduce reward engineering effort for new Mini Cheetah skills. The Riemannian motion policies are particularly valuable for Mini Cheetah's backdrivable actuators, ensuring smooth joint trajectories. The emergent gait discovery (walk, trot, bound, gallop appearing without explicit specification) complements the DIAYN-based skill discovery approach — MARL coordination can serve as the low-level execution mechanism for discovered skills.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The multi-agent per-limb decomposition could be adapted for Cassie's bipedal locomotion, treating each leg as an independent agent. With only 2 agents (vs. 4 for quadrupeds), the coordination problem is simpler but the per-leg autonomy is even more critical — each Cassie leg must handle larger fractions of body weight and more complex swing dynamics. The communication mechanism could enable better left-right leg coordination for walking. The Riemannian motion policy concept applies to Cassie's constrained leg workspace (closed kinematic chains). However, Cassie's underactuation (passive ankle springs) and higher-dimensional leg kinematics (5 DoF per leg) present different challenges than the quadruped setting.

## What to Borrow / Implement
- Implement MAPPO with 4 leg agents for Mini Cheetah, with shared parameters and agent-specific input encoding for front/rear leg differences
- Design per-leg reward components (swing clearance, stance GRF, smoothness, energy) that combine to produce coordinated locomotion
- Use the 8-dimensional communication message protocol for inter-leg coordination without centralized control
- Explore Riemannian motion policy parameterization for Mini Cheetah's 3-DoF legs to ensure kinematically valid and smooth joint trajectories
- Leverage the MARL framework as the execution layer beneath DIAYN-discovered skill primitives

## Limitations & Open Questions
- Backflip success rate of 78% indicates significant failure rate for the most extreme behaviors — real-hardware deployment of failure-prone agile skills requires safety mechanisms
- The Riemannian manifold discretization and exponential map computation add computational overhead that may impact real-time control on embedded hardware
- Communication delay between agents is assumed zero (same timestep broadcast); real distributed systems would introduce latency that could degrade coordination
- Parameter sharing across legs assumes morphological similarity; robots with significantly different front and rear leg designs may require independent training
