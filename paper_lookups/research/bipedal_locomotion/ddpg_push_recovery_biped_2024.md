# Push Recovery Control for a Biped Robot Using DDPG Reinforcement Learning Algorithm

**Authors:** Noorani et al.
**Year:** 2024 | **Venue:** ResearchGate
**Links:** ResearchGate (2024)

---

## Abstract Summary
This paper applies the Deep Deterministic Policy Gradient (DDPG) algorithm to the problem of bipedal robot push recovery. Unlike model-based approaches that rely on simplified dynamics models (LIPM, ZMP), this work uses model-free deep reinforcement learning to learn reactive stepping policies directly from interaction experience. The DDPG agent learns to map the robot's state (body pose, velocity, estimated disturbance) to recovery actions (step location, hip/knee torques) through trial-and-error in simulation, without requiring an explicit dynamics model.

The key insight is that DDPG's continuous action space is well-suited for push recovery, where the recovery step placement and joint torques are naturally continuous variables. Unlike DQN or discrete RL methods that would require discretizing the action space (losing precision in foot placement), DDPG outputs exact step locations and torque values. The actor-critic architecture enables stable learning in the high-dimensional, continuous state-action space of bipedal push recovery.

The system is validated in simulation on a planar biped model experiencing external pushes of varying magnitude and direction. The learned policy successfully recovers balance in standing scenarios (resisting pushes without walking) and walking scenarios (maintaining gait stability under perturbation). The DDPG policy shows online adaptation to novel disturbance patterns not seen during training, suggesting generalization capability.

## Core Contributions
- **DDPG application to bipedal push recovery** — first systematic study of off-policy continuous RL for bipedal balance recovery, demonstrating feasibility and advantages over discrete RL methods
- **Continuous action space for step placement** — DDPG's continuous output enables precise foot placement optimization without discretization artifacts
- **Dual-mode push recovery** — separate standing recovery (in-place balance) and walking recovery (gait perturbation rejection) policies trained within a unified DDPG framework
- **Disturbance estimation integration** — the policy observation includes an online disturbance estimate (from a momentum observer), enabling anticipatory recovery actions
- **Comparison with model-based baselines** — systematic comparison with CP-based and ZMP-based recovery controllers showing competitive or superior performance for large disturbances
- **Generalization analysis** — evaluation on untrained push magnitudes and directions demonstrating the policy's interpolation and mild extrapolation capabilities

## Methodology Deep-Dive
The DDPG architecture consists of an actor network and a critic network, both implemented as feedforward neural networks. The actor network (4 layers: 256→256→128→action_dim) maps the observation to continuous actions. The critic network (4 layers, with observation and action concatenated at the second layer) estimates the Q-value for state-action pairs. Both networks use ReLU activations with batch normalization after each hidden layer. Target networks with soft update (τ = 0.001) provide stable training targets.

The observation space includes: (1) CoM position and velocity in the body frame (6D), (2) body orientation (roll, pitch, yaw) and angular velocity (6D), (3) joint positions and velocities for both legs (12D for a 6-DoF planar biped), (4) foot contact states (2D binary), (5) estimated external disturbance force and torque (6D from the momentum observer), and (6) gait phase (1D sinusoidal encoding). The total observation dimension is 33.

The action space includes: (1) desired foot placement position relative to the hip (2D: forward/backward and lateral offset), (2) desired step timing (1D: when to initiate the recovery step), and (3) hip and knee torque adjustments (4D: modifying the nominal gait torques during recovery). The total action dimension is 7. Actions are bounded using tanh activation in the actor's output layer, scaled to physically feasible ranges.

The reward function is carefully designed for push recovery: R = w₁ · r_survival + w₂ · r_upright + w₃ · r_velocity + w₄ · r_energy + w₅ · r_smooth. The survival reward (r_survival = 1.0 per timestep without falling) dominates to encourage balance maintenance. The upright reward penalizes body tilt from vertical. The velocity reward encourages returning to the commanded walking speed after recovery. The energy reward penalizes excessive joint torques. The smoothness reward penalizes large action changes between timesteps.

Training proceeds in a curriculum over disturbance magnitude. Initially, small pushes (10-30N) are applied at random times during standing and walking. As the policy improves (measured by average episode survival time), the push magnitude is progressively increased up to 200N. The push direction is randomized uniformly over the horizontal plane. An experience replay buffer of 1M transitions stores training data, with prioritized experience replay (PER) emphasizing transitions where the robot narrowly avoided falling (high TD-error transitions near the stability boundary).

The disturbance estimation module uses a generalized momentum observer that estimates external forces from the discrepancy between expected and actual joint torques/accelerations. The observer runs at 1kHz (control rate) with a low-pass filter at 10Hz to remove noise. The estimated disturbance is fed into the policy with a 50ms delay to account for the observer's settling time. This anticipatory information allows the policy to initiate recovery actions slightly before the full disturbance effect is felt.

## Key Results & Numbers
- **Standing push recovery**: withstands pushes up to **180N for 0.2s** (36 N·s impulse) from all directions, compared to 120N for ZMP-based controller
- **Walking push recovery**: maintains gait stability under **120N for 0.2s** (24 N·s impulse) pushes, comparable to CP-MPC baseline
- **Recovery success rate**: **95%** for trained push magnitudes, **82%** for untrained magnitudes up to 20% beyond training range (interpolation/extrapolation)
- **Step placement precision**: mean error of **2.3cm** from optimal step location (computed by offline optimization), compared to 3.8cm for the CP-based heuristic
- **Training efficiency**: converges in **2M environment steps** (~15 hours on a single GPU with simulation parallelism)
- **Online adaptation**: policy adjusts recovery strategy within **3-5 pushes** of a new disturbance pattern (direction/magnitude) not seen during training
- **Comparison**: DDPG outperforms DQN (discretized action) by **25%** in average recoverable push magnitude, validating continuous action space advantage
- **Computation**: policy inference in **<1ms**, compatible with 1kHz control loops on standard hardware

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
While this paper focuses on bipedal push recovery, the DDPG-based approach to learning reactive recovery policies has moderate relevance for Mini Cheetah perturbation handling. The continuous action space for step placement and torque adjustment translates to quadruped foot placement during recovery from pushes or terrain perturbations. The disturbance estimation integration (momentum observer) could enhance Mini Cheetah's awareness of external forces for more proactive stabilization.

The curriculum over disturbance magnitude is directly applicable to Mini Cheetah's domain randomization and curriculum learning approach for robustness training.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
DDPG-based push recovery is directly applicable to Cassie's Safety level in the 4-level hierarchy. The learned reactive stepping policy could serve as Cassie's push recovery primitive, triggered by the LCBF when balance is threatened. The continuous action space for foot placement maps naturally to Cassie's Controller level, where precise step location is critical for bipedal balance.

The disturbance estimation module (momentum observer) is relevant to Cassie's state estimation pipeline, providing external force information to the Safety level. The dual-mode design (standing vs. walking recovery) aligns with Cassie's need for context-dependent recovery strategies — different recovery behaviors when standing still vs. walking vs. running. The demonstrated DDPG advantages over discrete RL validate using continuous action-space algorithms in Cassie's architecture. However, the project's use of PPO (on-policy) may benefit from comparing with off-policy DDPG specifically for the push recovery subtask where sample efficiency matters.

## What to Borrow / Implement
- **DDPG or SAC for push recovery sub-policy** — consider training Cassie's push recovery primitive with off-policy continuous RL (DDPG/SAC) for sample-efficient recovery learning, potentially alongside PPO for the main locomotion policy
- **Momentum-based disturbance observer** — implement external force estimation from joint torque discrepancy for Cassie's Safety level, providing anticipatory balance information
- **Push magnitude curriculum** — design progressive disturbance training for Cassie's push recovery, starting with small perturbations and increasing to near-fall scenarios
- **Continuous foot placement action** — output precise step location rather than discretized options for Cassie's Controller level recovery stepping
- **Prioritized experience replay for recovery** — emphasize near-fall transitions in the replay buffer to improve learning at the stability boundary

## Limitations & Open Questions
- **Planar biped only** — all experiments use a 2D planar biped model; 3D bipedal dynamics with lateral stability are not addressed, which is essential for Cassie
- **Simulation-only results** — no sim-to-real transfer demonstrated; the gap between simulated and real push recovery dynamics could be significant
- **Fixed disturbance model** — pushes are modeled as instantaneous force pulses; continuous or time-varying disturbances (pushing against a moving object, wind) may require different recovery strategies
- **No formal safety analysis** — the DDPG policy has no guaranteed recovery bounds; combining with barrier functions or capturability analysis could provide safety certificates
