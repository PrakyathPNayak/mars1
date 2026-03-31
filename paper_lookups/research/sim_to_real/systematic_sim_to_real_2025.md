# Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robot Locomotion

**Authors:** arXiv Authors (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2509.06342)

---

## Abstract Summary
This paper presents a comprehensive, end-to-end pipeline for sim-to-real transfer in legged robot locomotion, addressing the reality gap through a multi-faceted approach that combines physics-informed actuator modeling, systematic observation design, domain randomization, curriculum learning, and online system identification. The key innovation is the integration of Permanent Magnet Synchronous Motor (PMSM) energy models directly into the RL training loop, enabling the simulator to accurately capture actuator dynamics including thermal effects, voltage limits, current saturation, and back-EMF that significantly affect real-world performance.

The pipeline proceeds through six stages: (1) System identification of rigid body and actuator parameters from real-world data, (2) Observation design incorporating input/output history windows for implicit system identification, (3) Structured domain randomization with physics-informed parameter ranges, (4) Curriculum learning that progressively increases task difficulty and randomization variance, (5) Simulation grounding through iterative refinement of simulator parameters, and (6) Online system identification at deployment for real-time adaptation.

Validated on multiple legged robot platforms (quadrupeds and bipeds), the approach demonstrates consistent sim-to-real transfer across walking, trotting, bounding, and dynamic locomotion gaits. The PMSM actuator model alone accounts for a 15-25% improvement in transfer fidelity, highlighting the importance of actuator-level physics in bridging the reality gap.

## Core Contributions
- Integration of PMSM motor energy models into RL training, capturing thermal limits, current saturation, and voltage constraints that dominate real-world actuator behavior
- Six-stage systematic pipeline covering the full sim-to-real workflow from system identification to online adaptation
- Observation design methodology using I/O history windows (past 10-50 timesteps) enabling implicit system identification by the policy network
- Physics-informed domain randomization with parameter ranges derived from actuator specifications and manufacturing tolerances rather than arbitrary uniform ranges
- Curriculum learning schedule that co-evolves task difficulty and domain randomization variance
- Simulation grounding: iterative process of comparing sim vs. real trajectories and adjusting simulator parameters
- Demonstration across multiple legged robot morphologies (quadruped and biped) and gait types

## Methodology Deep-Dive
The PMSM actuator model is the centerpiece of this work. Standard RL simulators model joints with simple torque limits or PD controllers, ignoring critical motor physics. The authors model the full electrical-mechanical coupling: the voltage equation V = RI + L(dI/dt) + K_e·ω (where R is resistance, L is inductance, I is current, K_e is back-EMF constant, ω is motor velocity), the torque equation τ = K_t·I (torque constant times current), and thermal dynamics dT/dt = (I²R - h(T-T_amb))/C_th (Joule heating minus convective cooling, divided by thermal capacitance). These equations are integrated at the simulation timestep, and the motor's torque output is clamped based on real-time temperature-dependent current limits. This captures phenomena like torque reduction during sustained high-effort movements and velocity-dependent torque limits from back-EMF.

The observation design uses a history window of the past H timesteps of joint positions, velocities, torques, and commanded actions: o_t = [s_t, s_{t-1}, ..., s_{t-H}, a_{t-1}, ..., a_{t-H}]. With H=10-50 and a 50 Hz control loop, this provides 0.2-1.0 seconds of history. The policy network (typically an MLP or Transformer) can extract implicit estimates of physical parameters (mass, friction, motor health) from this history, enabling adaptive behavior without explicit system identification. An ablation study shows H=20 (0.4s) provides the best trade-off between information content and input dimensionality.

Domain randomization is structured in three tiers: Tier 1 (always randomized) includes friction coefficients, ground restitution, and observation noise with ranges from actuator datasheets. Tier 2 (moderately randomized) includes link masses (±10%), CoM positions (±2cm), and motor gains (±15%), with ranges from manufacturing tolerances. Tier 3 (lightly randomized) includes joint damping and armature inertia (±5%), reflecting well-controlled manufacturing. The curriculum progressively widens Tier 2 and 3 ranges over training, starting from nominal values and reaching full randomization by 60% of training.

Simulation grounding uses a Bayesian optimization loop: collect 5-10 real trajectories, run the same actions through the simulator with current best parameters, compute trajectory divergence (DTW distance on joint trajectories), and optimize simulator parameters to minimize divergence. This process iterates 3-5 times, with each iteration improving simulator fidelity. Online system identification at deployment uses a lightweight Extended Kalman Filter (EKF) estimating key parameters (ground friction, payload mass) from recent observations, feeding these estimates into the policy's observation vector.

The curriculum learning combines three progressions: (1) Command difficulty (speed: 0→max, turning rate: 0→max), (2) Terrain difficulty (flat→rough→stairs), and (3) Domain randomization variance (narrow→full). These are scheduled based on policy performance thresholds rather than fixed timestep schedules, ensuring stable progression.

## Key Results & Numbers
- PMSM actuator model alone improves sim-to-real velocity tracking by 15-25% compared to simple torque-limit models
- I/O history window (H=20) improves transfer fidelity by 12% over memoryless policies
- Full pipeline achieves 90%+ sim-to-real velocity tracking across walking, trotting, and bounding gaits
- Simulation grounding reduces trajectory divergence (DTW) by 40% over 3 iterations
- Online EKF adaptation improves performance by 8-10% on unseen terrains vs. fixed-parameter deployment
- Training time: ~12 hours on single GPU for quadruped, ~24 hours for biped (with curriculum)
- Policies generalize to 5+ terrains not seen during training after full pipeline application
- Energy efficiency within 10% of model-based controllers, while handling much more diverse conditions

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper provides a near-complete blueprint for Project A's sim-to-real pipeline. The MIT Mini Cheetah uses PMSM motors (specifically custom proprioceptive actuators), making the PMSM energy model directly applicable. Integrating thermal modeling and back-EMF constraints into the MuJoCo simulation would significantly improve transfer fidelity, especially for dynamic gaits where motors approach their limits.

The six-stage pipeline maps directly to Project A's workflow: system identification from Mini Cheetah hardware data, I/O history observation design (the paper's H=20 at 50Hz is compatible with Mini Cheetah's control frequency), physics-informed domain randomization using Mini Cheetah actuator datasheets, curriculum learning for progressive gait training, simulation grounding with real Mini Cheetah trajectories, and online EKF adaptation for deployment. The curriculum learning approach that co-evolves task difficulty with randomization variance is particularly valuable for stable training.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The systematic approach is highly applicable to Cassie's hierarchical RL framework. The PMSM actuator model is relevant since Cassie uses similar brushless DC motors with significant thermal and voltage constraints during dynamic locomotion. At the Controller level, accurate actuator models improve joint-tracking fidelity. At the Safety level (LCBF), physics-informed motor limits provide tighter and more accurate constraint sets for the CBF-QP formulation.

The I/O history observation design complements Cassie's Dual Asymmetric-Context Transformer architecture. The history window approach aligns with the Transformer's context window, and the paper's finding that H=20 is optimal provides a useful starting point for Cassie's context length. The curriculum learning schedule (command difficulty + terrain difficulty + DR variance) maps naturally to Project B's Adversarial Curriculum, where the adversary progressively challenges the policy. The online EKF system identification could augment the Neural ODE Gait Phase estimator with real-time physical parameter estimates.

## What to Borrow / Implement
- Implement PMSM motor energy model in MuJoCo for both Mini Cheetah and Cassie simulators, including thermal dynamics and voltage-dependent torque limits
- Adopt I/O history window (H=20) in observation design for both projects; integrate with Transformer architecture for Project B
- Use the three-tier domain randomization structure with physics-informed ranges from actuator datasheets
- Implement the co-evolving curriculum (task difficulty + terrain + DR variance) with performance-based progression thresholds
- Deploy lightweight EKF for online parameter estimation during real-world deployment on both platforms

## Limitations & Open Questions
- PMSM model adds computational overhead (~15% slower simulation); impact on training throughput at scale unclear
- System identification requires hardware access and careful data collection; methodology for teams without real hardware not addressed
- Online EKF assumes linear parameter dynamics; highly nonlinear effects (e.g., gear backlash, cable stretch) may not be well-captured
- Curriculum learning schedule requires manual design of progression thresholds; automatic curriculum approaches not explored
