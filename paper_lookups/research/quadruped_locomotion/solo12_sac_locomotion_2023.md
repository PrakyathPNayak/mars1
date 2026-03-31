# Controlling the Solo12 Quadruped Robot with Deep Reinforcement Learning

**Authors:** (Nature Scientific Reports 2023)
**Year:** 2023 | **Venue:** Nature Scientific Reports
**Links:** https://www.nature.com/articles/s41598-023-38259-7

---

## Abstract Summary
This paper presents a comprehensive study of training a Soft Actor-Critic (SAC) agent to control the Solo12 open-source quadruped robot for locomotion tasks. Solo12 is a lightweight, 12-DoF quadruped with transparent hardware and software design, making it an ideal platform for reproducible RL locomotion research. The paper provides an end-to-end pipeline from simulation training to real-world deployment, with detailed analysis of the challenges encountered at each stage.

The authors train locomotion policies in the PyBullet simulator using SAC with carefully designed reward functions incorporating forward velocity tracking, energy minimization, foot clearance, and smoothness terms. The policies learn diverse gaits (trotting, bounding, pacing) from scratch without gait references. Critical attention is paid to the sim-to-real gap: the paper systematically evaluates the effect of sensor noise modeling, actuator dynamics, contact model fidelity, and domain randomization on transfer quality.

A key contribution is the side-by-side comparison of RL-trained controllers with traditional model-based controllers (MPC, whole-body control) on the same hardware. The RL policies achieve competitive or superior performance in terms of velocity tracking and robustness to perturbations, while requiring significantly less expert knowledge and tuning. The entire pipeline—simulation, training, deployment—is open-source, providing a reproducible baseline for the community.

## Core Contributions
- Provides a complete, open-source, reproducible pipeline for training quadruped locomotion policies via SAC and deploying on real hardware
- Systematically analyzes sim-to-real transfer challenges: sensor noise, actuator modeling, contact dynamics, and domain randomization
- Compares RL-trained policies with model-based controllers (MPC, WBC) on the same hardware platform, showing competitive performance
- Demonstrates that SAC (off-policy) can learn diverse quadruped gaits from scratch without gait references or demonstrations
- Provides detailed reward function design analysis, showing the contribution of each reward component to final locomotion quality
- Identifies key factors for successful sim-to-real transfer: motor model accuracy, sensor noise injection, and conservative action limits
- Releases all code, trained models, and hardware specifications for community use

## Methodology Deep-Dive
The Solo12 robot has 12 actuated degrees of freedom (3 per leg: hip abduction/adduction, hip flexion/extension, knee flexion/extension). The observation space includes: joint positions (12), joint velocities (12), base orientation (quaternion, 4), base angular velocity (3), previous action (12), and command inputs (target linear velocity x/y, angular velocity yaw). Total observation dimension: 43. The action space is 12 desired joint positions, sent to PD controllers running at 1 kHz on the motor controllers. The RL policy runs at 50 Hz, generating a new action every 20ms.

SAC is chosen over PPO for several reasons: (1) sample efficiency—SAC's off-policy nature reuses data from the replay buffer, requiring 5-10x fewer environment interactions than PPO; (2) automatic entropy tuning—SAC's entropy regularization prevents premature convergence to a single gait, encouraging exploration of diverse locomotion strategies; (3) stability—SAC's clipped double-Q critic reduces overestimation bias. The algorithm uses a replay buffer of 1M transitions, batch size 256, learning rate 3e-4 for all networks, soft target update τ=0.005, and discount factor γ=0.99.

The reward function is a weighted sum of multiple components: r = w_vel * r_velocity + w_orn * r_orientation + w_energy * r_energy + w_smooth * r_smoothness + w_foot * r_foot_clearance + w_alive * r_alive. The velocity reward tracks a commanded velocity: r_velocity = exp(-4 * ||v_cmd - v_actual||²). The orientation reward penalizes deviation from upright: r_orientation = exp(-3 * ||q_base - q_upright||²). The energy reward penalizes joint power: r_energy = -Σ|τ_i * ω_i|. The smoothness reward penalizes action changes: r_smoothness = -||a_t - a_{t-1}||². The foot clearance reward encourages swing leg height: r_foot_clearance = Σ max(0, h_target - h_foot) for swing legs. The alive bonus r_alive = 1.0 is given at each step the robot has not fallen.

Sim-to-real transfer is addressed through several mechanisms. First, the actuator model is carefully calibrated: rather than ideal PD controllers, the simulation includes motor friction, backlash, current limits, and thermal constraints measured from the real Solo12 motors. Second, sensor noise is injected during training: Gaussian noise on joint positions (σ=0.01 rad), joint velocities (σ=0.1 rad/s), and IMU (orientation σ=0.02 rad, angular velocity σ=0.05 rad/s). Third, domain randomization covers: mass (±15%), friction coefficient (0.4-1.2), motor strength (±10%), ground restitution (0.0-0.3), and PD gains (±10%). Fourth, action limits are conservatively set to avoid configurations unreachable on real hardware.

The comparison with model-based controllers uses a Model Predictive Controller (convex MPC formulating a simplified single rigid body dynamics model) and a Whole Body Controller (WBC using full rigid body dynamics with contact constraints). Both baselines are well-tuned implementations from the open-source control literature. The comparison evaluates: velocity tracking accuracy, energy consumption, robustness to external pushes (50N lateral impulse), and terrain adaptability (small obstacles, ±3cm surface perturbations).

## Key Results & Numbers
- Velocity tracking error: SAC 0.05 m/s (RMSE) vs. MPC 0.08 m/s vs. WBC 0.06 m/s at 0.5 m/s target
- Energy consumption: SAC 12.3 J/m vs. MPC 14.7 J/m vs. WBC 13.1 J/m (SAC learns more efficient gaits)
- Robustness to 50N lateral push: SAC recovers 92% of trials vs. MPC 78% vs. WBC 85%
- Training sample efficiency: SAC converges in 2M timesteps (~11 hours wall-clock) vs. PPO baseline requiring 20M timesteps
- Sim-to-real velocity tracking degradation: 0.05 m/s (sim) → 0.09 m/s (real), a 80% increase in error
- Domain randomization reduces sim-to-real gap by 45% (velocity tracking error: 0.16 without DR → 0.09 with DR)
- Motor model calibration contributes 30% of the sim-to-real gap reduction
- Gait emergence: trotting gait emerges naturally at 0.3-0.8 m/s; bounding at 0.8-1.2 m/s; transition is smooth
- Real robot operation: stable walking at 0.1-0.6 m/s, turning up to 1.0 rad/s, robust to small terrain irregularities

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to the Mini Cheetah project. Solo12 and Mini Cheetah are comparable 12-DoF quadrupeds, and the lessons transfer directly. The SAC vs. PPO comparison (SAC is 5-10x more sample-efficient) is valuable for algorithm selection on Mini Cheetah. The detailed sim-to-real analysis provides a practical checklist: motor model calibration, sensor noise modeling, domain randomization ranges, and conservative action limits. The open-source pipeline could be adapted for Mini Cheetah with minimal modification.

The reward function design with explicit velocity tracking, energy minimization, and smoothness terms provides a tested starting point for Mini Cheetah reward engineering. The comparison with model-based controllers (MPC, WBC) on the same hardware is particularly informative for justifying the RL approach and understanding where RL excels (robustness, energy efficiency) and where it lags (precise velocity tracking at low speeds).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The off-policy SAC training insights are transferable to Cassie's Controller level, where sample-efficient learning is desirable. The systematic sim-to-real transfer analysis (motor modeling, sensor noise, domain randomization) provides methodology applicable to Cassie's deployment. The reward function design principles (multi-component with careful weighting) are relevant for Cassie, though the specific terms would differ due to bipedal balance requirements.

However, the paper focuses on quadruped locomotion, which has fundamentally different balance dynamics than bipedal locomotion. The solo quadruped's static stability margin provides robustness not available to Cassie. The hierarchical RL aspects of Cassie's project are not addressed by this paper, limiting its relevance to the lower levels of the hierarchy (Controller, parts of Primitives).

## What to Borrow / Implement
- Adapt the SAC training pipeline for Mini Cheetah, leveraging the sample efficiency advantage over PPO for faster iteration
- Use the systematic sim-to-real transfer checklist: motor calibration → sensor noise → domain randomization → action limits
- Adopt the multi-component reward function structure (velocity tracking, orientation, energy, smoothness, foot clearance) as a starting template for both robots
- Implement the automatic entropy tuning mechanism from SAC to encourage gait diversity during training
- Use the MPC/WBC comparison methodology to benchmark RL policies against traditional controllers

## Limitations & Open Questions
- Solo12 is a lightweight research platform (2.5 kg) with limited torque; results may not transfer directly to heavier production robots like Mini Cheetah (9 kg) or ANYmal (30 kg)
- SAC's sample efficiency advantage is demonstrated in simulation; real-world sample efficiency (critical for hardware training) is not evaluated
- The sim-to-real gap remains significant (80% velocity tracking error increase), suggesting that current domain randomization and motor modeling are insufficient for precision tasks
- The paper does not address high-speed locomotion (>1 m/s), highly dynamic maneuvers, or challenging terrains (stairs, slopes) where the gap between RL and model-based methods may differ
