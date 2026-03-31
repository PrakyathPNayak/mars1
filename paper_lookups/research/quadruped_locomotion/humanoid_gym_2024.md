# Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim-to-Real Transfer

**Authors:** Xinyang Gu, Yen-Jen Wang, Jianyu Chen
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2404.05695

---

## Abstract Summary
Humanoid-Gym presents a reinforcement learning framework specifically designed for humanoid robot locomotion, extending the Legged Gym paradigm to bipedal and humanoid platforms. The framework adopts a dual-simulator approach: Isaac Gym is used for fast, massively parallel policy training (leveraging GPU acceleration for thousands of environments), while MuJoCo serves as an independent validation environment for sim-to-sim transfer testing before real-world deployment. This dual-simulator methodology provides a crucial intermediate validation step that catches simulation-specific artifacts before they reach hardware.

The core contribution is demonstrating that the Legged Gym training pipeline, originally designed for quadrupeds, can be successfully adapted for humanoid locomotion with appropriate modifications to the reward function, observation space, and training curriculum. The framework achieves zero-shot sim-to-real transfer for stable humanoid walking, validating the sim-to-sim approach as a reliable predictor of real-world performance. Policies that transfer successfully between Isaac Gym and MuJoCo consistently transfer to real hardware, while policies that fail sim-to-sim transfer also fail on hardware.

The framework is open-source and provides reference configurations for multiple humanoid platforms, establishing a standardized training pipeline that reduces the engineering effort required to train humanoid locomotion policies from scratch.

## Core Contributions
- **Dual-simulator validation pipeline** using Isaac Gym for training and MuJoCo for sim-to-sim transfer testing, providing a reliable pre-deployment quality gate
- **Humanoid-adapted Legged Gym** with bipedal-specific reward terms, observation spaces, and training curricula
- **Zero-shot sim-to-real transfer** demonstrated on humanoid hardware with stable walking gaits
- **Sim-to-sim transfer protocol** that identifies policies likely to succeed in real-world deployment by testing in an independent simulator
- **Open-source framework** with reference configurations for humanoid platforms, lowering the barrier to entry for humanoid RL research
- **Systematic comparison** of training configurations (reward weights, domain randomization, curriculum schedules) for humanoid vs. quadruped locomotion

## Methodology Deep-Dive
The training pipeline uses Isaac Gym with 4096 parallel environments, each containing a humanoid robot model loaded from URDF. The humanoid observation space is expanded compared to quadrupeds to include: base linear and angular velocity (6), projected gravity (3), velocity commands (3), joint positions (number of actuated joints, typically 10-20 for lower body), joint velocities (matching joint count), and previous actions. Additional observations specific to humanoid locomotion include: foot contact binary flags (2), center-of-pressure estimate (2), and phase signals for gait timing (2, representing the desired phase of left and right legs as sinusoidal signals).

The action space consists of joint position targets for all actuated lower-body joints, processed through PD controllers. For a typical humanoid, this includes hip roll, hip pitch, hip yaw, knee pitch, and ankle pitch for each leg (10 DoF total). The PD gains are tuned per-joint to match the physical actuator characteristics. Actions are output at 50 Hz, with the physics simulation running at 200 Hz (4 substeps per policy step).

The reward function is heavily modified from the quadruped version. Key bipedal-specific reward terms include: (1) feet air time reward that encourages alternating single-support phases with appropriate swing durations (critical for bipedal gait timing); (2) base height reward that penalizes deviation from a target standing height; (3) feet distance reward that encourages a target step width to prevent leg crossing or excessive lateral spread; (4) symmetry reward that encourages left-right symmetric joint trajectories, promoting human-like gait; (5) zero-moment point (ZMP) reward that encourages the ground projection of the center of pressure to remain within the support polygon; and (6) phase tracking reward that aligns foot contact events with the desired gait phase signals.

The dual-simulator validation works as follows: after training in Isaac Gym, the policy network weights are exported and loaded into a MuJoCo environment with the same robot model (converted from URDF to MJCF). The policy is then evaluated in MuJoCo across a set of test scenarios (flat ground walking, velocity tracking, disturbance recovery). Key metrics are compared between Isaac Gym and MuJoCo: average forward velocity, lateral velocity error, average torque, energy consumption, and stability (time before falling). A policy is considered transfer-ready if its MuJoCo performance is within 20% of its Isaac Gym performance on all key metrics.

The domain randomization includes standard physics parameters (friction, mass, CoM offset, motor strength) plus humanoid-specific randomizations: foot-ground contact stiffness and damping (which significantly affect bipedal balance), ankle joint range limits (simulating loose vs. tight ankle mechanical play), and gait timing perturbations (varying the desired step frequency by plus or minus 15%).

## Key Results & Numbers
- **Training time:** Stable humanoid walking policy trained in approximately 2-4 hours on a single RTX 4090 (4096 environments), significantly longer than quadruped training (~20 min) due to the difficulty of bipedal balance
- **Sim-to-sim transfer:** Policies that achieve less than 15% performance gap between Isaac Gym and MuJoCo consistently transfer to real hardware; policies with greater than 30% gap fail on hardware
- **Walking speed:** Achieves stable forward walking at 0.3-0.6 m/s on flat ground with zero-shot sim-to-real transfer
- **Disturbance rejection:** Recovers from lateral pushes up to 30N applied for 0.2 seconds in both simulation and real world
- **Gait quality:** Mean step length of 0.35m, step width of 0.18m, consistent with human-like proportions for the robot's size
- **Energy efficiency:** Average CoT (Cost of Transport) of 3.5, comparable to other humanoid RL approaches
- **MuJoCo validation accuracy:** 89% of policies that pass sim-to-sim validation also succeed in real-world deployment

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The dual-simulator validation approach is applicable to Mini Cheetah, providing a sim-to-sim quality gate before real hardware deployment. Train in Isaac Gym (or MuJoCo with GPU acceleration), then validate in the other simulator. If policies transfer between simulators, they are more likely to transfer to real hardware. This is particularly valuable for Mini Cheetah where hardware time is expensive and risky.

The systematic comparison of quadruped vs. humanoid training configurations provides insights into which Legged Gym components need modification for different robot morphologies. While Mini Cheetah is a quadruped (closer to Legged Gym's default), the paper's analysis of how reward terms, observation spaces, and curricula need to change for different platforms is instructive for tuning the Mini Cheetah training pipeline.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Humanoid-Gym is directly relevant to Cassie bipedal training. The bipedal-specific reward terms (feet air time, ZMP, symmetry, base height, phase tracking) are immediately applicable to Cassie's Controller-level policy training. The dual-simulator validation approach is critical for Cassie's sim-to-real pipeline, given Cassie's expensive hardware and the difficulty of bipedal balance where sim-to-real failures often result in hardware damage.

Specific elements for Project B include: (1) the gait phase signal design (sinusoidal phase inputs that help the policy learn periodic gaits) directly relates to the Neural ODE Gait Phase component in Project B; (2) the ZMP reward term can be integrated into the Controller level's reward function to encourage dynamic balance; (3) the sim-to-sim validation protocol provides a pre-deployment safety check before each policy iteration is tested on Cassie hardware; and (4) the 2-4 hour training time estimate for humanoid locomotion helps calibrate expectations for Cassie training time. The observation space design (including foot contact flags and CoP estimates) informs the proprioceptive input design for Cassie's lower hierarchy levels.

## What to Borrow / Implement
- **Dual-simulator validation protocol:** train in one simulator, validate in another before hardware deployment, using the 15-20% performance gap threshold as the transfer-readiness criterion
- **Bipedal reward terms:** feet air time, ZMP, symmetry, base height, and phase tracking rewards for Cassie's Controller-level training
- **Gait phase signal design:** sinusoidal phase inputs for periodic gait encouragement, feeding into Neural ODE Gait Phase module design
- **Humanoid domain randomization:** foot-ground contact stiffness, ankle range limits, and gait timing perturbations specific to bipedal dynamics
- **Sim-to-sim transfer metrics:** velocity tracking, torque, energy, and stability as standard evaluation metrics across simulators

## Limitations & Open Questions
- **Limited to flat ground:** demonstrated sim-to-real transfer is primarily on flat ground; stairs, slopes, and rough terrain are not validated in the real world
- **Slow training relative to quadrupeds:** 2-4 hours vs. 20 minutes reflects the fundamental difficulty of bipedal balance; hierarchical decomposition (as in Project B) may be needed to reduce training time
- **No hierarchical control:** the framework trains a flat policy, which may not scale to the diverse locomotion behaviors needed for complex tasks (walking, running, turning, recovering from falls)
- **Limited actuator modeling:** the PD controller abstraction may not capture the full dynamics of real humanoid actuators, particularly for Cassie's series-elastic actuators and leaf spring compliance
