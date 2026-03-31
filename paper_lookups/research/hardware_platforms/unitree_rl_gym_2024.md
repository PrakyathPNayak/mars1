# Unitree RL Gym: An Open-Source Framework for Reinforcement Learning on Unitree Robots

**Authors:** Unitree Robotics
**Year:** 2024 | **Venue:** GitHub/Open-Source
**Links:** https://github.com/unitreerobotics/unitree_rl_gym

---

## Abstract Summary
Unitree RL Gym is an open-source reinforcement learning framework developed by Unitree Robotics for training locomotion policies on their robot platforms, including the Go1, Go2 quadrupeds and H1, G1 humanoid robots. The framework provides a complete end-to-end pipeline from RL training in simulation to deployment on physical hardware, covering Train, Play (evaluation), Sim2Sim (cross-simulator validation), and Sim2Real (hardware deployment) stages.

The framework supports dual simulation backends: NVIDIA Isaac Gym for GPU-accelerated parallel training and MuJoCo for physics-accurate validation and deployment verification. This dual-backend approach enables researchers to leverage Isaac Gym's massive parallelism for fast training iterations while using MuJoCo as a higher-fidelity validation environment. The Sim2Sim pipeline between Isaac Gym and MuJoCo serves as an intermediate validation step before real-world deployment, catching policies that exploit simulator-specific artifacts.

The codebase includes carefully tuned reward functions for velocity tracking, energy efficiency, gait style, and recovery behaviors. It provides pre-configured environments for each robot platform with calibrated URDF/MJCF models, actuator dynamics, and sensor configurations. The community-driven development model has led to continuous improvements in reward design, domain randomization, and deployment procedures, making it a practical reference for anyone building RL-based locomotion systems for legged robots.

## Core Contributions
- Complete Train-Play-Sim2Sim-Sim2Real pipeline for Unitree quadruped and humanoid robots
- Dual simulation backend support (Isaac Gym and MuJoCo) with cross-simulator validation
- Calibrated robot models (URDF/MJCF) for Go1, Go2, H1, and G1 platforms
- Carefully tuned reward functions for velocity tracking, energy efficiency, and gait quality
- Domain randomization configurations validated through sim-to-real transfer
- Open-source community with continuous improvements and real-world deployment demonstrations
- Reference implementation for the full RL locomotion pipeline applicable to other robot platforms

## Methodology Deep-Dive
The training pipeline uses PPO (Proximal Policy Optimization) as the core RL algorithm, implemented through the rsl_rl library (originally developed at RSL, ETH Zurich). The policy network is an MLP with configurable hidden layers (typically 3 layers of 128 or 256 units), taking proprioceptive observations as input and outputting target joint positions. The observations include joint positions (relative to default standing configuration), joint velocities, projected gravity vector, commanded velocity (linear x, y and angular yaw), and previous actions. A short history of observations (typically 3-5 timesteps) provides temporal context.

The reward function is a weighted sum of multiple terms designed to encourage natural and efficient locomotion. The primary reward is velocity tracking: r_vel = exp(-||v_cmd - v_actual||^2 / sigma_v), which uses a Gaussian kernel to reward proximity to the commanded velocity. Secondary rewards include angular velocity tracking (yaw rate), energy penalty (sum of |tau * q_dot| across joints), action smoothness (penalizing action differences between timesteps), foot contact pattern (encouraging alternating foot contacts for trot gait), body orientation (penalizing roll and pitch deviations), and foot clearance (encouraging sufficient foot height during swing phase).

Domain randomization is applied to bridge the sim-to-real gap. The randomized parameters include ground friction (0.3 to 1.5), payload mass (added to the body, 0 to 3kg for Go2), motor strength scaling (0.85 to 1.15), PD gain variations (plus/minus 10%), observation noise (Gaussian with per-sensor standard deviations), and external force perturbations (random pushes up to 20N for 0.1-0.3s). Terrain randomization includes flat ground, rough terrain (height noise up to 3cm), slopes (up to 15 degrees), and stairs (8-15cm step height).

The Sim2Sim validation pipeline trains the policy in Isaac Gym, exports the neural network weights, and then evaluates the same policy in MuJoCo. The MuJoCo environment uses calibrated MJCF models of the robots with independently measured physical parameters. Discrepancies between Isaac Gym and MuJoCo performance indicate policies that may exploit simulator-specific contact dynamics or integration artifacts. This cross-simulator validation significantly improves sim-to-real transfer success rates by catching overfitting to a single simulator's characteristics.

The Sim2Real deployment uses the trained policy exported as an ONNX model running on the robot's onboard computer. The control loop runs at 50Hz (matching the training frequency), with the onboard IMU providing body orientation and angular velocity, and motor encoders providing joint positions and velocities. A simple state estimator computes the projected gravity vector from IMU data. The commanded velocity is provided through a joystick or higher-level navigation planner.

## Key Results & Numbers
- Successful sim-to-real transfer on Unitree Go2 quadruped with stable outdoor walking
- H1 humanoid locomotion policy achieving stable walking at up to 1.5 m/s
- Sim2Sim validation showing less than 10% performance gap between Isaac Gym and MuJoCo
- Training from scratch to deployment-ready policy in approximately 30 minutes on a single GPU (Isaac Gym)
- Community contributions including improved reward functions, new terrain types, and deployment scripts
- Go2 achieving robust locomotion on grass, gravel, and gentle slopes after sim-to-real transfer
- Energy efficiency improvements of 15-20% through reward function tuning compared to initial baselines

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Unitree RL Gym serves as a direct reference implementation for the Mini Cheetah RL pipeline. The framework's architecture—PPO with MLP policy, proprioceptive observations, domain randomization, and MuJoCo simulation—closely matches the planned Mini Cheetah setup. The reward function design (velocity tracking with Gaussian kernel, energy penalty, action smoothness, foot contact pattern) provides a proven starting point that can be adapted for Mini Cheetah's specific dynamics.

The Sim2Sim validation approach (Isaac Gym to MuJoCo) is directly applicable: even if Mini Cheetah training uses only MuJoCo, validating across simulator configurations (different integration timesteps, contact parameters) provides similar benefits. The domain randomization ranges (friction, mass, motor strength) are well-validated through real-world deployment and serve as concrete reference values for Mini Cheetah training. The entire Train-Play-Sim2Sim-Sim2Real pipeline can be adapted with minimal modifications by swapping the robot model.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The H1 humanoid training pipeline in Unitree RL Gym provides a reference for bipedal RL training that is relevant to Cassie. While Cassie's dynamics are significantly different from H1 (Cassie has passive springs, underactuated ankles, and a different morphology), the general approach—PPO training with proprioceptive observations and domain randomization—transfers. The reward function components for bipedal locomotion (balance, gait periodicity, foot clearance) provide starting points for Cassie's Controller level.

The Sim2Sim validation concept is particularly valuable for Project B, where the MuJoCo 3.4.0 Cassie model should be validated against other simulators or model configurations to ensure the learned hierarchical controller doesn't overfit to specific simulation artifacts. The dual-backend approach could be adapted for Cassie by training in Isaac Gym for speed and validating in MuJoCo for accuracy.

## What to Borrow / Implement
- Adapt the complete Train-Play-Sim2Sim-Sim2Real pipeline architecture for Mini Cheetah, swapping the robot model and tuning reward weights
- Use the reward function template (Gaussian velocity tracking, energy penalty, smoothness, contact pattern) as a starting point for both projects
- Implement the Sim2Sim validation procedure between different MuJoCo configurations to catch simulator-specific overfitting
- Adopt the domain randomization ranges (friction 0.3-1.5, mass variations, motor strength 85-115%) as validated starting points
- Study the H1 humanoid training configuration for insights applicable to Cassie bipedal training

## Limitations & Open Questions
- Framework is optimized for Unitree robots; adaptation to other platforms (Mini Cheetah, Cassie) requires model calibration and reward tuning
- The MLP policy architecture may not be sufficient for the complex hierarchical controller required by Project B
- Limited support for exteroceptive sensing (cameras, lidar) in the current pipeline—primarily proprioceptive
- Community-driven development means quality and documentation vary; some features may be experimental or poorly documented
