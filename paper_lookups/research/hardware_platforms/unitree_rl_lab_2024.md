# Unitree RL Lab: Open-Source Reinforcement Learning Framework for Unitree Robots

**Authors:** Unitree Robotics
**Year:** 2024 | **Venue:** GitHub / Technical Report
**Links:** [GitHub](https://github.com/unitreerobotics/unitree_rl_lab)

---

## Abstract Summary
Unitree RL Lab is the official open-source reinforcement learning framework developed by Unitree Robotics for training and deploying RL-based locomotion controllers on their robot platforms, including the Go2 quadruped, H1 humanoid, and G1 humanoid. The framework provides a complete pipeline from RL training in simulation through sim-to-sim validation to real-world hardware deployment. It is built on top of NVIDIA Isaac Lab (formerly Isaac Gym) for high-throughput parallel simulation and uses standard RL algorithms (primarily PPO) for policy training.

The framework integrates several key components: a configurable environment suite with task definitions for locomotion (velocity tracking, terrain traversal), domain randomization modules for robust sim-to-real transfer, curriculum learning for progressive difficulty scaling, policy export utilities (ONNX and PyTorch formats), and a C++ deployment controller for real-time policy inference on robot hardware. The framework also includes utilities for sim-to-sim transfer (testing policies in different simulators before hardware deployment) and comprehensive logging/visualization tools.

Unitree RL Lab represents a production-quality reference implementation of the modern quadruped/humanoid RL deployment pipeline, incorporating lessons learned from deploying RL controllers on thousands of commercial robots. Its open-source release provides the robotics community with a tested, well-documented codebase that can be adapted to other robot platforms.

## Core Contributions
- End-to-end Train→Play→Sim2Sim→Sim2Real pipeline for legged robot RL, validated on commercial hardware
- Isaac Lab integration for massively parallel environment simulation (4096+ environments simultaneously)
- Comprehensive domain randomization module covering physics parameters (friction, mass, motor strength), sensor noise, and terrain properties
- Curriculum learning framework with configurable progression metrics and difficulty schedules
- Policy export in ONNX and PyTorch formats for cross-platform deployment flexibility
- C++ deployment controller for real-time inference (~500 Hz) on robot embedded systems
- SDK2 integration for direct hardware communication with Unitree robots (Go2, H1, G1)

## Methodology Deep-Dive
The training pipeline begins with environment configuration in Isaac Lab. Each environment instance simulates a single robot on a procedurally generated terrain tile. Terrains are sampled from a configurable distribution: flat, rough, slopes, stairs, discrete obstacles. The robot is parameterized by its URDF, and the simulation runs at 200 Hz physics / 50 Hz control frequency. Observations include proprioceptive signals (joint positions, velocities, torques, IMU orientation, angular velocity) and optionally exteroceptive signals (height scans from raycasts).

The RL algorithm is PPO with Generalized Advantage Estimation (GAE). The policy network is a standard MLP (3 layers, 256 units, ELU activation) that outputs mean actions for a Gaussian policy. The value function shares the same architecture but with a separate head. Key hyperparameters: learning rate 1e-4 with cosine annealing, discount γ = 0.99, GAE λ = 0.95, clip ratio ε = 0.2, entropy coefficient 0.01 (annealed), minibatch size 4096, 5 epochs per PPO update.

Domain randomization is applied at environment reset and continuously during episodes. Randomized parameters include: ground friction coefficient (0.5–1.5), robot base mass (±15%), motor strength scaling (0.85–1.15), joint damping (±20%), push perturbation forces (random impulses up to 10N applied to the base), and sensor noise (Gaussian noise on joint encoders ±0.01 rad, IMU orientation ±2°). Terrain curriculum progresses from flat surfaces to increasingly rough terrain based on the agent's average velocity achievement.

The reward function is a weighted sum of terms: (1) velocity tracking reward—exponential penalty for deviation from commanded linear and angular velocities; (2) base stability—penalizes excessive roll, pitch, and z-axis oscillation; (3) energy efficiency—penalizes large joint torques and joint velocity; (4) foot clearance—rewards sufficient foot height during swing phase; (5) smoothness—penalizes large action changes between timesteps; (6) contact pattern—rewards desired gait patterns. Reward weights are tuned per task.

Policy export converts the trained PyTorch model to ONNX format using torch.onnx.export with operator set version 11. The ONNX model is then loaded into the C++ deployment controller, which runs ONNX Runtime for inference. The controller communicates with the robot via Unitree SDK2, sending joint position targets at 50 Hz. A safety layer clips joint position commands to predefined limits and enforces velocity/torque constraints.

Sim-to-sim transfer is validated by loading the trained policy into MuJoCo (separate from the Isaac Lab training simulator) and verifying that locomotion performance is maintained. This intermediate step catches simulation-specific artifacts before hardware deployment.

## Key Results & Numbers
- Training time: ~30 minutes for basic locomotion on a single NVIDIA RTX 4090 with 4096 parallel environments
- Velocity tracking error: <0.1 m/s on flat terrain, <0.2 m/s on moderate rough terrain
- Sim-to-real transfer success rate: >90% across Go2 hardware units with standard domain randomization
- Deployment controller inference time: ~2ms per step in C++ ONNX Runtime (500 Hz capable)
- Supported platforms: Go2 (12 DoF quadruped), H1 (19 DoF humanoid), G1 (23 DoF humanoid)
- Domain randomization with 6 parameter categories significantly reduces sim-to-real gap compared to no randomization
- Curriculum learning reaches maximum terrain difficulty ~2× faster than fixed difficulty training

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Unitree RL Lab serves as a near-ideal reference implementation for the Mini Cheetah RL pipeline. The Go2 quadruped has a very similar morphology to the Mini Cheetah (12 DoF, similar mass class), making the training configurations directly transferable. The reward function design, domain randomization ranges, curriculum learning schedule, and PPO hyperparameters provide thoroughly tested starting points for Mini Cheetah training.

The sim-to-real pipeline (Isaac Lab → ONNX export → C++ controller) can be adapted to Mini Cheetah hardware by replacing the Unitree SDK2 interface with the MIT Mini Cheetah SDK. The terrain curriculum design and the specific domain randomization parameter ranges are informed by real-world deployment experience on thousands of units, giving them a reliability advantage over ad-hoc parameter choices.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The H1 humanoid pipeline within Unitree RL Lab demonstrates successful RL training and deployment for a high-DoF bipedal platform, providing a direct reference for Cassie. While H1's morphology differs from Cassie's (H1 has arms, Cassie does not), the bipedal locomotion aspects—balance, gait generation, terrain adaptation—are shared challenges. The PPO training setup, reward function structure for bipedal walking (with appropriate modifications for Cassie's underactuation), and domain randomization strategy are all applicable.

The deployment pipeline (policy export → C++ controller → real-time inference) provides a tested template for Cassie's own deployment workflow. The sim-to-sim validation step (Isaac Lab → MuJoCo) is particularly relevant since Cassie's training likely uses MuJoCo as the primary simulator.

## What to Borrow / Implement
- Complete reward function structure with velocity tracking, stability, energy efficiency, foot clearance, and smoothness terms
- Domain randomization ranges for physical parameters (friction 0.5–1.5, mass ±15%, motor strength 0.85–1.15) as tested starting points
- PPO hyperparameters (lr=1e-4 cosine, γ=0.99, λ=0.95, ε=0.2) validated on commercial hardware deployment
- ONNX policy export pipeline for deployment flexibility across different inference runtimes
- Terrain curriculum with velocity-based progression metric for progressive difficulty scaling

## Limitations & Open Questions
- MLP policy architecture only; no GNN or attention-based policy options, limiting morphology-aware learning
- Single-level RL (flat PPO); no hierarchical control architecture, which limits applicability to Cassie's 4-level hierarchy
- Domain randomization ranges are Unitree-specific; Mini Cheetah and Cassie have different physical parameter distributions requiring re-tuning
- No support for exteroceptive sensing beyond basic height scans; vision-based policies not included
