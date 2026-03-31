# Unitree Go2 EDU: RL Training and Deployment Tutorial

**Authors:** TheX1an (2024)
**Year:** 2024 | **Venue:** GitHub Tutorial
**Links:** [GitHub](https://github.com/TheX1an/Unitree-Go2-EDU-RL-Training-and-Deployment-Tutorial)

---

## Abstract Summary
This repository provides a comprehensive, step-by-step tutorial for training reinforcement learning locomotion policies in simulation and deploying them on the Unitree Go2 EDU quadruped robot. The tutorial covers the entire pipeline from simulation environment setup using NVIDIA Isaac Lab, through RL policy training with PPO, to real-world deployment on the Go2 EDU hardware equipped with a Jetson Orin Nano. It bridges the gap between academic RL research and practical robotic deployment by addressing the numerous engineering details that papers often omit.

The tutorial is structured in four phases: (1) Environment Setup—installing Isaac Lab, configuring the Go2 simulation environment, and verifying the physics simulation; (2) Policy Training—configuring rewards, domain randomization, curriculum, and PPO hyperparameters, then training to convergence; (3) Policy Export—converting the trained PyTorch model to a deployment-ready format and validating in simulation; (4) Hardware Deployment—setting up the Go2 EDU's onboard computer, integrating with Unitree SDK2, implementing the real-time control loop, and conducting physical experiments.

A key contribution is the detailed treatment of practical deployment challenges: control loop timing, sensor latency compensation, safety limits, emergency stop procedures, battery management during experiments, and debugging real-time performance issues. The tutorial also documents common failure modes and their solutions, providing troubleshooting guidance for researchers new to sim-to-real deployment.

## Core Contributions
- End-to-end tutorial from Isaac Lab simulation setup to physical Go2 EDU hardware deployment with detailed instructions
- Practical deployment engineering details: control loop timing, sensor latency, safety limits, emergency stop implementation
- SDK2 integration guide for direct low-level communication with Go2 EDU actuators and sensors
- Policy export pipeline: PyTorch → TorchScript/ONNX → Jetson Orin deployment with inference optimization
- Comprehensive troubleshooting guide documenting common failure modes and solutions for sim-to-real transfer
- Real-world validation on various terrains (indoor, outdoor, slopes) with performance characterization
- Modular code structure with clear separation between training, export, and deployment components

## Methodology Deep-Dive
**Phase 1: Environment Setup.** The tutorial begins with installing NVIDIA Isaac Lab (version 1.0+) with CUDA support and the IsaacLabExtension for Unitree robots. The Go2 URDF is loaded with carefully calibrated collision meshes and inertial parameters matching the physical robot. The simulation environment includes configurable terrain generation (flat, rough, stairs), sensor simulation (IMU with noise models, joint encoder noise), and visualization tools. The tutorial emphasizes verifying the simulation's physical accuracy by comparing simulated joint responses to measured hardware responses.

**Phase 2: Policy Training.** The RL training uses PPO with a standard MLP policy network. The observation space includes proprioceptive signals (joint positions/velocities relative to default stance, projected gravity, base angular velocity, previous actions) totaling 48 dimensions. The action space is 12-dimensional joint position targets. The reward function follows the established quadruped locomotion template: velocity tracking (forward, lateral, yaw), stability penalties (roll/pitch rate, z-velocity), energy penalties (torque, joint velocity, action rate), and gait rewards (foot air time, contact pattern). Domain randomization covers friction (0.5–1.5), payload mass (0–2 kg added), motor strength (0.85–1.15×), and push perturbations (random forces up to 10N).

The curriculum starts with flat terrain and commanded velocities in a narrow range (0–0.5 m/s forward), then progressively introduces rougher terrain, higher velocities (up to 1.5 m/s), and lateral/turning commands. Progression criteria are based on average episode return exceeding thresholds. Training runs for approximately 1000 PPO iterations (2–4 hours depending on GPU) with 2048 parallel environments.

**Phase 3: Policy Export.** The trained policy is exported using two methods: (1) TorchScript via torch.jit.trace for PyTorch-native deployment, and (2) ONNX via torch.onnx.export for cross-platform deployment. The tutorial validates the exported model by running inference in the simulation environment and comparing outputs against the original PyTorch model, ensuring numerical consistency within tolerance (max absolute difference < 1e-5). Inference timing is benchmarked on the Jetson Orin Nano to confirm real-time feasibility.

**Phase 4: Hardware Deployment.** The deployment controller is implemented in Python (with critical paths in C++ via pybind11 for latency reduction). The control loop architecture is: (1) Read sensor data from Go2 via SDK2 at 500 Hz; (2) Construct observation vector with proper normalization and coordinate frame transformations; (3) Run policy inference at 50 Hz (every 10th sensor reading); (4) Clip and safety-check output joint position targets; (5) Send commands to Go2 actuators via SDK2. Safety measures include: joint position limits enforced in software (±0.5 rad from safe range per joint), velocity limits (max joint velocity 10 rad/s), torque limits, watchdog timer (if no policy output for >100ms, transition to standing pose), and a hardware emergency stop button.

The tutorial addresses sensor latency: the IMU has ~2ms latency and joint encoders ~1ms, which is compensated by using the most recent readings and applying a simple forward prediction. Observation normalization uses running statistics computed during training, loaded from the training checkpoint to ensure consistency between simulation and deployment.

## Key Results & Numbers
- Training time: 2–4 hours on RTX 3080/4090 with 2048 parallel environments for fully converged policy
- Policy inference time on Jetson Orin Nano: ~3ms for TorchScript, ~2ms for ONNX Runtime
- Control loop frequency: 50 Hz policy / 500 Hz sensor reading, well within real-time requirements
- Successful deployment on: indoor tile, carpet, outdoor concrete, grass, mild slopes (up to ~10°)
- Velocity tracking on flat terrain: commanded 1.0 m/s → achieved 0.92 ± 0.05 m/s
- Fall rate: <5% over 100 trial runs of 30-second episodes on flat terrain; <15% on rough terrain
- Battery life during RL policy execution: ~45 minutes of continuous locomotion on Go2 EDU's standard battery
- Deployment setup time from trained policy to walking robot: ~2 hours for first-time setup, ~15 minutes for subsequent deployments

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This tutorial provides the most practical and detailed reference for the Mini Cheetah's sim-to-real deployment workflow. While the target hardware differs (Go2 EDU vs Mini Cheetah), the deployment engineering challenges are universal: control loop timing, sensor latency compensation, safety limits, observation normalization consistency, and policy export/inference optimization. The specific solutions (watchdog timer, joint position clipping, running statistics export) can be directly adapted to the Mini Cheetah platform.

The troubleshooting guide is particularly valuable, as it documents failure modes that are common to all quadruped RL deployments: observation normalization mismatches between sim and real, action scaling errors, IMU frame convention differences, and PD gain sensitivity. Learning from these documented failures can save weeks of debugging time on the Mini Cheetah.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The deployment engineering patterns are transferable to Cassie, particularly the safety layer implementation (joint limits, velocity limits, watchdog timer) which aligns with Cassie's Safety level (CBF-QP). The policy export and inference optimization techniques (TorchScript/ONNX, Jetson deployment) are directly applicable regardless of robot morphology. However, Cassie's more complex hierarchical architecture means that the simple single-policy deployment pattern needs to be extended to deploy multiple interacting policy components.

The sensor latency compensation approach is relevant to Cassie, where IMU latency and encoder latency affect the state estimation inputs to the hierarchy. The observation normalization consistency check (comparing running statistics between sim and real) is critical for any sim-to-real pipeline.

## What to Borrow / Implement
- Control loop architecture with explicit frequency separation (500 Hz sensor, 50 Hz policy) and timing guarantees
- Safety layer implementation: joint position clipping, velocity limits, watchdog timer, emergency stop integration
- Observation normalization pipeline: export running statistics from training, load at deployment for exact consistency
- Policy export validation: compare PyTorch vs TorchScript/ONNX outputs to ensure numerical consistency before deployment
- Sensor latency compensation strategy with simple forward prediction for real-time control

## Limitations & Open Questions
- Tutorial is Go2-specific; adaptation to Mini Cheetah or Cassie requires significant hardware interface changes
- Python deployment controller with C++ critical paths may not achieve the lowest possible latency; a full C++ implementation would be more robust
- Single flat policy deployment only; no guidance for deploying hierarchical multi-policy architectures (critical for Cassie)
- No handling of exteroceptive sensors (cameras, LiDAR); deployment is proprioception-only, limiting terrain-awareness capabilities
