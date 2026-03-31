# Reinforcement Learning Framework for Go2 Quadruped Locomotion in Isaac Lab

**Authors:** (2024)
**Year:** 2024 | **Venue:** GitHub
**Links:** [GitHub](https://github.com/jazhanma/go2-rl-locomotion)

---

## Abstract Summary
This project presents a complete reinforcement learning training and deployment framework for the Unitree Go2 quadruped robot using NVIDIA Isaac Lab. The framework focuses on velocity-tracking locomotion over randomized terrains, with the primary goal of producing a policy that transfers reliably from simulation to the real Go2 hardware. The pipeline encompasses environment setup with procedural terrain generation, PPO-based policy training with curriculum learning and domain randomization, policy evaluation and visualization, and export for hardware deployment.

The framework implements a random-box terrain curriculum where the difficulty of terrain features (box height, spacing, randomness) increases as the agent demonstrates competency at current levels. The training produces a policy that tracks commanded forward, lateral, and yaw velocities while maintaining stability and energy efficiency. Domain randomization over friction, mass, motor strength, and external perturbations ensures that the policy is robust to the inevitable discrepancies between simulation and reality.

The project demonstrates successful sim-to-real transfer on the physical Go2 robot, with the policy navigating various real-world terrains including indoor floors, outdoor grass, and gravel paths. It serves as a practical, reproducible implementation of the modern quadruped RL pipeline that can be studied, modified, and extended.

## Core Contributions
- Complete, reproducible Isaac Lab pipeline for Go2 quadruped locomotion RL with documented configuration files
- Random-box terrain curriculum with configurable difficulty progression for robust terrain traversal
- Proven sim-to-real transfer on physical Go2 hardware across multiple terrain types
- Velocity-tracking task formulation with comprehensive reward shaping for stable, energy-efficient locomotion
- PPO training with domain randomization covering physics parameters, sensor noise, and external perturbations
- Detailed documentation and tutorials for training, evaluation, and deployment steps
- Modular code structure enabling easy adaptation to other quadruped platforms

## Methodology Deep-Dive
The environment is built in NVIDIA Isaac Lab with the Go2 URDF model. The simulation runs at 200 Hz physics with 50 Hz control decimation (4 physics steps per control step). Observations include: joint positions (12), joint velocities (12), projected gravity vector (3), base angular velocity (3), commanded velocities (3), and previous actions (12), totaling a 45-dimensional observation vector. Actions are desired joint position offsets relative to a default standing configuration, converted to joint torques via a PD controller (Kp = 40, Kd = 1.0).

The random-box terrain is generated procedurally. A terrain grid is divided into cells, and each cell is assigned a random height offset drawn from a uniform distribution. The curriculum controls the range of height offsets: starting from [−0.02, 0.02] meters (nearly flat) and progressing to [−0.10, 0.10] meters (challenging rough terrain). Progression is triggered when the agent achieves >80% of commanded velocity averaged over the last 100 episodes. Additionally, stairs and slopes are introduced at higher curriculum levels.

The reward function includes the following terms: (1) Linear velocity tracking: r_v = exp(−||v_cmd − v_actual||² / σ²) with σ = 0.25; (2) Angular velocity tracking: r_ω = exp(−||ω_cmd − ω_actual||² / σ²) with σ = 0.25; (3) Linear velocity z-penalty: −|v_z|² to discourage bouncing; (4) Angular velocity xy-penalty: −||ω_xy||² to keep the base level; (5) Joint torque penalty: −Σ|τ_i|² scaled by 1e-5; (6) Joint acceleration penalty: −Σ|a_i|² scaled by 2.5e-7; (7) Action rate penalty: −Σ|a_t − a_{t−1}|²; (8) Feet air time reward: encourages swing duration within target range for desired gait timing; (9) Collision penalty: negative reward for undesired body contacts (base, thighs).

Domain randomization applies the following perturbations at environment reset and during episodes: ground friction coefficient sampled from U(0.5, 1.25), base mass added from U(−1.0, 3.0) kg, motor strength factor from U(0.9, 1.1), external push forces applied randomly every 8–12 seconds with magnitude up to 15N, and Gaussian noise on joint position observations (σ = 0.01 rad) and joint velocity observations (σ = 1.5 rad/s).

PPO training uses 4096 parallel environments, a learning rate of 1e-3 with linear decay, γ = 0.99, λ = 0.95, clip ratio 0.2, 5 epochs per update, and minibatch size 24576. Training typically converges in 15–25 minutes on an RTX 3090 GPU. The policy network is an MLP with layers [256, 128, 64] and ELU activation. The value network shares the first two layers with the policy.

Sim-to-real transfer uses the trained policy exported as a PyTorch JIT model. On the Go2 hardware, a Python inference script loads the model and communicates with the robot's low-level controller via the Unitree SDK. The control loop runs at 50 Hz, matching the training control frequency.

## Key Results & Numbers
- Training convergence in 15–25 minutes on RTX 3090 with 4096 environments
- Velocity tracking RMSE: <0.08 m/s on flat terrain, <0.15 m/s on rough terrain (box height ±0.08m)
- Successful sim-to-real transfer on indoor tile, outdoor grass, gravel, and mild slopes
- Curriculum reaches maximum terrain difficulty within ~500 PPO iterations
- Domain randomization reduces sim-to-real velocity tracking error by ~40% compared to no randomization
- Policy robustness: recovers from push perturbations up to 20N in real-world tests
- Energy cost of transport (CoT): within 15% of manually tuned model-based controllers

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This framework is directly comparable to the Mini Cheetah RL pipeline. The Go2 and Mini Cheetah share the same morphological template (12 DoF quadruped) with similar mass ranges (~12 kg Go2 vs ~9 kg Mini Cheetah). The observation space, action space (joint position offsets + PD controller), reward function, and domain randomization strategy can be transferred to Mini Cheetah with minimal modifications. The specific numerical values for reward weights, PD gains, and domain randomization ranges provide empirically validated starting points.

The terrain curriculum design with random-box height progression is directly implementable in Mini Cheetah's MuJoCo environment. The velocity-tracking task formulation and the specific reward terms (velocity tracking, z-velocity penalty, torque penalty, action rate penalty, air time) constitute a well-tested reward template.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
While this framework is quadruped-focused, several design patterns transfer to Cassie's bipedal pipeline. The domain randomization methodology (parameter ranges, application schedule, noise models) is morphology-agnostic and directly applicable. The curriculum learning approach (terrain difficulty progression based on task competency) can be adapted for Cassie with bipedal-specific terrains and progression metrics. The sim-to-real deployment pipeline pattern (train → export → deploy with matched control frequency) provides a tested workflow template.

However, the flat PPO architecture and single-level control structure differ fundamentally from Cassie's 4-level hierarchical architecture, limiting the direct applicability of the policy design.

## What to Borrow / Implement
- Exact reward function formulation with exponential velocity tracking terms and specific σ values as starting configuration
- PD controller gains (Kp=40, Kd=1.0) as initial values for Mini Cheetah's joint-level PD control
- Domain randomization ranges (friction 0.5–1.25, mass ±1-3 kg, motor strength 0.9–1.1) calibrated for similar-mass quadrupeds
- Random-box terrain curriculum with height range progression and 80% velocity threshold for advancement
- Training hyperparameters (lr=1e-3 with linear decay, 4096 envs, 5 epochs) as validated starting point

## Limitations & Open Questions
- Single velocity-tracking task; no support for diverse skill repertoires or behavior switching
- MLP policy architecture without morphology-awareness; GNN-based policies could improve generalization
- No hierarchical control; all complexity must be handled by a single flat policy
- Go2-specific PD gains and reward weights may need significant re-tuning for Mini Cheetah's different actuator characteristics
