# Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

**Authors:** Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter
**Year:** 2022 | **Venue:** CoRL (Conference on Robot Learning)
**Links:** https://leggedrobotics.github.io/legged_gym/

---

## Abstract Summary
This paper introduces Legged Gym, an open-source framework built on NVIDIA Isaac Gym for training legged robot locomotion policies using massively parallel deep reinforcement learning. The key innovation is leveraging GPU-accelerated physics simulation to run thousands of robot environments simultaneously on a single GPU, reducing policy training time from hours or days to minutes. The framework demonstrates that a robust walking policy for the ANYmal quadruped can be trained in under 20 minutes on a single consumer GPU.

Legged Gym provides a complete pipeline from simulation to deployment: environment creation from URDF/MJCF descriptions, reward function specification, domain randomization, terrain generation, PPO training with the rsl_rl library, and policy export for real-robot deployment. The framework includes pre-configured environments for ANYmal-C, ANYmal-D, and other quadrupedal platforms, with modular components that allow researchers to customize any aspect of the training pipeline.

The paper became highly influential in the legged robotics community, establishing Legged Gym as the standard training framework for quadruped RL research. The open-source release democratized access to state-of-the-art legged robot training, enabling dozens of subsequent papers to build on the framework. The massively parallel training paradigm it popularized fundamentally changed the field's approach to policy training, making rapid iteration and large-scale hyperparameter sweeps practical.

## Core Contributions
- **Massively parallel training framework** that runs 4096+ robot environments simultaneously on a single GPU using Isaac Gym's tensor-based API
- **Minutes-scale training times** for robust locomotion policies, reducing iteration cycles from days to minutes
- **Open-source complete pipeline** from URDF import to real-robot deployment, including terrain generation, reward design, and domain randomization
- **Modular environment design** with configurable reward terms, observation spaces, action spaces, and terrain types
- **Reference implementation for PPO** on legged robots (rsl_rl library) optimized for GPU-parallel environments
- **Terrain curriculum** with procedurally generated heightfields of increasing difficulty (flat, slopes, stairs, rough terrain)
- **Demonstrated sim-to-real transfer** on ANYmal quadruped without additional real-world fine-tuning

## Methodology Deep-Dive
Legged Gym leverages Isaac Gym's GPU-accelerated physics simulation, which runs the entire simulation pipeline (collision detection, contact resolution, rigid body dynamics, tensor operations) on the GPU without CPU-GPU data transfer bottlenecks. Each training step processes 4096 environments in parallel: observations are collected as a single tensor of shape (4096, obs_dim), passed through the policy network (a 3-layer MLP with 256 units per layer by default), and actions are applied back to all environments simultaneously. This data-parallel approach achieves throughput of over 100,000 simulation steps per second on a single RTX 3090.

The observation space for the default quadruped configuration includes: base linear velocity (3), base angular velocity (3), projected gravity vector (3), velocity commands (3), joint positions relative to default (12), joint velocities (12), and previous actions (12), totaling 48 dimensions. The action space consists of 12 joint position targets, which are converted to torques via a PD controller with configurable gains. Actions are output at 50 Hz (every 0.02s simulation time), while the physics simulation runs at 200 Hz (0.005s timestep) with multiple substeps per policy step.

The reward function is a weighted sum of modular reward terms. Default terms include: linear velocity tracking (reward for matching commanded forward/lateral velocity, weight 1.0), angular velocity tracking (reward for matching yaw rate command, weight 0.5), linear velocity z-penalty (penalize vertical bouncing, weight -2.0), angular velocity xy-penalty (penalize roll/pitch oscillation, weight -0.05), orientation penalty (penalize deviation from upright, weight -0.0), torque penalty (minimize energy, weight -0.00001), joint acceleration penalty (encourage smooth motion, weight -2.5e-7), action rate penalty (penalize jerky actions, weight -0.01), collision penalty (penalize body/thigh ground contact, weight -1.0), and feet air time reward (encourage proper swing phase, weight 1.0). Each term can be independently weighted, and custom terms are easily added.

The terrain curriculum generates procedural heightfield terrains organized in a grid. Columns represent terrain types (flat, slopes, stairs up, stairs down, rough, discrete obstacles), and rows represent difficulty levels. Each robot is assigned to a terrain cell; when it successfully traverses the terrain (reaches a distance threshold), it advances to a harder row. Failed robots are moved to easier rows. This automated curriculum ensures robots are always training on appropriately challenging terrain.

Domain randomization covers: friction coefficients (0.5-1.25), restitution (0.0), added mass on base (minus 1 to 3 kg), center of mass displacement (plus or minus 0.15m), motor strength scaling (0.9-1.1), PD gain variation (plus or minus 10%), initial joint position noise, push disturbances (random force applied to base every 8-15 seconds), and observation noise on all sensor readings.

## Key Results & Numbers
- **Training time:** ANYmal walking policy converges in approximately 20 minutes on a single NVIDIA RTX 3090 (4096 environments)
- **Throughput:** Over 100,000 simulation steps per second at 4096 parallel environments
- **Environment scaling:** Linear scaling from 512 to 8192 environments; 4096 provides the best time-to-convergence
- **PPO iterations:** Converges in approximately 1000-2000 PPO iterations (20M-40M total environment steps)
- **Terrain curriculum:** Robots learn to traverse stairs (step height up to 0.25m), slopes (up to 30 degrees), and rough terrain (0.1m roughness) within the 20-minute training window
- **Sim-to-real:** Trained policies deploy zero-shot on ANYmal-C and ANYmal-D with robust walking across indoor environments
- **Community adoption:** 1500+ GitHub stars, used as the base framework in 50+ published papers on legged locomotion

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Legged Gym is the reference framework for Project A's training pipeline. The Mini Cheetah training environment should be built either directly in Legged Gym (by adding a Mini Cheetah URDF/MJCF configuration) or in a framework modeled closely after it. Key elements to adopt include: the modular reward function design (allowing systematic ablation of reward terms), the terrain curriculum (critical for progressive difficulty in locomotion training), the domain randomization parameter ranges (calibrated for quadrupedal robots), and the PPO training configuration (learning rate schedules, GAE parameters, etc.).

The minutes-scale training time enables the rapid iteration cycle that is essential for Project A: testing different reward formulations, domain randomization ranges, and curriculum schedules in a single afternoon rather than waiting days per experiment. The transition from Isaac Gym to MuJoCo for Project A (if using MuJoCo) requires attention to physics parameter differences, but the overall framework design translates directly. The rsl_rl PPO implementation provides a reference for the RL algorithm configuration.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
Legged Gym's framework concepts are applicable to Cassie training, though bipedal locomotion requires different reward design, observation spaces, and terrain challenges. The terrain curriculum and domain randomization approaches transfer directly. However, Cassie's dynamics (underactuated, compliant legs, single-support phases) require significant modifications to the reward function and training procedure.

Specific adaptations needed for Cassie include: (1) different observation space including spring deflections and four-bar linkage states, (2) bipedal-specific reward terms for symmetry, step width, and zero-moment point, (3) different terrain curriculum focused on flat ground stability before advancing to slopes and stairs, and (4) longer training times due to the difficulty of bipedal balance (likely hours rather than minutes). The massively parallel training approach remains valuable for Cassie's multi-level hierarchy, where each level's policy can be trained rapidly.

## What to Borrow / Implement
- **Complete framework architecture:** adopt Legged Gym's modular design for environment, reward, terrain, domain randomization, and training configuration
- **Terrain curriculum system:** implement the row/column terrain grid with automatic difficulty progression based on agent performance
- **Modular reward function:** use the weighted-sum-of-terms design for systematic reward engineering and ablation studies
- **Domain randomization ranges:** start with Legged Gym's published ranges and calibrate for Mini Cheetah and Cassie hardware
- **PPO hyperparameters:** use rsl_rl's default settings (learning rate 1e-3 with decay, GAE lambda 0.95, clip ratio 0.2, minibatch size 4096) as starting points

## Limitations & Open Questions
- **Isaac Gym dependency:** the framework is tightly coupled to NVIDIA Isaac Gym, which requires NVIDIA GPUs and has limited macOS/ARM support; migration to MuJoCo MJX or Brax requires significant effort
- **Quadruped-centric:** default environments and reward terms are designed for quadrupedal robots; bipedal configurations require substantial customization
- **Flat MLP policy only:** the default policy architecture is a simple 3-layer MLP; more complex architectures (transformers, graph networks, recurrent policies) require custom implementation
- **No perception integration:** the framework handles proprioception-only policies; adding camera-based perception or elevation maps requires additional modules not included in the base framework
