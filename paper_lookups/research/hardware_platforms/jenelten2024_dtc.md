# DTC: Deep Tracking Control

**Authors:** Fabian Jenelten, Junzhe He, Farbod Farshidian, Marco Hutter
**Year:** 2024 | **Venue:** Science Robotics
**Links:** Science Robotics (2024)

---

## Abstract Summary
Deep Tracking Control (DTC) presents a powerful fusion of trajectory optimization and reinforcement learning for robust quadruped locomotion on challenging terrain. The core idea is to use a model-based trajectory optimizer to generate reference motions that encode physically feasible and efficient locomotion patterns, and then train an RL policy to track these references while adapting to terrain variations, model inaccuracies, and external disturbances. This approach combines the physical grounding of trajectory optimization with the adaptability and robustness of learned controllers.

The system is developed and deployed on the ANYmal quadruped robot, demonstrating remarkable capabilities on terrains that defeat both pure model-based and pure learning-based controllers. These include sparse stepping stones requiring precise foot placement, slippery surfaces where friction is severely limited, and deformable terrain where ground compliance varies unpredictably. DTC outperforms pure MPC solutions (which cannot adapt to unmodeled terrain properties) and pure RL solutions (which struggle with the precise foot placement required for stepping stones).

The key insight is that trajectory optimization provides a structured prior for the RL policy, dramatically reducing the exploration burden. Instead of discovering locomotion from scratch, the RL policy starts with a physically meaningful reference and learns when and how to deviate from it. This results in faster training, more natural motions, and better generalization to unseen terrain conditions. The paper represents a significant advance in the field of legged locomotion, published in the prestigious Science Robotics journal.

## Core Contributions
- Fusion of trajectory optimization and RL where the optimizer provides structured reference motions for RL policy tracking
- Demonstration on ANYmal robot traversing sparse stepping stones with precise foot placement
- Robust locomotion on slippery surfaces with severely reduced friction coefficients
- Successful navigation of deformable terrain with varying ground compliance
- Outperforms both pure MPC and pure RL baselines on challenging terrain benchmarks
- Efficient training by leveraging trajectory optimization as a structured exploration prior
- Real-world deployment on ANYmal with extensive hardware experiments

## Methodology Deep-Dive
DTC's architecture consists of three main components: a model-based trajectory optimizer, an RL tracking policy, and a terrain perception module. The trajectory optimizer uses a centroidal dynamics model with pre-defined gait patterns to generate reference trajectories. It solves a nonlinear optimization problem that minimizes a cost function including CoM tracking error, foot placement accuracy, energy consumption, and smoothness, subject to dynamics constraints, friction cone constraints, and kinematic limits.

The trajectory optimizer operates at a lower frequency (10-50Hz) and provides reference signals to the RL policy: desired foot trajectories, CoM trajectory, joint position references, and contact schedule. These references are physically feasible by construction (satisfying dynamics and constraints) but may not account for terrain variations, model errors, or disturbances that the real robot encounters.

The RL tracking policy is a neural network (MLP with 3 hidden layers of 256 units each) trained with PPO in Isaac Gym simulation. Its observation space includes proprioceptive state (joint positions, velocities, body orientation, angular velocity), the current reference trajectory values from the optimizer (desired foot positions, CoM position, contact schedule), and a terrain encoding from the perception module. The action space is target joint positions fed to a PD controller. The reward function encourages tracking the reference trajectory while allowing deviations when necessary for stability.

The reward design is critical: r = w_track * r_track + w_alive * r_alive + w_style * r_style + w_terrain * r_terrain. The tracking reward r_track measures deviation from the optimizer's reference (joint positions, foot positions, body orientation). The alive reward encourages survival. The style reward penalizes unnatural motions (excessive torques, large joint velocities, contact timing errors). The terrain reward provides bonuses for successfully navigating difficult terrain features.

The terrain perception module processes height map information around the robot into a compact terrain encoding. During training, this encoding is generated from the ground-truth simulation height field. For deployment, it is generated from onboard depth cameras or lidar using a learned encoder. The terrain encoding informs both the trajectory optimizer (for foot placement planning) and the RL policy (for adaptive behavior).

Domain randomization during RL training includes friction coefficients (0.2 to 1.5), ground compliance (rigid to soft), body mass perturbations (plus/minus 20%), motor strength variations (80-120%), sensor noise, and communication delays (0 to 20ms). A curriculum over terrain difficulty progressively introduces more challenging stepping stone configurations, lower friction surfaces, and softer ground.

## Key Results & Numbers
- ANYmal successfully traverses stepping stones with 15cm gaps between footholds
- Stable locomotion on surfaces with friction coefficient as low as 0.15
- Successful navigation of deformable terrain with stiffness variations of 5x
- 40% higher success rate on stepping stones compared to pure RL baseline
- 60% higher success rate on slippery terrain compared to pure MPC baseline
- Training converges in approximately 2 hours on a single GPU with Isaac Gym
- Real-world deployment validated with extensive outdoor experiments on ANYmal C and D platforms

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The DTC paradigm is directly applicable to Mini Cheetah for challenging terrain locomotion. Instead of training an end-to-end RL policy from scratch, a trajectory optimizer could provide physically feasible reference gaits that the RL policy learns to track and adapt. This would be particularly valuable for terrain types that require precise foot placement, such as stepping stones or narrow beams, where pure RL may struggle to discover appropriate behaviors through random exploration.

For the Mini Cheetah project, DTC offers a concrete path to handling difficult terrain: use MuJoCo's trajectory optimization capabilities to generate reference motions for various gait patterns, then train the PPO policy to track these references with domain randomization. The reference tracking formulation would also simplify reward design—instead of hand-crafting complex reward functions for each terrain type, the rewards center on tracking quality with terrain-adaptive deviations.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The trajectory optimization + RL fusion concept is relevant to Cassie's hierarchical controller design, particularly at the interface between the Planner and Primitives levels. The Planner could generate reference trajectories using model-based optimization (similar to DTC's trajectory optimizer), and the Primitives/Controller levels could learn to track these references while adapting to terrain and disturbances. This would provide a principled way to decompose the control hierarchy.

However, bipedal locomotion on Cassie presents additional challenges not addressed by DTC: the underactuated ankle dynamics, the passive spring-damper elements, and the fundamentally different balance requirements. The terrain perception module design and the domain randomization strategy are more directly transferable.

## What to Borrow / Implement
- Adopt the trajectory optimization + RL tracking paradigm for Mini Cheetah's challenging terrain locomotion
- Use the reward structure (tracking + alive + style + terrain) as a template for both Mini Cheetah and Cassie RL training
- Implement the progressive terrain curriculum (stepping stones, low friction, deformable ground) for domain randomization
- Apply the terrain encoding approach (compact representation from height field) for both projects' terrain perception
- Consider the reference trajectory deviation analysis for understanding when and why the RL policy deviates from the model-based reference

## Limitations & Open Questions
- The trajectory optimizer requires a dynamics model that may not be available or accurate for all robot platforms
- The approach is demonstrated primarily for quadruped locomotion; extension to bipedal robots with underactuation requires further research
- Real-time trajectory optimization at 10-50Hz may be computationally challenging on embedded hardware
- The terrain perception module's accuracy on real-world data (vs. perfect simulation height maps) significantly impacts performance
