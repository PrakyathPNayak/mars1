# Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning

**Authors:** Shi et al.
**Year:** 2023 | **Venue:** arXiv
**Links:** [arXiv:2310.04675](https://arxiv.org/abs/2310.04675)

---

## Abstract Summary
Shi et al. present a deep reinforcement learning framework for terrain-aware quadruped locomotion that fuses proprioceptive and exteroceptive sensor inputs to enable robust outdoor navigation. The system uses a deep neural network controller that processes joint encoders, IMU data, and terrain perception (depth cameras or lidar-derived heightmaps) to dynamically adapt gait parameters—step height, stride length, body posture, and foot placement—based on the perceived terrain ahead.

The key innovation is a terrain encoding module that compresses high-dimensional exteroceptive observations into a compact latent representation suitable for real-time policy inference. This terrain embedding is concatenated with proprioceptive features and fed into a locomotion policy that outputs target joint positions. The approach handles a wide variety of challenging terrains including stairs (ascending/descending), stepping stones, gaps exceeding 25.5 cm, slopes, and rubble fields.

Validation spans both extensive simulation testing in Isaac Gym and real-world deployment on quadruped hardware. The real-world experiments demonstrate successful traversal of outdoor environments with terrain transitions (e.g., flat → stairs → rubble) without requiring manual gait switching or terrain-specific policy selection. The unified policy handles all terrains through a single network conditioned on terrain perception.

## Core Contributions
- Unified RL controller fusing proprioceptive and exteroceptive inputs for terrain-adaptive quadruped locomotion
- Compact terrain encoding module that compresses depth/heightmap data into low-dimensional latent features
- Demonstration of gait adaptation to stairs, stepping stones, gaps (>25.5 cm), slopes, and rubble
- Validated in both simulation (Isaac Gym) and real-world outdoor environments
- Single policy handles diverse terrains without terrain-specific modules or manual gait switching
- Curriculum learning strategy that progressively increases terrain difficulty during training
- Analysis of which exteroceptive features contribute most to terrain adaptation

## Methodology Deep-Dive
The system architecture consists of three modules: a proprioceptive encoder, a terrain encoder, and a locomotion policy. The proprioceptive encoder processes a 48-dimensional observation vector: 12 joint positions, 12 joint velocities, body orientation (roll, pitch, yaw), body angular velocity (3D), gravity vector in body frame (3D), commanded velocity (3D, forward/lateral/turning), and previous action (12D). This vector is encoded through a 2-layer MLP (128–64 units) into a 32-dimensional proprioceptive feature vector.

The terrain encoder receives a heightmap scan of the terrain ahead of the robot, represented as a 21×11 grid (231 points) sampled at 5 cm resolution in the robot's local frame, extending 1.0 m forward and 0.5 m laterally. This 231-dimensional heightmap is processed by a 3-layer convolutional network (1D convolutions over the scan rays) followed by a fully-connected layer, producing a 32-dimensional terrain embedding. The terrain encoder is trained end-to-end with the policy but uses a separate learning rate (1e-4 vs. 3e-4 for the policy) to stabilize learning.

The locomotion policy concatenates the proprioceptive feature (32D) and terrain embedding (32D) into a 64-dimensional state representation, which is processed by a 2-layer MLP (256–128 units) with ELU activations to produce 12 target joint positions. A PD controller at 200 Hz converts targets to torques. The policy operates at 50 Hz, with the terrain encoder updating at the same rate (or lower, 10–25 Hz, with zero-order hold).

Training uses PPO in Isaac Gym with 4096 parallel environments. The reward function includes: velocity tracking (||v - v_cmd||²), orientation stability (||roll, pitch||²), foot clearance (encouraging sufficient step height over obstacles), base height maintenance, action smoothness, joint velocity penalty, torque penalty, and survival bonus. A curriculum over terrain difficulty starts with flat ground and progressively introduces rougher terrain, steeper slopes, and taller stairs over 500M environment steps (~4 hours on 8 GPUs).

A key training technique is the privileged information framework: during training, the terrain encoder receives perfect heightmap data, while during deployment, the heightmap is estimated from onboard depth cameras using a separate perception pipeline. A distillation step transfers the privileged-information policy to a deployable policy that operates on noisy depth observations via a teacher-student framework.

## Key Results & Numbers
- Successful traversal of stairs up to 20 cm height (ascending and descending)
- Gap crossing capability exceeding 25.5 cm (approximately 1.5× hip width)
- Slope traversal up to 30° inclination without slipping
- Stepping stone navigation with 15 cm spacing between viable footholds
- Real-world deployment success rate: >85% on mixed outdoor terrains
- Training time: ~4 hours on 8× NVIDIA A100 GPUs with 4096 parallel environments
- Policy inference at 50 Hz with total latency <5 ms on embedded NVIDIA Jetson
- Terrain encoder contributes 15–35% performance improvement over proprioceptive-only baselines

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper provides a complete recipe for extending the Mini Cheetah from flat-terrain locomotion to outdoor terrain navigation. The proprioceptive observation space (48D) closely matches the Mini Cheetah's 12-DoF configuration, and the PPO-based training in Isaac Gym aligns with the project's existing methodology. The terrain encoding module can be integrated on top of the Mini Cheetah's base locomotion policy.

The curriculum learning strategy—starting from flat ground and progressively introducing terrain challenges—is directly applicable to the Mini Cheetah's training pipeline. The teacher-student distillation for handling noisy depth observations in deployment is essential for any real-world terrain-aware deployment. The 50 Hz policy rate and 200 Hz PD control rate are compatible with the Mini Cheetah's hardware capabilities.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The terrain encoding approach is directly relevant to Cassie's Contrastive Predictive Terrain Encoder (CPTE). Shi et al.'s heightmap-to-latent encoding validates the general approach of compressing terrain information into compact representations, and the specific architecture choices (1D convolutions over scan rays, 32D latent space) provide a baseline for CPTE design. The privileged-information training framework can be adopted for Cassie's terrain module.

The exteroceptive fusion methodology—concatenating terrain embeddings with proprioceptive features—maps to Cassie's mid-level Controller, which must integrate terrain awareness for foothold planning. The teacher-student distillation pipeline is applicable to transferring Cassie's privileged-information policies to deployable versions.

## What to Borrow / Implement
- Adopt the 21×11 heightmap representation at 5 cm resolution as terrain input for both platforms
- Implement the privileged-information → teacher-student distillation pipeline for sim-to-real terrain adaptation
- Use the terrain difficulty curriculum (flat → rough → stairs → gaps) for progressive training
- Apply the dual learning rate strategy (separate rates for terrain encoder vs. policy) for stable training
- Integrate the terrain encoding module architecture as a baseline for Cassie's CPTE

## Limitations & Open Questions
- Heightmap representation assumes rigid terrain; deformable surfaces (mud, sand, snow) not addressed
- Terrain perception pipeline (depth → heightmap) introduces latency and noise not fully characterized
- Single-policy approach may sacrifice peak performance on specific terrains compared to terrain-specialized policies
- Gap crossing limited to ~25.5 cm; larger gaps may require fundamentally different strategies (jumping)
