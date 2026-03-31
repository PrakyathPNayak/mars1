# Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning

**Authors:** Various
**Year:** 2023 | **Venue:** arXiv 2023
**Links:** https://arxiv.org/abs/2310.04675

---

## Abstract Summary
This paper develops RL-based terrain-aware locomotion where the policy adapts foot height, step frequency, and gait parameters based on terrain features. The system uses terrain heightmaps and proprioceptive feedback to modulate locomotion in real-time. It was validated on stepping stones and rough terrain in real-world experiments.

## Core Contributions
- Develops a terrain-aware RL policy that dynamically adjusts gait parameters based on perceived terrain features
- Demonstrates adaptive foot height modulation for stepping stone traversal — the robot lifts feet higher for obstacles and lower for flat terrain
- Integrates terrain heightmap information directly into the policy observation space for real-time adaptation
- Validates on real quadruped hardware across stepping stones, rough terrain, and mixed surfaces
- Shows that explicit terrain encoding outperforms end-to-end learning without structured terrain features
- Achieves robust locomotion across terrain types with a single unified policy rather than terrain-specific policies

## Methodology Deep-Dive
The approach centers on providing the RL policy with structured terrain information alongside standard proprioceptive observations. The terrain heightmap is sampled around each foot at a grid of points, providing local elevation information that the policy uses to plan foot placements. This terrain encoding is combined with the robot's joint positions, velocities, body orientation, and previous actions to form the full observation vector.

The policy architecture uses an MLP that processes the combined observation and outputs joint position targets for PD controllers. A key design choice is the separation of terrain features and proprioceptive features in early network layers, which are then fused in later layers. This architectural inductive bias helps the network learn to associate terrain features with appropriate gait modifications.

Training occurs in simulation with procedurally generated terrain. The terrain generator creates heightmaps with controlled roughness, step heights, gaps, and slopes. Curriculum learning progressively increases terrain difficulty — starting with nearly flat terrain and gradually introducing stepping stones and rough surfaces. Domain randomization is applied to terrain properties (friction, restitution), robot dynamics (mass, inertia), and sensor noise to facilitate sim-to-real transfer.

The gait adaptation emerges naturally from the reward structure without explicit gait specification. The reward encourages forward velocity, penalizes large joint torques and velocities, and includes a foot clearance bonus that rewards lifting feet above terrain obstacles. The policy learns to increase step height on rough terrain, decrease step frequency on challenging surfaces, and maintain a stable trot on flat ground — all from a single reward function.

Real-world experiments deploy the trained policy on a quadruped with onboard depth cameras for terrain perception. The heightmap is computed from depth images in real-time and fed to the policy at each control step. The robot successfully traverses stepping stones with 15cm gaps, rough terrain with 8cm height variations, and transitions between different terrain types without manual intervention.

## Key Results & Numbers
- Successful stepping stone traversal with gaps up to 15cm and height variations up to 8cm
- Real-world deployment with >90% success rate across diverse terrain types
- Terrain-aware policy outperforms terrain-blind baselines by 35-45% in success rate on challenging terrain
- Adaptive foot clearance ranges from 2cm (flat terrain) to 12cm (rough terrain) automatically
- Step frequency modulation: 2.5 Hz on flat terrain, reducing to 1.8 Hz on difficult terrain
- Single policy covers all terrain types — no mode switching needed
- Sim-to-real transfer with <10% performance degradation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to Mini Cheetah's outdoor deployment. The terrain heightmap encoding, adaptive foot clearance, and gait modulation can be directly implemented in Mini Cheetah's MuJoCo training pipeline. The 12-DoF PD control at 500 Hz matches the paper's control architecture. The curriculum learning approach for terrain difficulty aligns with Mini Cheetah's training methodology. The sim-to-real transfer results on similar-scale quadrupeds provide confidence that the approach will transfer to Mini Cheetah hardware.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Terrain features from this paper directly inform the CPTE (Contrastive Terrain Encoder) module in Project B. The heightmap sampling strategy around foot locations provides a structured input for terrain encoding. The adaptive gait parameters (foot height, step frequency) are relevant to the Controller level of Cassie's hierarchy, where the Neural ODE Gait Phase module modulates gait timing. The terrain-aware observation design can be adapted for bipedal locomotion, feeding into both the Planner and Controller levels of the hierarchy.

## What to Borrow / Implement
- Implement the heightmap sampling strategy around foot positions for both projects
- Adopt the separated terrain/proprioceptive feature processing architecture
- Use the procedural terrain generation pipeline for MuJoCo training environments
- Apply the curriculum learning schedule for progressive terrain difficulty
- Integrate the foot clearance reward term into Mini Cheetah's reward function
- Adapt the terrain encoding for Cassie's CPTE module — use heightmap features as input to the contrastive encoder
- Use the sim-to-real transfer protocol (domain randomization + terrain randomization) as a baseline

## Limitations & Open Questions
- Heightmap computation from depth images introduces latency and noise not present in simulation
- The approach assumes relatively static terrain — dynamic obstacles are not addressed
- Stepping stone traversal requires precise terrain perception; errors in heightmap estimation cause failures
- Bipedal adaptation is non-trivial — the gait modulation strategies may not transfer from quadruped to biped
- The reward function requires careful tuning of the foot clearance bonus weight
- Performance degrades significantly in low-visibility conditions where depth cameras fail
