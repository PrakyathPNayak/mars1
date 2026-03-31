# Legged Locomotion in Challenging Terrains using Egocentric Vision

**Authors:** Ananye Agarwal, Ashish Kumar, Jitendra Malik, Deepak Pathak
**Year:** 2023 | **Venue:** CoRL 2023
**Links:** https://vision-locomotion.github.io/

---

## Abstract Summary
An end-to-end RL system enables a quadruped robot to traverse diverse obstacles—stairs, stepping stones, and gaps—using only a single egocentric depth camera. The policy is trained in simulation with a privileged teacher and then distilled to a student network that relies solely on depth images and proprioception. A key insight is the need for temporal memory of prior observations to infer terrain features beneath the robot's body that are no longer visible.

## Core Contributions
- End-to-end vision-based locomotion policy using a single egocentric depth camera
- Privileged teacher–student distillation framework for sim-to-real transfer
- Temporal memory mechanism to handle partial observability of terrain under the robot
- Zero-shot sim-to-real transfer across diverse challenging terrains
- Demonstrates robustness to perturbations and unseen obstacles without any fine-tuning
- Unified single policy handles stairs, gaps, stepping stones, and rough terrain

## Methodology Deep-Dive
The system follows a two-phase training pipeline. In the first phase, a privileged teacher policy is trained in simulation with access to ground-truth terrain information (heightmaps, contact states, and full environment geometry). This teacher uses PPO and receives a rich observation vector including terrain scans that would be unavailable on the real robot. The teacher learns robust locomotion behaviors across randomized terrains.

In the second phase, a student policy is trained via DAgger-style distillation. The student only receives egocentric depth images and proprioceptive data (joint positions, velocities, IMU). To handle the key challenge—that terrain directly beneath and behind the robot is occluded from the forward-facing depth camera—the student network incorporates a recurrent memory module (GRU or similar). This memory aggregates information from past depth frames, effectively building an implicit representation of recently observed terrain.

The simulation environment is built with extensive domain randomization: terrain geometry, friction coefficients, depth camera noise, robot dynamics parameters, and latency are all randomized. This ensures the learned policy generalizes to real-world conditions without fine-tuning.

The depth image processing pipeline uses a compact CNN encoder that extracts spatial features from the depth image, which are then concatenated with proprioceptive features and fed through the recurrent memory and MLP policy head. The entire student network is lightweight enough to run in real-time on onboard compute.

Deployment on the real robot demonstrates zero-shot transfer: the policy trained entirely in simulation successfully navigates stairs of varying heights, stepping stones with gaps, and rough outdoor terrain. The system recovers from pushes and unexpected obstacles without any adaptation or calibration.

## Key Results & Numbers
- Traverses stairs (up to 20 cm height), stepping stones, and gaps up to 30 cm
- Single egocentric depth camera (no LiDAR, no external sensing)
- Zero-shot sim-to-real transfer with no real-world fine-tuning
- Robust to external perturbations (pushes, unexpected obstacles)
- Outperforms proprioception-only baselines on all challenging terrains
- Memory-based student outperforms memoryless student by significant margin on occluded terrain features
- Real-time inference at control frequency on onboard compute

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to equipping the Mini Cheetah with vision-based locomotion. The egocentric depth camera approach maps naturally to adding a front-facing depth sensor to Mini Cheetah. The teacher-student distillation framework is compatible with the existing PPO training pipeline—the teacher can leverage MuJoCo's ground-truth heightfield data while the student learns from simulated depth only. The memory mechanism addresses a critical gap in proprioception-only policies: anticipating upcoming terrain changes. The domain randomization strategy (friction, dynamics, camera noise) complements the existing randomization for the 12-DoF system with PD control at 500 Hz.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The egocentric depth processing pipeline could serve as a front-end for Project B's CPTE (Contrastive Pretrained Terrain Encoder). Rather than raw depth or hand-crafted features, the learned depth encoder from this work provides a compact terrain representation suitable for the Dual Asymmetric-Context Transformer. The teacher-student distillation approach mirrors the asymmetric training already used in Project B, where the teacher has privileged terrain information. The temporal memory mechanism is complementary to the RSSM/Dreamer world model—it provides a geometric memory of terrain while RSSM captures dynamics. The approach could feed into the Planner level of the 4-level hierarchy for terrain-aware path selection.

## What to Borrow / Implement
- Adopt the depth CNN encoder architecture as input preprocessing for CPTE terrain encoder
- Implement the privileged teacher → student distillation pipeline alongside existing PPO training
- Add GRU-based temporal memory for depth observations to handle terrain occlusion
- Use the domain randomization protocol for depth camera simulation (noise, latency, field-of-view)
- Integrate depth-based terrain anticipation into the Planner level of Project B's hierarchy

## Limitations & Open Questions
- Single forward-facing camera has limited field of view; side obstacles may be missed
- Memory module may struggle with very long temporal dependencies or rapid direction changes
- Depth camera performance degrades in bright sunlight and with reflective/transparent surfaces
- No explicit foothold planning—the policy implicitly learns foot placement
- Computational cost of depth processing may constrain control frequency on resource-limited hardware
- How well does the temporal memory scale to more complex 3D environments with dynamic obstacles?
