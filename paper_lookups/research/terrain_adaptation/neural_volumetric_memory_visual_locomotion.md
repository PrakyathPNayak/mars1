# Neural Volumetric Memory for Visual Locomotion Control

**Authors:** Ruihan Yang et al.
**Year:** 2023 | **Venue:** CVPR 2023
**Links:** https://rchalyang.github.io/NVM

---

## Abstract Summary
This paper proposes a geometric memory architecture that stores SE(3)-aligned 3D feature volumes from egocentric depth images accumulated over time. This neural volumetric memory enables effective aggregation of visual information to handle partial observability in legged locomotion, where the robot cannot see beneath its body. The approach is validated on a real quadruped robot traversing stairs and obstacles.

## Core Contributions
- SE(3)-equivariant 3D volumetric memory architecture for visual locomotion
- Geometric aggregation of depth observations over time into a body-centric 3D feature volume
- Principled handling of partial observability through spatial memory rather than flat temporal memory
- Real robot deployment demonstrating stair climbing and obstacle traversal
- Outperforms flat memory baselines (GRU, frame stacking) on challenging terrain
- Efficient voxel-based representation enabling real-time inference

## Methodology Deep-Dive
The core innovation is representing the robot's terrain memory as a 3D voxel grid in the robot's body frame, rather than using flat temporal memory (e.g., GRU over past observations). At each timestep, the egocentric depth image is back-projected into 3D space and encoded into a feature volume. This volume is then transformed into the robot's current body frame using the known SE(3) pose transformation (from IMU and odometry), and fused with the existing volumetric memory via learned aggregation operations.

The SE(3) alignment is critical: as the robot moves, past observations are properly registered in the current frame of reference. This means terrain features observed several steps ago remain geometrically accurate in the current body-centric coordinate system. This is fundamentally more principled than flat temporal memory, which must implicitly learn to de-rotate and de-translate past observations.

The volumetric memory is implemented as a 3D convolutional neural network operating on a discretized voxel grid around the robot. The grid resolution and spatial extent are chosen to cover the robot's immediate locomotion-relevant area (roughly 1-2 meters in each direction). Features from the current depth observation are integrated into this grid using differentiable voxel operations, allowing end-to-end training with RL.

The locomotion policy reads from this volumetric memory along with proprioceptive state to produce joint actions. Training follows the standard privileged teacher–student paradigm: the teacher has access to ground-truth heightmaps, while the student must build its terrain understanding from the volumetric memory. The entire pipeline is trained in simulation with domain randomization and deployed zero-shot on real hardware.

Experiments compare against several baselines: proprioception-only, frame-stacking (concatenating recent depth frames), GRU-based temporal memory, and the proposed volumetric memory. The volumetric approach consistently outperforms alternatives, especially in scenarios requiring memory of terrain behind or beneath the robot.

## Key Results & Numbers
- 3D volumetric memory outperforms GRU-based flat memory on stair traversal and obstacle courses
- Real robot deployment on A1 quadruped with zero-shot sim-to-real transfer
- Handles terrain occlusion more effectively than frame-stacking approaches
- SE(3) alignment preserves geometric accuracy across multiple seconds of locomotion
- Real-time inference achieved with efficient voxel operations
- Significant improvement on terrains requiring multi-step planning (tall stairs, gaps)
- Ablation studies confirm importance of both 3D structure and SE(3) alignment

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The neural volumetric memory is directly applicable to Mini Cheetah with a depth camera. The 3D voxel representation provides a more principled terrain representation than simple observation stacking. For the MuJoCo simulation, the depth camera can be simulated and the volumetric memory trained alongside the PPO policy. The SE(3) alignment leverages the IMU data already available on Mini Cheetah. The body-centric voxel grid naturally scales to the 12-DoF system's workspace. Key consideration: computational cost of 3D convolutions must fit within the 500 Hz control loop—likely the memory update runs at a lower frequency (e.g., 30 Hz camera rate) while the low-level PD controller runs at 500 Hz.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The volumetric terrain representation could significantly enhance the CPTE (Contrastive Pretrained Terrain Encoder) in Project B. Instead of encoding terrain as a flat embedding, CPTE could incorporate 3D geometric structure through a volumetric memory. This 3D representation is particularly valuable for bipedal locomotion where precise foot placement matters. The volumetric memory could serve as input to the MC-GAT (GATv2 on kinematic tree), providing each joint node with spatially localized terrain features from the voxel grid. The SE(3) alignment is complementary to the Neural ODE Gait Phase module, as both maintain geometric consistency. The volumetric representation could also benefit the Differentiable Capture Point computation by providing accurate 3D terrain geometry for stability estimation.

## What to Borrow / Implement
- Implement SE(3)-aligned volumetric memory as a terrain representation module for both projects
- Use the 3D feature volume as input to CPTE instead of or alongside flat terrain embeddings
- Adopt the voxel-based depth fusion pipeline for integrating temporal depth observations
- Consider a two-frequency architecture: volumetric memory at camera rate, control policy at 500 Hz
- Use the geometric aggregation approach to provide per-joint terrain context to MC-GAT nodes
- Implement differentiable voxel operations for end-to-end training compatibility with PPO

## Limitations & Open Questions
- 3D convolutions are computationally expensive; may be challenging for real-time deployment on limited onboard compute
- Voxel grid resolution trades off between spatial precision and memory/compute requirements
- Relies on accurate SE(3) pose estimation; drift in odometry degrades memory alignment over time
- Fixed-size voxel grid limits the spatial extent of memory; very far terrain is discarded
- How well does the volumetric memory handle dynamic obstacles that move between observations?
- Integration with world models (RSSM/Dreamer) for predictive terrain estimation remains unexplored
