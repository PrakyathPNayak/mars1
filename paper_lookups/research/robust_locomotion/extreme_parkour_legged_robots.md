# Extreme Parkour with Legged Robots

**Authors:** Xuxin Cheng, Kexin Shi, Ananye Agarwal, Deepak Pathak
**Year:** 2024 | **Venue:** ICRA 2024
**Links:** https://arxiv.org/abs/2309.14341

---

## Abstract Summary
Demonstrates extreme parkour capabilities (high jumps, long jumps, handstands) on a Unitree A1 quadruped using a teacher-student framework. The teacher is trained with privileged environment information, while the student is distilled to use only proprioception and egocentric depth. Achieves dynamic maneuvers previously thought impossible for small quadrupeds.

## Core Contributions
- Achieves extreme parkour maneuvers (high jumps, long jumps, handstands) on small-scale quadruped hardware
- Develops a teacher-student distillation framework where the teacher has access to privileged state information
- Student policy operates solely on proprioception and egocentric depth images
- Demonstrates that small quadrupeds can achieve agility levels comparable to animals
- Validates robust real-world deployment with perturbation recovery
- Provides benchmarks for dynamic maneuver capabilities on the Unitree A1 platform
- Shows that privileged information training significantly accelerates learning of extreme behaviors

## Methodology Deep-Dive
The core of the approach is an asymmetric teacher-student training paradigm. The teacher policy is trained in simulation with full access to privileged information including exact terrain geometry, ground-truth contact states, and full rigid-body dynamics state. This privileged access allows the teacher to efficiently learn extreme maneuvers without the perception bottleneck, focusing purely on motor skill acquisition.

The teacher is trained using reinforcement learning (PPO) with carefully shaped reward functions that encourage specific parkour behaviors. Reward terms include forward velocity, height targets for jumps, orientation targets for handstands, and energy penalties for efficiency. The privileged teacher can rapidly discover and refine extreme behaviors because it knows the exact environment state.

The student policy is then distilled from the teacher using supervised learning (behavior cloning with DAgger-style corrections). The student receives only proprioceptive readings (joint positions, velocities, body IMU) and egocentric depth images from a forward-facing camera. The distillation process forces the student to infer the relevant environment information from its limited sensory input, effectively learning an implicit terrain representation.

Domain randomization is applied extensively during both teacher training and student distillation to ensure robustness. Physical parameters (mass, friction, motor strength), sensor noise models, and terrain variations are all randomized. This ensures the student policy generalizes across the sim-to-real gap and handles real-world perturbations.

The system achieves real-time inference on the robot's onboard compute, with the depth processing network and policy network running within the control loop. The depth images are processed through a lightweight CNN encoder that produces a compact latent representation fed into the policy MLP alongside proprioceptive features.

## Key Results & Numbers
- Achieves jumps spanning 2× the robot's body length
- High jumps exceeding the robot's own height
- Successful handstand maneuvers on real hardware
- Real-world deployment on Unitree A1 with zero-shot sim-to-real transfer
- Robust to external perturbations (pushes, terrain variations) during parkour execution
- Single policy handles multiple extreme maneuvers without manual switching

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper sets the benchmark for extreme agility on small quadrupeds, directly relevant to pushing Mini Cheetah's capabilities. The teacher-student distillation paradigm is directly implementable: train a privileged teacher with full MuJoCo state access, then distill to a deployable student using only proprioception (and optionally depth). The reward shaping strategies for extreme behaviors provide templates for Mini Cheetah's PPO training. The demonstrated sim-to-real transfer validates the approach for the 12 DoF, PD control at 500 Hz setup.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
The teacher-student framework maps directly to asymmetric training paradigms used throughout Cassie's hierarchy. The Dual Asymmetric-Context Transformer already uses privileged information during training; this paper validates the approach for extreme dynamic behaviors. The distillation from privileged teacher to limited-observation student is relevant to how the Controller level could be trained with full state access then deployed with only proprioception.

## What to Borrow / Implement
- Implement teacher-student distillation for Mini Cheetah: privileged teacher in MuJoCo → proprioceptive student for deployment
- Adapt reward shaping templates for specific dynamic maneuvers (jumping, bounding)
- Use the depth encoder architecture as a starting point for vision-augmented Mini Cheetah policies
- Apply DAgger-style distillation to refine the student policy beyond pure behavior cloning
- Benchmark Mini Cheetah against the Unitree A1 parkour results as an agility target

## Limitations & Open Questions
- Extreme maneuvers may exceed the mechanical limits of platforms with different actuator capabilities (e.g., Mini Cheetah vs. A1)
- Teacher-student gap: student may not fully recover all teacher behaviors, especially at the extremes
- Depth-based perception limited by camera field of view and frame rate during fast maneuvers
- Energy consumption during extreme maneuvers not characterized; may limit deployment duration
- Generalization to entirely novel obstacle configurations not extensively tested
