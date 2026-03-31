---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/rl_versatile_dynamic_robust_bipedal_locomotion.md

**Title:** Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control
**Authors:** Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, Koushil Sreenath
**Year:** 2024
**Venue:** IJRR (International Journal of Robotics Research)
**arXiv / DOI:** arXiv:2401.16889

**Abstract Summary (2–3 sentences):**
This paper presents a unified deep reinforcement learning framework for the Cassie bipedal robot that achieves versatile locomotion including walking, running, standing, and jumping through a single learned policy with zero-shot sim-to-real transfer. The key innovation is a dual-history policy architecture that maintains both short-term and long-term input/output histories to enable online system identification and rapid adaptation to changing dynamics. The framework demonstrates remarkable real-world performance, including a 400-meter dash and 1.4-meter standing jumps on hardware.

**Core Contributions (bullet list, 4–7 items):**
- Unified RL framework achieving multiple dynamic locomotion skills (walk, run, stand, jump) within a single policy
- Dual-history policy architecture combining short-term proprioceptive history for reactive control with long-term I/O history for implicit system identification
- Zero-shot sim-to-real transfer on Cassie without any real-world fine-tuning
- Demonstration of a 400-meter running dash on physical hardware, pushing the limits of bipedal robot speed
- 1.4-meter standing jumps achieved through the same policy framework
- Comprehensive robustness evaluation under external perturbations, payload variations, and terrain disturbances
- Extensive ablation studies demonstrating the necessity of both history components for robust transfer

**Methodology Deep-Dive (3–5 paragraphs):**
The policy architecture is built around a dual-history representation that processes two distinct temporal streams of information. The short-term history encodes a sliding window of recent proprioceptive observations (joint positions, velocities, IMU readings) and actions over the last 10–50 timesteps, providing the policy with immediate context about the robot's current dynamic state. This short-term buffer allows the policy to infer contact states, detect perturbations, and react to rapid changes in the environment. The long-term history spans hundreds of timesteps and is processed through a separate encoder, capturing slow-varying dynamics such as changes in terrain friction, payload shifts, or gradual actuator degradation that require implicit system identification.

Training is conducted entirely in simulation using the Isaac Gym parallel simulator with the MuJoCo-based Cassie model. The authors employ PPO with a carefully designed multi-term reward function that blends task-specific objectives (velocity tracking, heading control, height maintenance) with regularization terms (energy minimization, smoothness penalties, symmetry rewards). A key aspect of the training pipeline is aggressive domain randomization applied to dynamics parameters including mass, center of mass location, joint friction, ground friction, motor strength, and observation noise. The randomization ranges are calibrated through iterative comparison between simulated and real-world rollouts, ensuring the training distribution covers the real system's parameter space.

The command interface allows a human operator to specify desired forward velocity, lateral velocity, and turning rate, which the policy translates into full-body joint torque commands at 50 Hz. For jumping behaviors, a separate command signal triggers a pre-learned phase variable that modulates the policy's output to produce explosive extension followed by aerial stabilization and landing absorption. The policy outputs target joint positions that are converted to torques through PD controllers running at a higher frequency on the robot's onboard computer.

The sim-to-real transfer pipeline leverages the dual-history architecture as an implicit domain adaptation mechanism. Rather than relying on explicit system identification or online adaptation modules, the long-term history encoder learns to extract latent representations of the current dynamics from the robot's behavioral history. During deployment, as the robot accumulates real-world experience, the long-term encoder automatically adjusts the policy's behavior to account for the reality gap. This approach eliminates the need for privileged information distillation or teacher-student training paradigms, simplifying the overall pipeline while maintaining strong transfer performance.

Evaluation is performed across multiple axes: velocity tracking accuracy across a range of commanded speeds (0–3 m/s), robustness to external pushes (up to 120 N lateral forces), payload carrying (up to 5 kg added mass), and terrain traversal (grass, slopes, gravel). The authors also conduct direct comparisons with single-history baselines and demonstrate that both temporal scales are necessary for the full range of demonstrated capabilities.

**Key Results & Numbers:**
- 400-meter running dash completed on real Cassie hardware
- 1.4-meter standing jump achieved on physical robot
- Zero-shot sim-to-real transfer with no real-world fine-tuning
- Robustness to lateral pushes up to 120 N
- Payload carrying up to 5 kg without performance degradation
- Walking speeds from 0 to 1.5 m/s, running speeds up to 3.0 m/s
- Superior perturbation recovery compared to single-history and model-based baselines
- Policy runs at 50 Hz on onboard computation with <5 ms inference time

**Relevance to Project A (Mini Cheetah):** MEDIUM — The dual-history architecture concept is transferable to quadruped locomotion, particularly the idea of separating short-term reactive control from long-term system identification. However, the specific reward design and behavior repertoire are bipedal-specific and would need significant adaptation for the 12-DoF Mini Cheetah.

**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant as it targets the same Cassie platform and demonstrates a dual-history approach that conceptually maps to the asymmetric context design in the proposed Dual Asymmetric-Context Transformer. The zero-shot sim-to-real results establish strong baselines, and the multi-skill policy framework informs how primitives might be unified or structured within the hierarchical controller.

**What to Borrow / Implement:**
- Dual-history architecture as inspiration for the asymmetric context windows in the Dual Asymmetric-Context Transformer
- Domain randomization ranges calibrated for Cassie hardware (mass, friction, motor strength distributions)
- Multi-term reward structure balancing task performance with energy efficiency and smoothness
- The implicit system identification paradigm through long-term history encoding
- Evaluation protocols for zero-shot sim-to-real assessment on Cassie

**Limitations & Open Questions:**
- Cassie-specific tuning of reward functions and domain randomization ranges limits direct transferability to other platforms
- Limited to flat and mildly uneven terrain; no demonstration on stairs, rough terrain, or highly dynamic environments
- Extensive reward engineering required with manual tuning of 10+ reward terms and their relative weights
- Single policy approach may hit scalability limits as the skill repertoire grows beyond basic locomotion
- No explicit safety guarantees or constraint enforcement during deployment
- Computational cost of training with dual-history in massively parallel simulation not thoroughly analyzed
---
