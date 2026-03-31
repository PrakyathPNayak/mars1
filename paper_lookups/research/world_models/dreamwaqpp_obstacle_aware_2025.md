# DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion with Resilient Multi-Modal Reinforcement Learning

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [Project Page](https://dreamwaqpp.github.io/)

---

## Abstract Summary
DreamWaQ++ extends the DreamWaQ framework to achieve robust obstacle-aware quadrupedal locomotion by fusing proprioception and vision in a learned latent space, with built-in resilience to partial sensor failure. The system handles complex terrain traversal—including stairs, slopes, gaps, and scattered obstacles—on four different hardware platforms, demonstrating strong generalization across morphologies and environments. The key design principle is that proprioception serves as a reliable safety net: even when visual observations are degraded or completely missing, the proprioceptive latent representation captures sufficient gait dynamics for stable locomotion.

The architecture uses a dual-encoder design where proprioceptive observations are mapped to a latent space that captures cyclic gait patterns, terrain contact dynamics, and body state estimation, while a separate visual encoder processes depth images into a terrain-aware latent representation. These two latent spaces are fused through a learned gating mechanism that dynamically weights each modality based on estimated reliability. During normal operation, vision provides anticipatory terrain information enabling proactive foothold planning, while proprioception provides reactive feedback ensuring stable gait execution.

The resilience to sensor failure is not merely a fallback mode—it is a designed capability trained through systematic modality dropout during training. By randomly masking visual inputs during 30-50% of training episodes, the policy learns to maintain competent locomotion using proprioception alone while leveraging vision when available for enhanced terrain navigation. This training protocol produces policies that gracefully degrade rather than catastrophically failing when sensors are compromised.

## Core Contributions
- Dual-encoder architecture fusing proprioceptive and visual latent spaces for obstacle-aware locomotion
- Learned gating mechanism for dynamic modality weighting based on estimated reliability
- Proprioceptive latent space that captures cyclic gait dynamics as a safety fallback
- Modality dropout training (30-50%) producing resilient policies that gracefully degrade under sensor failure
- Validated on 4 hardware platforms handling stairs, slopes, gaps, and obstacles
- Multi-modal world model that supports both reactive (proprioceptive) and anticipatory (visual) control
- Quantitative analysis of performance degradation curves under progressive vision corruption

## Methodology Deep-Dive
The proprioceptive encoder processes a state vector s_t = [q, q̇, ω, g, a_{t-1}, c_t] consisting of 12 joint positions, 12 joint velocities, 3-axis angular velocity, projected gravity vector (3D), previous action (12D), and estimated foot contact states (4D)—totaling ~46 dimensions. A history of H=50 timesteps (1 second at 50 Hz) is processed by a 1D temporal convolutional network (TCN) with 3 layers, kernel size 5, and 128 channels, producing a proprioceptive latent z_prop ∈ R^64. This encoder is designed to capture periodic gait patterns through the temporal convolutions, with the kernel size and depth tuned to cover full gait cycles (~0.5s for trotting).

The visual encoder processes 64×64 depth images from a forward-facing camera at 10 Hz (lower than the 50 Hz control loop). A lightweight ConvNet (4 convolutional layers with 32→64→128→128 channels, stride 2) produces a 1024-dimensional feature map, which is then projected to a visual latent z_vis ∈ R^64 via a linear layer. A temporal aggregation module (2-layer GRU with 128 hidden units) integrates visual features across 5 frames to provide stable terrain estimates despite camera jitter during dynamic locomotion. The 10 Hz visual processing rate is matched to the control loop through a zero-order hold on the visual latent between camera frames.

The modality fusion uses a learned gating network g_θ(z_prop, z_vis) → [α_prop, α_vis] where α_prop + α_vis = 1. The gate is implemented as a 2-layer MLP with softmax output that takes both latent representations as input and produces reliability-weighted fusion coefficients. The fused latent is z_fused = α_prop · z_prop + α_vis · z_vis. During training, the gate learns to assign high weight to vision when visual features are coherent (low reconstruction error from a visual decoder auxiliary task) and high weight to proprioception when visual features are noisy or missing.

The modality dropout training protocol randomly masks visual inputs during training with probability p_drop ∈ [0.3, 0.5]. When vision is masked, the visual encoder receives zero inputs and produces a default embedding, and the gating network is trained to recognize this situation and assign α_vis ≈ 0. This forces the policy to develop competent proprioception-only behavior while maintaining the ability to leverage vision when available. The dropout probability is annealed: starting at 0.5 for the first 30% of training (forcing strong proprioceptive capabilities) and decreasing to 0.3 for the remainder (allowing refinement of vision-enhanced behavior).

The policy network π(a|z_fused, cmd) takes the fused latent representation and velocity commands (v_x, v_y, ω_z) and outputs 12 target joint positions through a 3-layer MLP with 512 hidden units and ELU activations. Training uses PPO with the standard locomotion reward: r = r_velocity + r_alive - r_energy - r_smoothness - r_orientation, with terrain-specific bonuses for successful stair climbing and obstacle avoidance. A terrain curriculum progressively introduces harder terrains as the policy improves, starting with flat ground and advancing through slopes (±15°), stairs (10-20 cm), and obstacle fields.

## Key Results & Numbers
- Validated on 4 hardware platforms: Unitree Go1, Go2, A1, and Aliengo
- Successfully traverses: stairs (up to 20 cm), slopes (±20°), gaps (15 cm), scattered obstacles
- Vision-enabled performance: 95% success on complex obstacle courses
- Proprioception-only fallback: 78% success on the same courses (graceful degradation)
- Vision corruption: 50% Gaussian noise → 88% success; complete blackout → 78% success
- Gating learns to assign α_vis > 0.7 in open areas, α_prop > 0.6 on uniform terrain
- Proprioceptive latent captures gait phase with 94% classification accuracy (trot, walk, gallop)
- Control frequency: 50 Hz proprioceptive loop, 10 Hz visual loop
- Training: ~2 billion environment steps in Isaac Gym across terrain curriculum

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

DreamWaQ++ is directly applicable to the Mini Cheetah for obstacle-aware locomotion. The dual-encoder architecture with modality dropout can be implemented in the Mini Cheetah's MuJoCo training pipeline: train a proprioceptive TCN encoder and a visual depth encoder with gated fusion. The 50 Hz/10 Hz dual-rate design matches typical Mini Cheetah control and camera architectures. The proprioceptive fallback capability is crucial for real-world deployment where camera occlusion or failure is common during dynamic quadruped locomotion.

The terrain curriculum (flat → slopes → stairs → obstacles) provides a concrete training schedule for the Mini Cheetah's progressive skill acquisition. The modality dropout probability schedule (0.5 → 0.3) and gating network design can be adopted directly. The validated results on similar Unitree quadrupeds (Go1, A1) confirm transferability to the Mini Cheetah's similar morphology.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchy, DreamWaQ++'s proprioceptive latent space concept is directly applicable to the Controller level, where a temporal encoding of proprioceptive history captures gait phase and contact dynamics for stable bipedal locomotion. The visual encoder maps to the Planner level, where depth-based terrain awareness supports subgoal planning. The gating mechanism provides a principled approach for the Planner to weight terrain information vs. proprioceptive state.

The modality dropout training protocol should be adopted for Cassie's Planner to ensure robust planning when elevation map inputs are noisy or unavailable. The 78% proprioception-only success rate demonstrates that meaningful locomotion control is achievable without exteroceptive sensing, validating the importance of the proprioceptive pathway in Cassie's lower hierarchy levels (Controller and Safety).

## What to Borrow / Implement
- Implement the dual-encoder (TCN proprioceptive + ConvNet visual) architecture for Mini Cheetah multi-modal locomotion
- Adopt the modality dropout training protocol (p=0.3-0.5) for both Mini Cheetah and Cassie visual encoder training
- Use the learned gating mechanism for dynamic modality weighting in Cassie's Planner
- Apply the terrain curriculum schedule (flat → slopes → stairs → obstacles) for progressive training
- Implement the proprioceptive latent space as a gait phase estimator for Cassie's Controller level

## Limitations & Open Questions
- 10 Hz visual processing may miss fast-approaching obstacles; higher frame rates increase computational cost
- Gating mechanism adds complexity; simpler attention mechanisms may achieve similar modality fusion
- Training requires ~2 billion environment steps in Isaac Gym—significant computational investment
- Hardware validation limited to Unitree platforms; transfer to robots with different sensing configurations not yet tested
