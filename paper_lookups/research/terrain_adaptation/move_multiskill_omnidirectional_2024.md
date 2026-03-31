# MOVE: Multi-skill Omnidirectional Legged Locomotion with Limited View in 3D Environments

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2412.03353)

---

## Abstract Summary
MOVE presents a framework for multi-skill omnidirectional legged locomotion in complex 3D environments using limited field-of-view (FoV) exteroception. The core challenge addressed is that onboard cameras provide only a partial view of the surrounding terrain, yet the robot must infer terrain properties beyond its immediate FoV to plan and execute complex maneuvers such as climbing, leaping, and navigating narrow passages. MOVE solves this through a pseudo-siamese network architecture that fuses egocentric vision with proprioceptive features using a combination of supervised and contrastive learning objectives.

The framework introduces a terrain inference module that predicts terrain characteristics beyond the robot's visible range by learning correlations between observed and unobserved terrain regions. This is achieved through a pseudo-siamese architecture where one branch processes the visible terrain heightmap and the other processes proprioceptive history, with contrastive learning aligning their representations. The fused terrain embedding enables the policy to anticipate upcoming terrain changes and pre-adapt its locomotion strategy.

MOVE demonstrates successful execution of complex locomotion skills including stair climbing, gap leaping, and obstacle negotiation in both simulation and real-world deployments. The system achieves robust performance even when the visual field of view is significantly restricted (down to 60°), showcasing the effectiveness of the terrain inference module in compensating for limited exteroception.

## Core Contributions
- **Pseudo-siamese terrain inference network** that fuses egocentric vision and proprioceptive features to predict terrain beyond the visible FoV
- **Combined supervised + contrastive learning** for terrain representation that provides both discriminative and reconstructive properties
- **Multi-skill locomotion policy** capable of diverse maneuvers (climbing, leaping, crawling) with smooth inter-skill transitions
- **Omnidirectional control** supporting arbitrary heading and velocity commands while adapting locomotion skills to terrain requirements
- **Robustness to limited exteroception** demonstrating graceful degradation as FoV narrows, rather than catastrophic failure
- **Real-world deployment** on quadruped hardware with onboard computation validating the full pipeline's practicality

## Methodology Deep-Dive
The MOVE architecture consists of three main components: a terrain inference module, a skill selection module, and a low-level locomotion controller. The terrain inference module is the primary contribution and uses a pseudo-siamese network design. The two branches share a similar architecture but have different input modalities. The vision branch processes a local elevation map extracted from the robot's depth camera into a multi-scale feature pyramid, capturing terrain geometry at different spatial scales. The proprioceptive branch processes a sliding window of joint states (positions, velocities, torques) and IMU data through a 1D convolutional encoder, capturing the robot's dynamic response to terrain.

The contrastive learning component aligns the two branch outputs in a shared latent space. When the robot has traversed a terrain region, both the visual observation of that region (from before traversal) and the proprioceptive experience (from during traversal) should map to similar embeddings. This cross-modal alignment enables the proprioceptive branch to compensate when visual information is unavailable or unreliable. The supervised component adds a terrain reconstruction loss, requiring the fused embedding to predict terrain heightmap statistics (mean height, roughness, slope, gap presence) in a local neighborhood. This dual objective ensures the embeddings are both discriminative (contrastive) and information-preserving (supervised).

The skill selection module operates on the terrain embedding and current robot state to choose among a discrete set of locomotion primitives: walking, trotting, climbing, leaping, and crawling. Each skill has an associated sub-policy trained with PPO in dedicated terrain scenarios. The skill selector uses a gating network that produces soft skill blending weights, enabling smooth transitions. A regularization term penalizes frequent skill switching to prevent oscillatory behavior.

Training follows a multi-stage curriculum: (1) individual skill policies are trained on their respective terrain types, (2) the terrain inference module is trained on diverse terrain sequences, and (3) the full system is fine-tuned end-to-end with the skill selector. Domain randomization covers camera intrinsics/extrinsics, terrain parameters, physics properties, and observation noise. The FoV is randomized during training from 45° to 120° to build robustness to varying visual coverage.

The omnidirectional control is achieved through a velocity-conditioned policy that takes desired linear and angular velocities as inputs. The terrain inference module modulates the velocity command feasibility, automatically reducing speed when approaching difficult terrain and suggesting directional adjustments to avoid impassable obstacles.

## Key Results & Numbers
- Successfully executes **5 distinct locomotion skills** with smooth transitions across diverse 3D terrain
- Maintains **>80% success rate** with FoV restricted to 60° (compared to 95% at full 120° FoV)
- **Terrain inference accuracy** of 87% for predicting terrain type 1-2 steps ahead of the robot
- Contrastive + supervised learning improves terrain embedding quality by **15-20%** over contrastive-only or supervised-only baselines (measured by downstream task performance)
- **Real-world transfer** demonstrated on Unitree quadruped navigating indoor/outdoor environments with stairs, ramps, and gaps
- Leap success rate of **75%** on 40cm gaps (compared to 45% for vision-only baseline without contrastive fusion)
- **Skill transition smoothness** measured by peak joint torque during switches: 30% reduction compared to hard-switching baseline
- Training time: approximately **48 GPU-hours** for the full pipeline

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
MOVE is highly relevant to the Mini Cheetah project as it demonstrates multi-skill locomotion with vision-proprioception fusion on quadruped hardware similar to the Mini Cheetah. The pseudo-siamese fusion architecture provides a blueprint for integrating depth camera data with the Mini Cheetah's 12 DoF proprioceptive information. The skill selection and blending mechanism could enable the Mini Cheetah to learn multiple gaits (walk, trot, gallop) and terrain-specific skills (stair climbing, gap crossing) within a single framework.

The FoV robustness is particularly relevant since the Mini Cheetah's onboard cameras have limited coverage. The demonstrated graceful degradation under reduced FoV means the system could maintain functionality even with the Mini Cheetah's specific sensor configuration. The multi-stage training curriculum aligns with the project's curriculum learning approach.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
MOVE's contrastive terrain fusion is directly applicable to Cassie's CPTE (Contrastive Terrain Encoder) design. The pseudo-siamese architecture that aligns vision and proprioceptive embeddings in a shared space provides a validated methodology for CPTE's cross-modal terrain encoding. The combination of contrastive and supervised objectives could enhance CPTE's terrain representations beyond pure contrastive learning.

The multi-skill architecture with smooth skill transitions maps naturally to Cassie's Primitives level in the 4-level hierarchy. The gating-based skill blending could inform how Cassie transitions between locomotion primitives (walking, running, stepping). The skill selector's terrain-conditioned operation mirrors the Planner→Primitives interface in Cassie's architecture. The demonstrated robustness to limited exteroception is crucial for Cassie's deployment where sensor reliability may vary.

## What to Borrow / Implement
- **Pseudo-siamese fusion architecture** — implement dual-branch (vision + proprioception) encoder with shared latent space for CPTE, using cross-modal contrastive alignment
- **Combined supervised + contrastive terrain loss** — augment CPTE's contrastive objective with terrain reconstruction loss for richer, more informative embeddings
- **Soft skill blending via gating network** — adapt the skill transition mechanism for Cassie's Primitives level to enable smooth locomotion mode switching
- **FoV randomization during training** — include camera FoV and occlusion randomization in domain randomization suite for both projects
- **Multi-scale terrain feature pyramid** — process elevation maps at multiple spatial resolutions for capturing both local foothold details and global terrain structure

## Limitations & Open Questions
- **Quadruped-specific design** — the multi-skill set and terrain maneuvers are tailored to quadrupeds; bipedal skill decomposition (walking, stepping, push recovery) requires different primitives
- **Computational overhead** — the pseudo-siamese network with multi-scale vision processing adds significant inference cost; real-time feasibility on Cassie's onboard compute needs validation
- **Discrete skill assumption** — the predefined skill set may not cover all locomotion scenarios; a continuous skill space (as in DIAYN/DADS) could be more flexible
- **Limited dynamic terrain handling** — the terrain inference predicts static terrain properties; moving obstacles or changing surfaces are not addressed
