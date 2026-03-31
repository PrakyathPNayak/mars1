# STERLING: Self-Supervised Terrain Representation Learning from Unconstrained Robot Experience

**Authors:** Haresh Karnan, Elvin Yang, Daniel Fink, Garrett Warnell, Joydeep Biswas, Peter Stone
**Year:** 2023 | **Venue:** CoRL 2023
**Links:** https://arxiv.org/abs/2309.15302

---

## Abstract Summary
STERLING learns terrain representations from unlabeled, non-expert robot experience using a non-contrastive (VICReg-inspired) framework. It aligns visual terrain patches with proprioceptive and inertial data to create task-relevant features without human supervision. The approach was demonstrated on a 3-mile autonomous robot hike with only 2 human interventions, showcasing robust terrain-aware navigation.

## Core Contributions
- Developed a self-supervised terrain representation learning framework that aligns visual and proprioceptive modalities without labels
- Adapted VICReg (Variance-Invariance-Covariance Regularization) loss for terrain feature learning, avoiding contrastive learning's need for negative sampling
- Demonstrated that proprioceptive/inertial signals contain rich terrain information that can supervise visual representations
- Achieved terrain classification performance matching supervised approaches using zero labeled data
- Validated on a 3-mile autonomous robot hike with only 2 human interventions required
- Outperformed contrastive baselines (SimCLR, MoCo) on terrain discrimination tasks
- Showed that the learned representations transfer across different robot platforms and environments

## Methodology Deep-Dive
STERLING's key insight is that a robot's proprioceptive and inertial experience while traversing terrain provides a natural supervision signal for visual terrain understanding. When a robot drives over gravel, its vibration patterns differ from driving over grass — this proprioceptive signature can teach a visual encoder what gravel and grass look like, without any human labels. The framework has three main components: a visual encoder, a proprioceptive encoder, and a self-supervised alignment objective.

The visual encoder processes terrain image patches (typically extracted from a downward-facing or forward-facing camera) through a CNN or Vision Transformer backbone. The proprioceptive encoder processes a window of IMU data (accelerometer, gyroscope) and motor current readings through a 1D CNN or temporal convolution network. Both encoders map their inputs to a shared embedding space where representations should be aligned.

The alignment uses VICReg (Variance-Invariance-Covariance Regularization), a non-contrastive self-supervised learning framework. VICReg has three loss terms: (1) Invariance — paired visual and proprioceptive embeddings from the same terrain patch should be close, (2) Variance — each dimension of the embedding should have sufficient variance across the batch (preventing representation collapse), and (3) Covariance — different dimensions should be decorrelated (encouraging diverse features). Unlike contrastive methods like SimCLR, VICReg does not require explicit negative pairs, which is important because defining "different terrain" is ambiguous without labels.

Data collection is deliberately unconstrained: the robot simply drives around in its environment, recording synchronized visual and proprioceptive streams. No special data collection protocol, no expert trajectories, no environment segmentation is needed. The only requirement is temporal alignment between visual and proprioceptive data, which is handled by hardware synchronization. This makes the approach highly practical — any robot with a camera and IMU can collect training data during normal operation.

The learned terrain representations are used downstream for terrain-aware navigation by training a simple preference model or classifier on top of the frozen embeddings. For the 3-mile hike demonstration, the robot used the representations to prefer traversable paths over difficult terrain (e.g., preferring sidewalks over bushes), with the preference model requiring only a few labeled examples on top of the self-supervised features. The approach is computationally lightweight enough to run in real-time on standard robot computing hardware.

## Key Results & Numbers
- Matches supervised terrain classification approaches using zero labeled data
- 3-mile autonomous robot hike with only 2 human interventions (for safety, not navigation failures)
- Outperforms contrastive baselines (SimCLR: -8%, MoCo: -12%) on terrain discrimination
- VICReg alignment achieves >85% terrain classification accuracy across 6 terrain types
- Training requires ~2 hours of unconstrained robot driving data
- Inference at >30 Hz on standard robot compute (Nvidia Jetson)
- Transfer learning: representations from one environment improve classification in unseen environments
- Ablation: removing proprioceptive alignment degrades performance by ~15%, confirming the value of multi-modal learning

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Terrain awareness is valuable for Mini Cheetah outdoor deployment. The self-supervised approach eliminates the labeling burden for terrain classification. Mini Cheetah's rich proprioceptive data (12 joint encoders, IMU) could provide an excellent supervision signal for learning visual terrain representations. If visual terrain awareness is added to the observation space, STERLING's framework could provide terrain embeddings without requiring manual terrain labeling in simulation or the real world. The VICReg loss is simpler and more stable than contrastive alternatives, making it easier to integrate into the existing training pipeline.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Directly relevant to the CPTE (Contrastive/self-supervised Terrain Encoder) module in Project B's architecture. STERLING's VICReg-inspired loss is a strong alternative to pure contrastive losses for the terrain encoder. The visual-proprioceptive alignment approach maps perfectly to Cassie's setup: the CPTE could align visual terrain observations (from depth cameras or lidar) with proprioceptive feedback (joint torques, IMU, foot contact forces) to learn terrain representations without supervision. The non-contrastive formulation avoids the need to define negative terrain pairs, which is especially challenging for continuous terrain variations. The downstream terrain preferences could inform the planner's locomotion primitive selection.

## What to Borrow / Implement
- Implement VICReg loss (variance + invariance + covariance) as the training objective for the CPTE module
- Use proprioceptive/inertial data to supervise visual terrain representation learning without labels
- Adopt the unconstrained data collection protocol: collect training data during normal robot operation
- Design the terrain encoder with separate visual and proprioceptive branches mapped to a shared embedding space
- Use temporal windows of IMU data (~0.5-1s) as the proprioceptive input for terrain characterization
- Test transfer of terrain representations across simulation environments before deploying to real hardware
- Combine STERLING's terrain embeddings with HiP-RSSM's hidden parameters for a comprehensive terrain characterization

## Limitations & Open Questions
- Proprioceptive terrain signatures may be confounded by gait phase — the same terrain feels different during stance vs. swing
- VICReg's variance regularization may prevent encoding fine-grained terrain differences that have low variance
- The approach assumes temporal alignment between visual and proprioceptive data, which may be noisy at high speeds
- Not directly validated on legged robots — the proprioceptive signatures of wheeled vs. legged traversal are very different
- Unclear how well representations generalize to terrains not seen during self-supervised training
- The framework does not explicitly model terrain geometry (slopes, steps) — only surface properties
- Real-time requirements for legged locomotion (500 Hz) may exceed the ~30 Hz inference speed demonstrated
