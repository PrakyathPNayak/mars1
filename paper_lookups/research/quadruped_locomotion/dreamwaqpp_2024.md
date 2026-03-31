# DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion with Resilient Multi-Modal Reinforcement Learning

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [Project Page](https://dreamwaqpp.github.io/)

---

## Abstract Summary
DreamWaQ++ extends the DreamWaQ framework with multi-modal sensor fusion combining proprioceptive state estimation with exteroceptive vision for obstacle-aware quadrupedal locomotion. The core innovation is a resilient multi-modal architecture that gracefully handles sensor dropouts, occlusions, and failures — maintaining robust locomotion even when vision is degraded or unavailable. This sensor resilience is achieved through a learned modality attention mechanism that dynamically re-weights proprioceptive and visual inputs based on their estimated reliability.

The system integrates a learned terrain representation from depth images with the proprioceptive state estimator from DreamWaQ, enabling the robot to perceive and navigate around obstacles while maintaining the blind-locomotion robustness of its predecessor. The architecture supports graceful degradation: when vision is fully occluded, the system seamlessly falls back to proprioception-only locomotion (equivalent to DreamWaQ behavior); when proprioceptive estimates are noisy, visual terrain information compensates.

DreamWaQ++ demonstrates deployment on diverse rough terrains including stairs, rubble, gaps, and obstacle courses, achieving state-of-the-art performance in terms of traversal success rate and locomotion robustness. The method also introduces cross-platform deployment capability, training policies that transfer across different quadruped morphologies with minimal fine-tuning.

## Core Contributions
- Multi-modal sensor fusion architecture combining proprioceptive state estimation with visual depth perception through a learned modality attention mechanism
- Resilient fusion that gracefully degrades when any modality is corrupted, occluded, or unavailable, maintaining locomotion robustness under sensor failure conditions
- Obstacle-aware navigation integrating terrain perception with locomotion control in a single end-to-end policy
- Extension of DreamWaQ's privileged learning framework to multi-modal observations with cross-modal distillation
- Cross-platform deployment capability enabling policy transfer across different quadruped morphologies
- State-of-the-art rough terrain traversal combining the blind robustness of DreamWaQ with the perceptive capability of vision-based methods
- Comprehensive real-world deployment on stairs, rubble, gaps, slopes, and cluttered environments

## Methodology Deep-Dive
The architecture consists of three main modules: a proprioceptive encoder E_prop, a visual encoder E_vis, and a modality fusion module F_att that combines their outputs into a unified latent representation for the locomotion policy.

The proprioceptive encoder E_prop follows the DreamWaQ design: it takes the robot's joint positions, velocities, base angular velocity, gravity vector estimate, and previous actions as input, processing through a 3-layer MLP [256, 128, 64] to produce a proprioceptive embedding h_prop ∈ R^64. This encoder is pre-trained using the DreamWaQ privileged learning paradigm: a teacher policy with access to ground-truth terrain information (friction, slope, height map) trains alongside a student encoder that must reconstruct a terrain embedding from proprioception alone. The proprioceptive encoder thus implicitly estimates terrain properties from the robot's dynamic response.

The visual encoder E_vis processes depth images from a front-facing camera. Raw depth images (64×64 resolution at 15Hz) are passed through a lightweight CNN (3 convolutional layers with 32, 64, 64 channels, followed by a 2-layer MLP [128, 64]) to produce a visual embedding h_vis ∈ R^64. The CNN is trained to extract terrain features relevant to locomotion: obstacle locations, step heights, gap widths, and surface normals. A key design choice is operating on egocentric depth rather than building explicit maps, reducing computational requirements and avoiding drift from SLAM errors.

The modality fusion module F_att implements a cross-attention mechanism: attention weights α_prop, α_vis = softmax(MLP_gate([h_prop; h_vis; σ_prop; σ_vis])), where σ_prop and σ_vis are learned uncertainty estimates for each modality. The fused embedding h_fused = α_prop · h_prop + α_vis · h_vis is fed to the locomotion policy π(a | h_fused, cmd). The gate MLP learns to detect sensor degradation: when visual depth estimates are noisy (high σ_vis), α_prop increases; when proprioceptive signals are unreliable (high σ_vis from mechanical vibration), α_vis compensates.

Resilience training is critical. During training, the method applies stochastic modality dropout: with probability p_vis = 0.3, the visual input is replaced with zeros (simulating camera occlusion), and with probability p_prop = 0.1, Gaussian noise is added to proprioceptive signals (simulating encoder noise or mechanical vibration). The policy must maintain locomotion under all dropout conditions. Additionally, adversarial visual perturbations (random depth artifacts, partial occlusion patterns) are applied to harden the visual pathway.

The training procedure uses a two-phase privileged learning approach. Phase 1: train a privileged teacher with access to ground-truth height maps and terrain properties using PPO. Phase 2: distill the teacher into the multi-modal student by matching the teacher's action distribution while training the proprioceptive and visual encoders to reconstruct the privileged terrain embedding. The distillation loss is: L = L_behavior_cloning + λ₁·L_terrain_reconstruction + λ₂·L_modality_consistency, where L_modality_consistency encourages h_prop and h_vis to agree on terrain properties when both modalities are available.

Cross-platform transfer is achieved through a morphology-agnostic trunk network with platform-specific input/output adapter layers. The trunk processes the fused embedding h_fused (which is morphology-agnostic after the adapter normalizes proprioceptive signals), while the adapters handle the mapping between platform-specific joint spaces and the normalized representation. Fine-tuning only the adapters (~5% of parameters) enables transfer to new platforms.

## Key Results & Numbers
- Rough terrain traversal success rate: 92% on obstacle courses (vs. 78% for DreamWaQ blind, 85% for vision-only baselines)
- Graceful degradation: maintained 81% success rate with complete visual occlusion (vs. 0% for vision-only methods)
- Stair climbing: 95% success rate on standard 15cm stairs, 82% on irregular stairs with varying heights
- Cross-platform transfer: 88% performance retention when transferring from Unitree A1 to Go1 with adapter-only fine-tuning
- Obstacle avoidance: 96% collision avoidance rate in cluttered environments at 0.5 m/s walking speed
- Latency: 2ms policy inference on NVIDIA Jetson Orin, enabling 500Hz control loop
- Camera-to-action end-to-end latency: 35ms including depth processing and policy inference
- Training: 12 hours on single A100 GPU for full two-phase training pipeline

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
DreamWaQ++ is directly relevant to Mini Cheetah as it extends DreamWaQ — a framework already highly applicable to Mini Cheetah's proprioception-first locomotion philosophy. The multi-modal fusion adds visual obstacle awareness while preserving the robust blind locomotion capabilities that are essential for Mini Cheetah operating in degraded visual conditions (dust, darkness, rain). The resilient fusion mechanism is critical for real-world Mini Cheetah deployment where camera failures are common. The cross-platform transfer capability could accelerate deployment if pre-trained on a similar quadruped. The low-latency inference (2ms) is compatible with Mini Cheetah's high-frequency control requirements.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The multi-modal resilient fusion architecture is conceptually applicable to Cassie, which also benefits from combining proprioceptive state estimation with visual terrain perception. The modality attention mechanism for handling sensor degradation is transferable. However, Cassie's bipedal locomotion has fundamentally different dynamics (underactuated ankle, single-support phases, higher fall risk) that require different proprioceptive encoders and terrain representations. The cross-platform transfer framework would need significant adaptation from quadruped to biped morphologies. The privileged learning approach for training terrain-aware policies is relevant to any legged robot platform.

## What to Borrow / Implement
- Adopt the modality attention fusion mechanism (learned gating between proprioceptive and visual embeddings) for Mini Cheetah multi-modal locomotion
- Implement stochastic modality dropout during training (p_vis=0.3, p_prop=0.1) to ensure robust performance under sensor failure
- Use the two-phase privileged learning pipeline: train privileged teacher → distill to multi-modal student with terrain reconstruction loss
- Apply the egocentric depth representation (64×64 front-facing depth) for lightweight obstacle perception without SLAM
- Consider the morphology-agnostic trunk + platform-specific adapter architecture for potential transfer between Mini Cheetah variants or other quadrupeds

## Limitations & Open Questions
- The egocentric depth representation limits perception to the front-facing camera's field of view — obstacles behind or to the side are not detected until the robot turns
- Cross-platform transfer assumes similar dynamic properties; transfer between significantly different platforms (e.g., heavy ANYmal to lightweight Mini Cheetah) may require full retraining
- The resilient fusion relies on learned uncertainty estimates that may not generalize to novel sensor failure modes not seen during training
- Real-time depth processing at 15Hz may miss fast-moving obstacles or rapidly changing terrain at higher locomotion speeds
