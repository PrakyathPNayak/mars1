# Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response

**Authors:** Junfeng Long, Zirui Wang, Quanyi Li, Jiabin Yu, Linghan Meng, Liu Cao, Zhengyou Zhang
**Year:** 2024 | **Venue:** ICLR 2024
**Links:** [OpenReview](https://openreview.net/forum?id=93LoCyww8o)

---

## Abstract Summary
This paper introduces the Hybrid Internal Model (HIM), a novel framework for learning agile legged locomotion that treats external terrain properties as implicit disturbances rather than explicitly estimating them. The key insight is that a robot's proprioceptive response to terrain interactions encodes sufficient information about the environment for robust locomotion control. By leveraging contrastive learning to optimize a hybrid internal embedding that captures both velocity tracking and stability-related features, HIM eliminates the need for exteroceptive sensors during deployment.

The framework trains entirely in simulation using only proprioceptive observations (joint positions, velocities, IMU data) and demonstrates remarkable zero-shot transfer to real-world scenarios. The hybrid internal model fuses two complementary representations: a velocity-conditioned embedding that captures locomotion dynamics and a stability embedding that encodes balance-critical information. The contrastive objective ensures these embeddings are discriminative across different terrain conditions while remaining invariant to irrelevant observation noise.

Extensive experiments show that HIM achieves state-of-the-art performance on diverse terrains including stairs, slopes, gaps, and deformable surfaces, significantly outperforming prior proprioceptive-only methods. The approach generalizes to unseen terrains and demonstrates open-world robustness in real-world deployment on quadruped robots.

## Core Contributions
- **Hybrid Internal Model architecture** that decomposes terrain adaptation into velocity tracking and stability maintenance sub-problems, each with its own learned embedding
- **Contrastive learning objective** for terrain-conditioned embeddings that creates discriminative representations without explicit terrain labels or classification
- **Proprioceptive-only deployment** pipeline that requires no exteroceptive sensors (cameras, LiDAR) at test time while maintaining terrain adaptiveness
- **Simulated Robot Response (SRR)** mechanism that leverages the robot's physical interaction signatures as implicit terrain descriptors
- **Open-world generalization** demonstrated across terrain types never seen during training, validating the transferability of the learned internal model
- **State-of-the-art results** on multiple locomotion benchmarks with significant margins over existing proprioceptive baselines

## Methodology Deep-Dive
The HIM framework operates within a teacher-student paradigm but fundamentally reimagines what the teacher encodes. During privileged training, a teacher policy has access to ground-truth terrain information (friction, height, deformability). The student policy only receives proprioceptive history. The key innovation is the intermediate representation: rather than distilling terrain parameters directly, HIM learns a hybrid embedding that captures the robot's expected response to terrain interactions.

The hybrid internal model consists of two parallel encoders. The velocity encoder processes proprioceptive history through a temporal convolution network (TCN) and maps it to a latent space conditioned on the commanded velocity. This embedding captures how the robot's gait should adapt to maintain velocity tracking on different terrains. The stability encoder processes the same history but focuses on balance-critical features like body orientation, angular velocity, and foot contact patterns. A contrastive loss (InfoNCE variant) is applied to both embeddings, using terrain-matched pairs as positives and mismatched pairs as negatives.

The contrastive learning setup is carefully designed. During training, trajectory segments from the same terrain type form positive pairs, while segments from different terrains form negatives. Importantly, the contrastive objective operates on the robot's response patterns rather than raw terrain features, meaning the embeddings learn to distinguish terrains by how the robot interacts with them. This is more transferable than explicit terrain classification because it captures functionally relevant differences.

The policy network takes the concatenated hybrid embedding as additional input alongside the standard proprioceptive observation. A PPO-based RL training loop optimizes the policy while the contrastive loss co-optimizes the embeddings. Domain randomization is applied to physical parameters (mass, friction, motor strength) to ensure robustness. The final deployed policy runs at 50Hz on the robot's onboard computer with only IMU and joint encoder inputs.

Training involves a curriculum over terrain difficulty, starting with flat ground and progressively introducing more challenging surfaces. The contrastive buffer maintains a diverse set of terrain interaction samples to prevent embedding collapse. Temperature scheduling in the InfoNCE loss helps balance between early exploration and late-stage discrimination.

## Key Results & Numbers
- Achieves **85%+ success rate** on unseen terrain combinations in simulation, compared to 60-70% for prior methods
- **Zero-shot sim-to-real transfer** demonstrated on Unitree A1 quadruped across stairs (15cm), slopes (30°), and deformable surfaces
- Velocity tracking RMSE reduced by **30-40%** compared to RMA and other proprioceptive baselines
- Contrastive embeddings show **clear terrain clustering** in t-SNE visualization with >90% linear separability
- Stability metric (body orientation variance) improved by **25%** on rough terrain versus non-contrastive ablation
- Inference latency of **<2ms** on onboard compute, well within the 20ms control loop budget
- Training convergence in **~2000 epochs** (approximately 8 hours on a single GPU)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
HIM is directly applicable to the Mini Cheetah project as it addresses the core challenge of terrain-adaptive locomotion using only proprioceptive feedback. The contrastive learning approach for terrain embedding is a strong candidate for the Mini Cheetah's perception pipeline, especially given the 12 DoF system operating in MuJoCo. The proprioceptive-only deployment is particularly attractive as it eliminates sensor dependency and simplifies the sim-to-real pipeline. The PPO-based training with domain randomization and curriculum learning aligns perfectly with the Mini Cheetah project's methodology.

The hybrid embedding concept (velocity + stability) could be directly adopted for Mini Cheetah, with the velocity encoder handling gait pattern adaptation and the stability encoder managing balance on challenging terrains. The demonstrated sim-to-real transfer validates that this approach works on real quadruped hardware.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critical to Project B as it directly validates the Contrastive Terrain Encoder (CPTE) concept in Cassie's architecture. The contrastive learning framework for terrain-conditioned embeddings provides a proven methodology that can be adapted for bipedal terrain reasoning. The InfoNCE-based contrastive objective with terrain-matched positive pairs is directly transferable to CPTE's design.

The hybrid embedding decomposition (velocity + stability) maps naturally to Cassie's hierarchical structure. The velocity embedding could inform the Planner level about terrain-appropriate velocity commands, while the stability embedding could feed into the Safety level for balance-aware control. The teacher-student paradigm with privileged terrain information during training matches the asymmetric training approach planned for Cassie's system.

## What to Borrow / Implement
- **Contrastive terrain embedding architecture** — adapt the InfoNCE-based contrastive loss with terrain-matched positive pairs for CPTE implementation
- **Hybrid embedding decomposition** — implement separate velocity and stability encoders that jointly capture terrain properties through robot response
- **Proprioceptive history encoding via TCN** — use temporal convolution networks for processing joint state history windows as terrain feature extractors
- **Contrastive buffer management** — implement diverse terrain interaction sample storage to prevent embedding collapse during training
- **Curriculum-based terrain difficulty scheduling** — adopt progressive terrain complexity increase for both Mini Cheetah and Cassie training pipelines

## Limitations & Open Questions
- **Bipedal applicability unclear** — all experiments conducted on quadrupeds; the hybrid embedding decomposition may need significant adaptation for bipedal balance dynamics
- **Static terrain assumption** — the contrastive learning assumes terrain properties are locally stationary; dynamic surfaces (moving platforms, mud) are not addressed
- **Contrastive objective sensitivity** — the InfoNCE temperature and negative sampling strategy require careful tuning; no principled method for selecting these hyperparameters is provided
- **No explicit safety guarantees** — despite the stability embedding, there are no formal safety constraints or fallback mechanisms when the terrain embedding fails to capture critical features
