# Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv:2509.12858](https://arxiv.org/abs/2509.12858)

---

## Abstract Summary
This paper presents a contrastive learning framework for sim-to-real transfer of adaptive humanoid locomotion that achieves zero-shot deployment on real hardware using only proprioceptive sensing. The key idea is to train a latent terrain encoder during simulation that maps proprioceptive history (joint positions, velocities, torques, IMU data) to a compact latent vector encoding terrain properties (friction, slope, compliance, roughness). This encoder is trained using a contrastive loss that pulls together latent representations from the same terrain type and pushes apart representations from different terrains.

The resulting "distilled awareness" enables the policy to proactively adjust its gait before encountering terrain changes, based solely on proprioceptive cues from recent footsteps. Unlike explicit terrain estimation (which predicts physical parameters), the contrastive approach learns a task-relevant latent space that captures exactly the terrain information needed for locomotion adaptation, discarding irrelevant details. The policy receives the latent terrain vector as an additional input alongside the standard proprioceptive state.

The framework is validated on a full-sized humanoid robot traversing diverse terrains: high steps (up to 15cm), steep slopes (up to 20°), loose gravel, and slippery surfaces—all with zero-shot sim-to-real transfer and no fine-tuning. The contrastive terrain representation transfers robustly because it is invariant to the sim-to-real domain gap: the proprioceptive signatures of different terrains are consistent between simulation and reality, even when absolute sensor values differ.

## Core Contributions
- Proposes a contrastive proprioceptive terrain encoder that learns a latent terrain representation from proprioceptive history without requiring explicit terrain labels or exteroceptive sensors
- Demonstrates zero-shot sim-to-real transfer of adaptive humanoid locomotion across diverse terrains using only proprioception, matching or exceeding vision-based baselines
- Introduces a terrain-contrastive loss function: InfoNCE applied to proprioceptive trajectory segments, with positive pairs from same terrain and negative pairs from different terrains
- Shows that the learned latent space captures physically meaningful terrain properties (friction clusters, slope gradients, compliance spectra) without explicit supervision
- Validates on a full-sized humanoid with 20+ DoF, demonstrating scalability beyond the small robots typically used in sim-to-real locomotion research
- Achieves proactive gait adaptation—the policy adjusts foot placement and swing trajectories 1-2 steps before reaching a terrain transition, based on proprioceptive cues from the current terrain contact
- Provides ablation studies showing that contrastive learning outperforms autoencoder-based and prediction-based terrain encoding approaches by 15-30% on traversal success rate

## Methodology Deep-Dive
The architecture consists of three components: a proprioceptive history encoder E_φ, a contrastive projection head P_ψ, and a locomotion policy π_θ. The history encoder E_φ takes a window of T recent proprioceptive observations o_{t-T:t} (each containing joint angles, joint velocities, joint torques, body angular velocity, body linear acceleration from IMU, and foot contact forces) and produces a latent terrain vector z_t = E_φ(o_{t-T:t}) ∈ R^d with d=32-64. The encoder architecture is a temporal convolutional network (TCN) with causal convolutions, ensuring that the latent vector only depends on past observations.

The contrastive training uses the InfoNCE loss. During simulation, terrain parameters are known, enabling automatic generation of positive and negative pairs. A batch of N trajectory segments is sampled: for each anchor segment τ_i, positive segments τ_i^+ come from rollouts on the same terrain (same friction, slope, etc.), and negative segments τ_i^- come from rollouts on different terrains. The loss is L_contrastive = -∑_i log(exp(sim(z_i, z_i^+)/τ) / ∑_j exp(sim(z_i, z_j)/τ)), where sim(·,·) is cosine similarity and τ is the temperature parameter. The projection head P_ψ maps z to a lower-dimensional space for the contrastive loss computation, following the SimCLR convention of discarding the projection head after training and using z directly.

Terrain sampling during training uses aggressive domain randomization: friction coefficients from [0.1, 2.0], slopes from [-25°, 25°], step heights from [0, 20cm], compliance from [rigid, soft foam], and surface roughness from [smooth, gravel]. Each training episode randomizes terrain parameters, and the contrastive loss is computed across the batch. Importantly, the terrain parameters are used only for constructing positive/negative pairs during training—they are never provided to the policy or encoder as inputs.

The locomotion policy π_θ is a standard MLP that receives the current proprioceptive state s_t concatenated with the terrain latent z_t and outputs joint position targets (PD control). The policy is trained with PPO in IsaacGym with thousands of parallel environments. The key insight is that the policy learns to condition its gait on z_t: on slippery terrain (low-friction cluster in z-space), the policy adopts a wider stance and slower gait; on stairs (high-step cluster), the policy increases foot clearance; on slopes, the policy shifts the center of mass forward/backward.

The proactive adaptation capability emerges naturally from the temporal encoding. Because the TCN encoder processes T=50 timesteps of history (~0.5 seconds at 100Hz), it detects terrain changes from the proprioceptive signatures of recent footsteps. When the robot steps from flat ground onto a slope, the encoder detects the change within 1-2 footsteps and updates z_t, causing the policy to adapt its gait for the new terrain before the next foot placement.

For sim-to-real transfer, the contrastive terrain representation is robust because it encodes relative proprioceptive patterns (e.g., the ratio of vertical to horizontal foot force on slopes) rather than absolute sensor values. Domain randomization of sensor noise, actuator delays, and contact dynamics during training further improves transfer robustness. The authors apply no real-world fine-tuning; the simulation-trained policy and encoder are deployed directly.

## Key Results & Numbers
- Zero-shot sim-to-real transfer on 5 terrain types: flat, stairs (15cm), slopes (20°), gravel, slippery tile—with 95%+ traversal success rate
- Contrastive terrain encoding outperforms autoencoder encoding by 23% and explicit parameter prediction by 18% on aggregate traversal success
- Latent space visualization (t-SNE) shows clear clustering by terrain type, with within-cluster variation capturing sub-properties (e.g., low vs high friction within the "slippery" cluster)
- Proactive gait adaptation detected 1.2±0.4 steps before terrain transition, compared to 3.5±1.1 steps for reactive-only baselines
- Proprioceptive-only policy matches the traversal success rate of a vision-augmented baseline on stairs and slopes, and exceeds it on slippery/deformable terrains where visual cues are unreliable
- TCN encoder processes 50-step history in 0.8ms on an NVIDIA Jetson, enabling real-time deployment at 100Hz
- Training requires ~2 billion environment steps in IsaacGym with 4096 parallel environments, completing in ~8 hours on 1 GPU

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The contrastive terrain encoding approach is applicable to Mini Cheetah's blind locomotion (proprioceptive-only). The TCN-based history encoder could replace or augment the standard observation history concatenation used in most quadruped RL papers. The proactive adaptation capability is particularly valuable for Mini Cheetah navigating outdoor terrain where foot contacts provide early warning of terrain changes.

However, Mini Cheetah's smaller size and faster dynamics (higher control frequency) may require different TCN window sizes and architecture choices. The quadruped's four-legged gait also provides richer terrain sampling per stride (4 foot contacts vs 2 for humanoid), potentially enabling faster terrain detection.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper directly validates the CPTE (Contrastive Proprioceptive Terrain Encoder) approach that is a core component of Project B's Planner level. The contrastive learning framework, TCN architecture, InfoNCE loss, and domain randomization strategy described in this paper provide a ready-to-implement blueprint for the CPTE.

Specific correspondences to Project B's architecture: (1) The terrain latent vector z_t corresponds to the CPTE output that feeds into the Planner's Dual Asymmetric-Context Transformer. (2) The contrastive loss construction using terrain parameters as grouping labels during simulation training is exactly the CPTE training procedure. (3) The proactive adaptation (1-2 steps ahead) validates the hypothesis that proprioceptive history contains sufficient information for anticipatory gait planning, which is the basis of Project B's Planner → Primitives → Controller cascade. (4) The zero-shot sim-to-real transfer validates that contrastive terrain representations bridge the domain gap, critical for Cassie's real-world deployment.

The main difference is that Project B's CPTE feeds into a hierarchical system (Planner level) rather than directly into a flat policy. The terrain latent should therefore encode higher-level terrain semantics (terrain type, traversability) suitable for the Planner's long-horizon decision-making, rather than low-level gait parameters. This may require modifying the contrastive loss to group by terrain traversal strategy rather than raw terrain parameters.

## What to Borrow / Implement
- Adopt the TCN-based proprioceptive history encoder architecture for Project B's CPTE, using causal convolutions over T=50-100 timestep windows
- Use the InfoNCE contrastive loss with terrain parameter-based positive/negative pair construction during simulation training
- Apply aggressive domain randomization of terrain parameters during contrastive training to ensure robust sim-to-real transfer of the terrain latent space
- Implement the cosine similarity metric with temperature τ=0.07 for the contrastive loss, following the paper's reported best configuration
- For Project A, integrate the contrastive terrain encoder as an auxiliary module feeding into the PPO policy's observation space

## Limitations & Open Questions
- The method requires known terrain parameters during training for contrastive pair construction; fully unsupervised terrain encoding (without simulation terrain labels) remains open
- The TCN architecture assumes a fixed history window T; adaptive window lengths based on gait phase or terrain detection confidence are not explored
- The paper does not address dynamic terrains (moving platforms, collapsing surfaces) where the terrain properties change during a single traversal
- Scaling to very diverse terrain distributions may cause latent space fragmentation; the paper tests only 5 terrain categories
