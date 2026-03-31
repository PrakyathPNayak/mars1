# AnyMorph: Learning Transferable Policies By Inferring Agent Morphology

**Authors:** Brandon Huang, Yiding Chen, Jonathan Tompson
**Year:** 2022 | **Venue:** ICML
**Links:** https://arxiv.org/abs/2206.12279

---

## Abstract Summary
AnyMorph presents a method for learning morphology-transferable reinforcement learning policies without requiring explicit morphological descriptors or hand-crafted morphology representations. Unlike prior approaches that feed URDF parameters, kinematic tree structures, or explicit limb features to the policy, AnyMorph learns a latent morphology embedding purely from the RL objective itself. The key insight is that a well-trained policy implicitly encodes information about the robot's morphology through its input-output mapping — AnyMorph makes this implicit knowledge explicit by training a morphology inference module alongside the policy.

The approach uses a permutation-invariant architecture that processes per-joint observations and infers a fixed-dimensional morphology embedding through a learned aggregation function. This embedding captures the essential morphological characteristics (number of limbs, symmetry structure, mass distribution) without being told what any of these concepts are. The embedding is then conditioned into the policy, enabling a single policy network to adapt its behavior to different morphologies at inference time based purely on proprioceptive observations.

AnyMorph achieves state-of-the-art zero-shot transfer to unseen robot morphologies on standard benchmarks, outperforming methods that use explicit morphology descriptions. This result suggests that learning to infer morphology is more effective than hand-specifying it, possibly because the learned embedding captures task-relevant morphological features that human-designed descriptors miss.

## Core Contributions
- **Learned morphology inference:** Demonstrated that a learned morphology embedding trained end-to-end with RL outperforms explicit hand-crafted morphological descriptors for cross-morphology transfer
- **Permutation-invariant architecture:** Designed a set-based processing pipeline where per-joint observation tokens are aggregated into a fixed-size morphology embedding via attention pooling, invariant to joint ordering
- **No hand-designed descriptors:** Eliminated the need for URDF parameters, link masses, joint types, or kinematic tree structure as explicit inputs — the policy learns what it needs purely from proprioception
- **State-of-the-art zero-shot transfer:** Achieved the best zero-shot generalization to unseen morphologies across multiple locomotion benchmarks
- **Morphology embedding analysis:** Showed that the learned embeddings cluster meaningfully — similar morphologies (e.g., all quadrupeds) map to nearby points in embedding space, despite no explicit supervision

## Methodology Deep-Dive
AnyMorph's architecture consists of three modules: a per-joint encoder, a morphology inference module, and a morphology-conditioned policy. The per-joint encoder processes each joint's local observation (joint angle, velocity, local body orientation, contact force) through a shared MLP, producing per-joint feature vectors `{f_1, f_2, ..., f_N}` where `N` is the number of joints (variable across morphologies).

The morphology inference module aggregates these per-joint features into a fixed-size morphology embedding `m ∈ R^d_m` using a multi-head attention pooling mechanism. Specifically, a set of `K` learnable query vectors attend to the per-joint features via cross-attention, producing `K` output vectors that are concatenated and projected to yield `m`. This is permutation-invariant (the output doesn't depend on how joints are ordered) and handles variable-length inputs (any number of joints). The key design choice is that the morphology embedding is computed from a short window of observations (16 timesteps), allowing it to capture dynamic properties (e.g., natural frequencies, inertial response) in addition to static properties (e.g., mass distribution).

The morphology-conditioned policy takes the morphology embedding and current per-joint observations as input. The embedding is injected via FiLM conditioning: it modulates the per-joint features through learned scale and shift parameters `γ(m), β(m)` applied after each MLP layer. This allows the policy's behavior to be continuously modulated by the inferred morphology, rather than discretely switching between morphology-specific sub-policies.

Training alternates between two phases: (1) policy optimization via PPO using the current morphology embedding module, and (2) morphology embedding refinement using a contrastive loss that encourages embeddings from the same morphology to be similar and embeddings from different morphologies to be dissimilar. This contrastive auxiliary loss is critical for preventing the embedding from collapsing (all morphologies mapping to the same point) and for ensuring the embedding captures morphology-discriminative features.

The training set consists of 80 procedurally-generated morphologies in MuJoCo. Each training batch samples 8 morphologies, with 512 environments per morphology (4096 total parallel environments). Evaluation is on 20 held-out morphologies that were never seen during training. The authors also test transfer to morphologies with qualitatively different structures (e.g., training on legged robots, transferring to snake-like robots), finding that transfer quality degrades when the held-out morphology is too dissimilar to the training distribution.

## Key Results & Numbers
- Zero-shot transfer to 20 held-out morphologies: AnyMorph achieves 91% of morphology-specific oracle performance, vs. 85% for MetaMorph and 82% for SMP (explicit morphology descriptors)
- Morphology embedding quality: t-SNE visualization shows clear clustering by morphology family (bipeds, quadrupeds, hexapods) with no supervision
- Ablation without contrastive loss: performance drops to 79%, confirming the auxiliary loss is critical
- Ablation without temporal observation window: drops to 84%, showing dynamic properties matter for morphology inference
- Training: 500M environment steps, ~16 hours on 8 GPUs (A100), 2.5M policy parameters + 500K inference parameters
- FiLM conditioning vs. concatenation: FiLM improves transfer by ~4% over simple concatenation of morphology embedding

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
AnyMorph's learned morphology inference is relevant to the Mini Cheetah project primarily in the context of sim-to-real transfer and domain randomization. If the sim-to-real gap can be framed as a "morphology" difference (different effective masses, frictions, motor characteristics between simulation and reality), AnyMorph's approach of inferring these properties from proprioception could enable automatic system identification during deployment. This is more elegant than explicit domain randomization because the policy learns to adapt rather than being forced to be robust across all possibilities. However, the Mini Cheetah's fixed morphology during deployment limits the direct applicability of cross-morphology transfer.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
AnyMorph's learned morphology embedding offers a compelling alternative or augmentation to the explicit kinematic tree representation used in MC-GAT. Instead of manually constructing Cassie's kinematic graph with GATv2, a learned morphology embedding could capture task-relevant structural features that emerge from training. This is especially relevant because: (1) the contrastive learning approach for morphology embeddings aligns with Project B's CPTE (Contrastive Pretrained Terrain Encoder) methodology — both use contrastive objectives to learn discriminative representations; (2) the FiLM conditioning mechanism could be used to modulate the Primitives or Controller level based on inferred morphological context; (3) if Project B aims to generalize to Cassie variants or other bipeds, learned morphology inference is more scalable than hand-designed kinematic graph representations. The attention pooling architecture could also replace mean/sum aggregation in MC-GAT's readout layer.

## What to Borrow / Implement
- **Contrastive morphology embedding training:** Use the contrastive auxiliary loss alongside RL training to learn a discriminative morphology/robot-state embedding — directly applicable to CPTE's contrastive approach
- **FiLM conditioning for morphology injection:** Replace concatenation-based morphology conditioning in MC-GAT with FiLM conditioning, which has been shown to be more effective for modulating policy behavior
- **Temporal observation window for inference:** Use a 16-step observation window for morphology/terrain inference, capturing dynamic properties that single-frame observations miss
- **Attention pooling for variable-length aggregation:** Apply attention pooling instead of mean pooling when aggregating per-joint features, for a learnable and more expressive readout from the kinematic graph
- **Ablation baseline:** Use AnyMorph's learned embedding as a baseline comparison against MC-GAT's explicit graph representation to determine whether explicit structure helps

## Limitations & Open Questions
- The learned morphology embedding is a black box — unlike MC-GAT's explicit kinematic graph, it provides no interpretability about which structural features are captured, making debugging harder
- Transfer quality degrades significantly when test morphologies are qualitatively different from training morphologies; Cassie's unique compliant leg mechanism may fall outside the training distribution
- The contrastive loss requires careful tuning (temperature, negative sampling strategy) and adds training complexity
- Integration with hierarchical RL is unexplored — AnyMorph operates as a flat policy, and it is unclear how morphology conditioning should interact with the Planner→Primitives→Controller→Safety hierarchy
