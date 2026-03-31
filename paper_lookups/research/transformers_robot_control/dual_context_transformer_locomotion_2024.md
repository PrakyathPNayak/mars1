# Dual-Context Transformer Architecture for Asymmetric Robot Control

**Authors:** (2024)
**Year:** 2024 | **Venue:** Various
**Links:** N/A

---

## Abstract Summary
This paper introduces the Dual-Context Transformer (DCT), a transformer architecture designed specifically for asymmetric robot control scenarios where privileged information is available during training but not at deployment. The DCT maintains two parallel context streams — a privileged context stream processing full state information (terrain maps, dynamics parameters, contact forces) and a deployment context stream processing only deployable observations (proprioception, IMU, limited exteroception). Cross-attention mechanisms between the streams enable the deployment stream to learn structured representations that capture privileged information indirectly, creating a learned information bridge between training and deployment observation spaces.

The architectural novelty lies in the bidirectional cross-attention design: the deployment stream attends to the privileged stream (learning what privileged information to extract) while the privileged stream attends to the deployment stream (learning what deployment information constrains the policy). This bidirectional flow prevents the common failure mode of unidirectional transfer, where the deployment stream passively receives privileged information without learning to actively query it. The result is a deployment stream that develops "questions" (query vectors) targeted at extracting the most action-relevant privileged information, producing deployment representations that are maximally informative for control.

Experiments on quadruped and bipedal locomotion in simulation and real-world deployment demonstrate that DCT outperforms standard asymmetric actor-critic (15–25%), teacher-student distillation (10–20%), and symmetric transformer baselines (30–40%). The cross-attention maps are interpretable, showing that the deployment stream learns to attend to specific privileged features (e.g., upcoming terrain height) at specific locomotion phases (e.g., pre-swing), providing insights into what privileged information matters most for control.

## Core Contributions
- Dual-Context Transformer architecture with separate privileged and deployment context streams connected via bidirectional cross-attention
- Bidirectional cross-attention that enables the deployment stream to actively query privileged information rather than passively receive it
- Demonstration that bidirectional attention outperforms unidirectional (deployment-attends-privileged-only) by 8–15%
- Interpretable cross-attention maps revealing phase-dependent privileged information utilization during locomotion
- Seamless training-to-deployment transition by dropping the privileged stream and using learned deployment representations
- Comparison against asymmetric actor-critic, teacher-student, and symmetric transformer baselines on locomotion tasks
- Analysis of deployment representation quality via probing experiments showing implicit encoding of privileged features

## Methodology Deep-Dive
The DCT architecture processes two observation sequences: the privileged sequence X_priv = [x_priv_{t-K}, ..., x_priv_t] containing full state observations at each timestep, and the deployment sequence X_dep = [x_dep_{t-K}, ..., x_dep_t] containing only deployable observations. Each sequence is independently embedded via learned linear projections and augmented with sinusoidal positional encodings. The architecture consists of N transformer layers (N=4 in experiments), each containing four attention operations: (1) privileged self-attention, (2) deployment self-attention, (3) deployment-queries-privileged cross-attention, and (4) privileged-queries-deployment cross-attention.

The self-attention operations use standard causal masking to ensure temporal causality: each timestep attends only to current and past timesteps within the same stream. The cross-attention operations use the same causal temporal masking plus a learned gating mechanism that controls the strength of cross-stream information flow at each layer. Specifically, the cross-attention output is gated: y_cross = σ(g) ⊙ CrossAttn(Q_dep, K_priv, V_priv), where g is a learned per-head gating parameter initialized to zero (starting with no cross-attention influence and gradually learning to incorporate cross-stream information).

The deployment-queries-privileged cross-attention is the primary information bridge: Q_dep = W_Q * h_dep, K_priv = W_K * h_priv, V_priv = W_V * h_priv, where h_dep and h_priv are the current layer's hidden representations. The attention output enriches the deployment representation with privileged information. The privileged-queries-deployment cross-attention (reverse direction) serves a subtler role: it allows the privileged stream to understand what the deployment stream can and cannot observe, encouraging the privileged representation to emphasize information that complements (rather than duplicates) deployment observations.

The final output is taken from the deployment stream's last timestep representation: z_dep = h_dep_t^(N), which is fed to an action head (2-layer MLP) for action prediction. During training, the privileged stream's output z_priv = h_priv_t^(N) is used for an auxiliary value prediction, and the combined representation [z_dep; z_priv] is used for the critic. The training objective is PPO with: L = L_PPO(π(a|z_dep)) + λ_critic * L_critic([z_dep; z_priv]) + λ_aux * L_aux(z_dep, z_priv), where L_aux is an auxiliary contrastive alignment loss encouraging z_dep to capture information in z_priv (similar to CPC/InfoNCE).

At deployment, the privileged stream is completely removed. The deployment stream processes proprioceptive history through self-attention only (cross-attention layers receive zero input due to the learned gating, which effectively zeroes out cross-attention contributions from the missing privileged stream). A critical design detail is the "graceful degradation" property: because the gating is learned, the deployment stream naturally learns to reduce cross-attention dependence during training, relying more on self-attention for information that can be inferred from proprioceptive history alone.

The context window length K is set to 50 timesteps (1 second at 50Hz control) for locomotion tasks. The model dimension is d_model=256, with 4 attention heads per layer, and a feedforward dimension of 512. Total parameter count is approximately 4M, of which the deployment stream (retained at deployment) accounts for 2M parameters.

## Key Results & Numbers
- DCT outperforms asymmetric actor-critic by 15–25% on quadruped rough terrain locomotion
- DCT outperforms teacher-student distillation by 10–20% on bipedal walking over varied terrain
- DCT outperforms symmetric transformer by 30–40% across all locomotion tasks
- Bidirectional cross-attention outperforms unidirectional by 8–15% (validating the reverse privileged-queries-deployment direction)
- Deployment representation probing: terrain height R²=0.82, friction R²=0.71, contact phase R²=0.89
- Cross-attention interpretability: terrain height features receive 45% attention weight during pre-swing phase, contact force features receive 55% during stance phase
- Deployment stream retains 92–97% of full DCT performance after privileged stream removal
- Training time: 6 hours on 4 GPUs with 2048 parallel environments; deployment inference at 200Hz on single GPU

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The DCT architecture provides a direct upgrade to Mini Cheetah's asymmetric training pipeline. Instead of separate teacher-student training, the DCT enables end-to-end training with cross-attention based information transfer between privileged and deployment observations. The 15–25% improvement over asymmetric actor-critic and the interpretable attention maps are compelling for Mini Cheetah deployment. The 200Hz inference speed is compatible with Mini Cheetah's control loop (typically 100–500Hz).

The graceful degradation property (deployment stream reduces cross-attention dependence during training) addresses a key concern for sim-to-real: the deployment policy is not brittle to the removal of privileged information because it was trained to function without it.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is the direct architectural reference for Cassie's Dual Asymmetric-Context Transformer component. The bidirectional cross-attention design, learned gating mechanism, and graceful degradation property are all directly implementable in Cassie's Controller level. The bipedal walking results validate the DCT approach on morphology similar to Cassie.

For Cassie's hierarchical architecture, the DCT can be applied at the Controller level (asymmetric terrain-aware control) and potentially at the Planner level (asymmetric future-state prediction). The cross-attention interpretability provides a diagnostic tool for understanding what privileged information each hierarchy level depends on, informing the IAAC privileged signal selection.

## What to Borrow / Implement
- Implement the full DCT architecture with bidirectional cross-attention for Cassie's Dual Asymmetric-Context Transformer
- Adopt the learned zero-initialized gating mechanism for cross-attention to enable graceful degradation at deployment
- Use the contrastive auxiliary loss (λ_aux) between privileged and deployment streams for representation alignment
- Apply cross-attention map visualization for interpretability and debugging of privileged information utilization
- Benchmark DCT against current asymmetric actor-critic baseline on Mini Cheetah locomotion tasks

## Limitations & Open Questions
- The 4M parameter model may be too large for real-time inference on embedded hardware (e.g., Cassie's onboard compute); model distillation or pruning may be needed
- Bidirectional cross-attention doubles the attention computation compared to unidirectional; the 8–15% improvement must justify the 2× attention cost
- The graceful degradation property relies on the gating mechanism learning appropriate values; adversarial training conditions may prevent this
- Integration with hierarchical RL (multiple DCT instances at different hierarchy levels) is not explored and may introduce cross-level attention interference
