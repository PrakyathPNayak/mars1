# Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv:2502.21142](https://arxiv.org/abs/2502.21142)

---

## Abstract Summary
This paper introduces a cognitive science-inspired world model architecture based on Global Workspace Theory (GWT) that natively handles multimodal sensory inputs—fusing vision, proprioception, and other modalities into a unified latent world model for reinforcement learning. Unlike standard RSSM-based world models that concatenate modality-specific features into a single representation, the GWT approach maintains separate modality-specific "specialist" modules that communicate through a shared "global workspace" bottleneck, enabling emergent cross-modal reasoning.

The key innovation is the broadcast-and-compete mechanism: each specialist module processes its modality independently, then competes for access to the shared workspace. The winning modality broadcasts its representation to all other specialists, enabling cross-modal information flow without requiring explicit fusion layers. This architecture naturally handles missing modalities—if vision is occluded or a sensor fails, proprioceptive specialists can still operate through the workspace without the system failing catastrophically.

Experimental results demonstrate that the GWT world model achieves competitive or superior performance compared to standard multimodal Dreamer variants while requiring fewer environment interactions. The robustness to missing modalities is particularly notable: performance degrades gracefully (10-20% drop) when one modality is completely removed at test time, compared to catastrophic failure (80%+ drop) for concatenation-based approaches. This makes the architecture especially suitable for real-robot deployment where sensor failures are common.

## Core Contributions
- Application of Global Workspace Theory from cognitive science to multimodal world model-based RL
- Broadcast-and-compete mechanism for dynamic cross-modal information routing
- Graceful degradation under missing modalities (10-20% performance drop vs. 80%+ for baselines)
- Separate modality-specific specialist modules maintaining inductive biases for each sensor type
- Reduced environment interaction requirements compared to standard multimodal Dreamer variants
- Emergent cross-modal attention patterns that qualitatively align with human multimodal integration
- Open architecture supporting arbitrary numbers of modality specialists

## Methodology Deep-Dive
The architecture consists of three main components: modality specialists S_i, a global workspace W, and a world model backbone. Each specialist S_i processes its respective modality through an appropriate encoder: a ConvNet for visual inputs, an MLP for proprioceptive vectors, and a PointNet for depth/lidar data. Each specialist produces a modality-specific latent representation h_i ∈ R^d, where d is the shared workspace dimensionality (typically 256-512).

The global workspace implements a soft competition mechanism using cross-attention. At each timestep, specialists produce "broadcast proposals" b_i = W_b · h_i, and the workspace computes attention weights α_i = softmax(W_q · w_{t-1} · (W_k · b_i)^T / √d) where w_{t-1} is the previous workspace state. The workspace state is updated as w_t = Σ_i α_i · (W_v · h_i), implementing a weighted combination where attention weights determine which modality dominates the workspace at each timestep. This soft attention replaces the hard winner-take-all competition from classical GWT, enabling gradient flow through all specialists.

After workspace update, the new workspace state w_t is broadcast back to all specialists through a feedback connection: h_i' = h_i + MLP_i(w_t). This broadcast enables cross-modal information transfer—proprioceptive specialists receive information from visual processing and vice versa—without explicit point-to-point connections between specialists. The feedback is additive (residual) to preserve modality-specific information.

The world model backbone follows the RSSM structure from Dreamer, but with the global workspace state w_t replacing the standard observation encoding. The recurrent dynamics model predicts h_t = GRU(h_{t-1}, w_t, a_{t-1}), and the stochastic latent z_t is sampled from a learned prior/posterior as in standard RSSM. Reward prediction, value estimation, and actor optimization follow standard Dreamer procedures using imagined rollouts in the latent space.

For missing modality handling, the key mechanism is the attention-weighted workspace update. When a modality is missing, its specialist produces a zero or default embedding, and the attention mechanism naturally assigns near-zero weight to this uninformative specialist. The remaining specialists contribute proportionally more to the workspace state, maintaining a meaningful representation for downstream processing. This contrasts with concatenation approaches where a missing modality produces a corrupted feature vector that propagates through the entire model.

Training uses a multi-task loss combining the standard Dreamer objectives (reconstruction, KL, reward prediction) with an auxiliary modality prediction loss: each specialist is trained to predict its modality from the workspace state alone, encouraging the workspace to maintain a rich cross-modal representation. The modality prediction loss weight is annealed from 1.0 to 0.1 over training to prevent interference with the primary RL objectives.

## Key Results & Numbers
- Competitive with standard Dreamer on fully-observed tasks; superior on partially-observed tasks
- Missing modality robustness: 10-20% performance drop with one modality removed vs. 80%+ for concatenation baselines
- 15-30% fewer environment interactions required compared to standard multimodal Dreamer
- Workspace dimensionality: 256-512 (shared across modalities)
- Specialist architectures: ConvNet (vision), MLP (proprioception), PointNet (depth)
- Attention patterns show task-relevant modality switching (e.g., vision dominant during navigation, proprioception during contact)
- Evaluated on locomotion (quadruped), manipulation, and navigation benchmarks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**

The GWT world model is relevant to the Mini Cheetah if the project extends to include exteroceptive sensing (cameras, depth sensors) alongside proprioception. The missing modality robustness is particularly valuable for a real quadruped where cameras can be occluded by the robot's own body during dynamic motions or become unreliable in rain/fog conditions. The proprioceptive specialist would maintain control authority even when vision fails.

For a proprioception-only setup, the GWT architecture provides less benefit over standard RSSM, as there is only one modality. However, the specialist module concept could still be useful for separating different proprioceptive sub-modalities (joint state, IMU, foot contact) and learning their relative importance dynamically through the workspace attention mechanism.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

The GWT architecture is highly relevant to Cassie's Planner level, which must integrate heterogeneous inputs: proprioceptive state, terrain elevation maps, task commands, and potentially visual observations. The workspace mechanism provides a principled way to fuse these diverse inputs for the RSSM world model, replacing ad-hoc concatenation with learned cross-modal attention. The Planner's RSSM backbone can be directly replaced with the GWT-RSSM variant.

The missing modality robustness is critical for Cassie's real-world deployment: if the elevation mapping sensor fails or becomes noisy, the Planner should fall back to proprioceptive planning rather than producing catastrophic predictions. The GWT architecture provides this graceful degradation by design. The attention patterns also offer interpretability—operators can monitor which modality the Planner is attending to at each timestep, aiding debugging and trust calibration.

## What to Borrow / Implement
- Implement the GWT workspace architecture for Cassie's Planner-level multimodal RSSM
- Use separate specialist modules for proprioception, elevation maps, and task embeddings in Cassie's Planner
- Test missing modality robustness by randomly dropping elevation map inputs during training
- Apply the modality prediction auxiliary loss to encourage rich cross-modal workspace representations
- Monitor attention weights during deployment for interpretability and debugging

## Limitations & Open Questions
- Broadcast-and-compete adds computational overhead (cross-attention per timestep) that may impact real-time control at high frequencies
- Soft attention may not fully replicate the binary winner-take-all dynamics of true Global Workspace Theory, potentially limiting emergent reasoning
- Evaluation environments are simulated; real-robot validation with actual sensor failures not yet demonstrated
- Scaling to many modalities (>5) may dilute workspace attention and reduce effectiveness
