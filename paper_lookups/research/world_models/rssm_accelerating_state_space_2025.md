# Accelerating Model-Based Reinforcement Learning with State-Space World Models

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2502.20168)

---

## Abstract Summary
This paper proposes architectural and algorithmic improvements to Recurrent State-Space Models (RSSM) that significantly accelerate model-based reinforcement learning (MBRL). The core insight is that traditional RSSM implementations suffer from sequential bottlenecks during both training and imagination rollouts, limiting their applicability to real-time robotic control. By leveraging parallel scan operations from structured state-space models (S4/S5/Mamba lineage), the authors replace the recurrent transitions in RSSM with state-space model (SSM) variants that can be trained in parallel across time steps while retaining the stochastic latent structure essential for uncertainty-aware planning.

The proposed State-Space World Model (SSWM) maintains the dual deterministic-stochastic latent structure of DreamerV3's RSSM but replaces the GRU-based deterministic path with a diagonal linear SSM that supports parallel associative scans during training. During imagination (planning), the model reverts to a recurrent mode but benefits from the simplified linear dynamics, achieving 3–5× faster rollout speeds. The stochastic component is retained via a learned prior/posterior pair, preserving the model's ability to capture multimodal transitions critical for contact-rich locomotion.

Experiments span DeepMind Control Suite, Atari, and simulated robotic manipulation, demonstrating that SSWM matches or exceeds DreamerV3 in asymptotic performance while reducing wall-clock training time by 40–60%. Ablation studies confirm that the parallel scan training and simplified deterministic backbone are the primary contributors to speedup, while the stochastic latent variables remain essential for tasks with partial observability and contact dynamics.

## Core Contributions
- Replacement of GRU-based deterministic transitions in RSSM with diagonal linear state-space model layers that support parallel scan training
- Demonstration that the dual deterministic-stochastic latent structure can be preserved with SSM backbones without sacrificing model expressiveness
- 3–5× faster imagination rollouts during planning due to simplified linear dynamics in the deterministic path
- 40–60% reduction in wall-clock training time compared to DreamerV3 across standard MBRL benchmarks
- Ablation studies isolating the contributions of parallel training, SSM architecture, and stochastic latent variables
- Analysis of sequence length scaling showing sublinear compute growth vs. quadratic growth in transformer-based world models
- Preliminary sim-to-real transfer results on a robotic manipulation task demonstrating practical applicability

## Methodology Deep-Dive
The architecture builds on the RSSM formulation from Dreamer, which models a latent state as a tuple (h_t, z_t) where h_t is a deterministic recurrent state and z_t is a stochastic latent. In standard RSSM, h_t is computed via a GRU: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), and z_t is sampled from a learned distribution conditioned on h_t. The key modification replaces the GRU with a diagonal linear SSM: h_t = A * h_{t-1} + B * [z_{t-1}; a_{t-1}], where A is a diagonal matrix with learned parameters constrained to have eigenvalues inside the unit disk (via exponential parameterization), and B is a learned input projection. This formulation enables parallel associative scan during training, processing entire sequences in O(L log L) time instead of O(L) sequential steps.

The stochastic component retains the standard RSSM structure: a prior p(z_t | h_t) and posterior q(z_t | h_t, o_t) both parameterized as diagonal Gaussians (or categorical distributions for discrete latents). The KL divergence loss between prior and posterior regularizes the latent space. The posterior is only used during training (when observations are available), while the prior drives imagination rollouts. An important design choice is the use of a "stop-gradient" on h_t when computing the posterior, preventing the observation encoder from dominating the deterministic dynamics learning.

For imagination-based planning (as in Dreamer's actor-critic framework), the model operates in recurrent mode: given an initial latent state, it autoregressively predicts future states using the prior and the SSM transition. The linear dynamics h_t = A * h_{t-1} + B * x_t are computationally cheaper than GRU forward passes (no sigmoid/tanh activations, no input/forget/output gates), yielding the 3–5× imagination speedup. The actor and critic are trained on imagined trajectories using the standard λ-return target from DreamerV3.

The observation encoder and decoder follow DreamerV3's design: a CNN for image observations or MLP for vector observations, mapping to/from the latent space. Reward and continuation predictors are MLPs operating on the concatenated latent [h_t; z_t]. The entire model is trained end-to-end with a composite loss: reconstruction loss + reward prediction loss + continuation prediction loss + KL regularization, using the symlog loss from DreamerV3 for improved gradient scaling.

Sequence length handling is a key advantage: the parallel scan enables efficient training on long sequences (up to 512 steps tested), whereas standard RSSM struggles beyond 50–100 steps due to vanishing/exploding gradients through the GRU. The diagonal SSM's eigenvalue parameterization provides stable long-range gradient flow by construction.

## Key Results & Numbers
- 40–60% wall-clock training time reduction vs. DreamerV3 across DMC, Atari, and robotic manipulation benchmarks
- 3–5× faster imagination rollouts compared to GRU-based RSSM
- Matches DreamerV3 asymptotic return on 15/18 DMC tasks, exceeds on 2 tasks (Walker Run, Humanoid Walk)
- Sublinear O(L log L) training compute scaling with sequence length vs. O(L) sequential for GRU-RSSM
- Supports training on 512-step sequences without gradient degradation (standard RSSM limited to ~50–100 steps)
- Model size comparable to DreamerV3-S (~10M parameters) with SSM backbone
- Preliminary sim-to-real transfer on robotic pushing task achieves 85% success rate vs. 82% for DreamerV3

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For the Mini Cheetah project, which primarily uses model-free PPO with domain randomization, the SSWM offers a potential model-based planning complement. Faster MBRL could enable online model learning during deployment for adaptive locomotion, where the Mini Cheetah learns a local dynamics model to anticipate terrain changes. The long-sequence training capability is particularly relevant for locomotion gaits that have long cycle times or require planning over extended horizons.

However, the primary pipeline (PPO + domain randomization + curriculum learning) is well-established for quadruped sim-to-real, and integrating a full world model adds architectural complexity. The main borrowable insight is the SSM backbone itself: even without full MBRL, an SSM-based state estimator could replace the GRU in the Mini Cheetah's observation encoder for better temporal modeling of proprioceptive history.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is directly applicable to Cassie's Planner component, which uses an RSSM-based world model (Dreamer-style) for high-level trajectory planning. The SSWM improvements translate to: (1) faster Planner training, reducing the overall multi-level training pipeline time; (2) longer planning horizons for the Planner, critical for bipedal locomotion where footstep planning requires looking 1–2 seconds ahead; (3) more stable gradient flow through the world model, improving the quality of imagined trajectories used for actor-critic training.

The parallel scan training is especially valuable for the hierarchical architecture, where the Planner operates at a coarser temporal resolution (e.g., every 10–20 control steps) but must model long-horizon dynamics. The SSM backbone could replace the GRU in Cassie's RSSM without modifying the stochastic latent structure, making it a relatively clean architectural swap. The uncertainty analysis from the stochastic component feeds directly into the CBF-QP safety layer.

## What to Borrow / Implement
- Replace the GRU deterministic backbone in Cassie's RSSM Planner with a diagonal linear SSM using parallel associative scans for training
- Adopt the exponential eigenvalue parameterization for stable long-sequence training of the world model
- Use the 512-step sequence training capability for longer Planner horizons in bipedal footstep planning
- Consider an SSM-based observation encoder for Mini Cheetah's proprioceptive history processing as a lighter alternative to GRU
- Benchmark imagination rollout speed improvement to validate real-time planning feasibility on Cassie hardware

## Limitations & Open Questions
- Preliminary sim-to-real results are limited to simple manipulation; locomotion transfer is not demonstrated
- The diagonal SSM may have limited expressiveness compared to GRU for highly nonlinear contact dynamics (no gating mechanism)
- Stochastic latent training stability with SSM backbone needs further investigation for contact-rich domains
- Integration with hierarchical RL architectures is not explored; the Planner-level world model may have different requirements than single-level Dreamer
