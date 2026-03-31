# On Uncertainty in Deep State Space Models for Model-Based Reinforcement Learning

**Authors:** (2024)
**Year:** 2024 | **Venue:** OpenReview
**Links:** [OpenReview](https://openreview.net/forum?id=UQXdQyoRZh)

---

## Abstract Summary
This paper provides a rigorous analysis of uncertainty estimation in deep state-space models (SSMs) used for model-based reinforcement learning, with a focus on the RSSM architecture underlying the Dreamer family of algorithms. The central finding is that standard RSSM implementations systematically overestimate aleatoric (irreducible) uncertainty while underestimating epistemic (reducible) uncertainty, leading to suboptimal planning behavior. The overestimation arises from the KL-regularized training objective, which encourages the stochastic latent prior to maintain high entropy even in deterministic transition regions, conflating model uncertainty about dynamics with genuine environmental stochasticity.

The authors propose a decomposed uncertainty framework that separates aleatoric and epistemic components within the RSSM latent space. Epistemic uncertainty is captured via an ensemble of posterior networks (each encoding a different hypothesis about the dynamics), while aleatoric uncertainty is estimated from a single, shared stochastic latent with a tighter KL constraint. This decomposition enables principled uncertainty-aware planning: the agent can be optimistic under epistemic uncertainty (exploration) while being conservative under aleatoric uncertainty (risk-aversion), or vice versa depending on the task.

Experiments on DeepMind Control Suite and robotic locomotion tasks demonstrate that the proposed decomposition improves both sample efficiency and asymptotic performance. Notably, tasks with mixed deterministic and stochastic dynamics (e.g., locomotion on uneven terrain) benefit most, as the agent correctly attributes contact uncertainty to aleatoric noise while treating novel terrain as epistemic uncertainty worthy of cautious exploration.

## Core Contributions
- Identification and formal analysis of aleatoric uncertainty overestimation in standard RSSM training due to KL regularization
- Proposed decomposed uncertainty framework separating epistemic and aleatoric components within the RSSM latent space
- Ensemble-based epistemic uncertainty estimation via multiple posterior networks sharing a common deterministic backbone
- Tighter KL constraint formulation that reduces aleatoric overestimation while preserving latent space expressiveness
- Principled uncertainty-aware planning strategies: optimistic (exploration) vs. conservative (safety) based on uncertainty type
- Empirical demonstration of improved sample efficiency on locomotion tasks with mixed deterministic-stochastic dynamics
- Analysis of how uncertainty decomposition interacts with reward prediction and value function learning

## Methodology Deep-Dive
The standard RSSM in DreamerV2/V3 models the generative process as: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) (deterministic), z_t ~ p(z_t | h_t) (prior), z_t ~ q(z_t | h_t, o_t) (posterior). The training objective includes a KL divergence term KL(q(z_t | h_t, o_t) || p(z_t | h_t)) that regularizes the posterior toward the prior. The authors show that this KL term acts as an implicit entropy maximizer on the prior: when the true dynamics are nearly deterministic, the prior should collapse to a near-delta distribution, but the KL penalty prevents this by penalizing low-entropy priors. This creates "phantom" stochasticity that degrades planning rollouts.

The decomposed framework introduces K posterior networks {q_k(z_t | h_t, o_t)}_{k=1}^K sharing a single deterministic backbone f. Epistemic uncertainty is estimated as the disagreement (variance) across the K posterior predictions: U_epistemic(h_t) = Var_k[μ_k(h_t, o_t)]. Aleatoric uncertainty is the average posterior variance: U_aleatoric(h_t) = E_k[σ_k²(h_t, o_t)]. The key insight is that ensemble disagreement vanishes with more data (epistemic uncertainty reduces), while average variance converges to the true noise level (aleatoric uncertainty persists).

The modified training objective replaces the standard KL term with a "free-bits" variant that allows the prior to have arbitrarily low entropy when the data supports it: KL_modified = max(KL(q || p), λ_free), where λ_free is a small constant (e.g., 0.1 nats). This prevents the KL term from forcing artificial stochasticity. Additionally, a mutual information regularizer I(z_t; o_t | h_t) encourages the stochastic latent to capture genuinely stochastic aspects of observations rather than deterministic patterns already captured by h_t.

For planning, the authors propose two strategies: (1) Optimistic Under Epistemic Uncertainty (OUEU): select actions that maximize expected reward plus a bonus proportional to epistemic uncertainty, encouraging exploration of uncertain regions; (2) Conservative Under Aleatoric Uncertainty (CUAU): penalize rewards in high aleatoric uncertainty regions, promoting risk-averse behavior in genuinely stochastic environments. These strategies can be combined, yielding an agent that explores novel situations while being cautious in inherently unpredictable ones.

The ensemble training is efficient due to shared backbone: only the posterior head is replicated K times (K=5 in experiments), adding roughly 15% parameter overhead. During imagination, the ensemble provides uncertainty estimates at negligible cost since the deterministic backbone (the computational bottleneck) is run only once.

## Key Results & Numbers
- 20–35% improvement in sample efficiency on DMC locomotion tasks (Walker, Hopper, Humanoid) with decomposed uncertainty
- Aleatoric uncertainty overestimation reduced by 60–80% compared to standard RSSM on deterministic benchmarks (CartPole, Pendulum)
- Ensemble disagreement (epistemic uncertainty) correctly decreases with more training data, reaching near-zero on well-explored regions
- Planning with CUAU strategy improves robustness to stochastic perturbations: 15% higher return on noisy-terrain Walker
- 5-member ensemble adds only 15% parameter overhead with shared backbone
- Free-bits KL modification alone (without ensemble) provides 10–15% performance improvement, indicating aleatoric overestimation is a significant issue
- Asymptotic performance matches or exceeds DreamerV3 on 12/15 DMC tasks while providing calibrated uncertainty estimates

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For the Mini Cheetah's primarily model-free PPO approach, the direct RSSM uncertainty analysis is less immediately applicable. However, the uncertainty decomposition framework is relevant if model-based components are added for terrain adaptation or online dynamics learning. The finding that KL regularization overestimates aleatoric uncertainty is particularly relevant for any Mini Cheetah state estimator using variational autoencoders or latent dynamics models. The conservative planning under aleatoric uncertainty (CUAU) strategy could inform safe exploration during sim-to-real transfer.

Domain randomization in Mini Cheetah training introduces artificial aleatoric uncertainty; understanding how the model handles this vs. genuine epistemic uncertainty about the real robot could improve domain randomization parameter selection.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is directly critical to Cassie's Planner component, which uses an RSSM-based world model. The uncertainty decomposition addresses a core challenge: the Planner must distinguish between genuine terrain stochasticity (e.g., loose gravel, aleatoric) and model uncertainty about novel terrain (epistemic). This distinction feeds directly into the CBF-QP safety layer — aleatoric uncertainty should trigger conservative control constraints, while epistemic uncertainty should trigger cautious exploration.

The free-bits KL modification is a low-cost improvement applicable to Cassie's RSSM training immediately. The ensemble posterior approach, with shared backbone and 15% overhead, is practical for the Planner's computational budget. The CUAU planning strategy aligns with the safety-critical nature of bipedal locomotion, where the consequences of falls are severe. The decomposed uncertainty estimates can also serve as inputs to the CBF constraint tightening: higher aleatoric uncertainty → wider safety margins in the control barrier function.

## What to Borrow / Implement
- Adopt the free-bits KL modification for Cassie's RSSM training to reduce aleatoric uncertainty overestimation
- Implement ensemble posteriors (K=5) with shared deterministic backbone for principled epistemic uncertainty estimation in the Planner
- Use decomposed uncertainty estimates as inputs to CBF-QP constraint tightening: aleatoric → wider safety margins, epistemic → exploration bonuses
- Apply CUAU planning strategy for Cassie's safety-critical locomotion planning on uncertain terrain
- Investigate aleatoric uncertainty overestimation in Mini Cheetah's domain randomization training to improve randomization parameter selection

## Limitations & Open Questions
- Ensemble training with K=5 posteriors may increase training instability in early stages when the backbone is rapidly changing
- The free-bits threshold λ_free is a sensitive hyperparameter; too low allows overestimation, too high suppresses useful stochasticity in contact dynamics
- Analysis is limited to single-level MBRL; interaction with hierarchical planning (where upper levels see coarser dynamics) is unexplored
- Computational cost of ensemble disagreement computation during real-time planning on embedded hardware (e.g., Cassie's onboard compute) may be prohibitive
