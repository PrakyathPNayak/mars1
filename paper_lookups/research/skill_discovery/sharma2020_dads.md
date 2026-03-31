# Dynamics-Aware Discovery of Skills (DADS)

**Authors:** Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, Karol Hausman
**Year:** 2020 | **Venue:** ICLR
**Links:** [OpenReview](https://openreview.net/forum?id=HJgLZR4KvH)

---

## Abstract Summary
DADS reformulates unsupervised skill discovery by maximizing the conditional mutual information I(s'; z | s) between the next state s' and the skill latent z, conditioned on the current state s. Unlike DIAYN which maximizes I(S; Z) encouraging skills to visit different absolute state regions, DADS encourages skills to induce predictable and diverse state transitions. This dynamics-awareness is the key innovation: skills are not just distinguishable but their effects on the environment are predictable, enabling model-based planning in the learned skill space.

The method simultaneously learns a skill-conditioned dynamics model p_θ(s' | s, z) that predicts the next state given the current state and active skill. This dynamics model serves dual purposes: its prediction error provides the intrinsic reward signal (encouraging the policy to produce transitions that are predictable given z), and it can be used at test time for model-based planning — composing skills in sequence to reach arbitrary goals without any additional training.

DADS demonstrates that dynamics-aware skills are qualitatively more useful than DIAYN skills: they cover the state space more uniformly, support zero-shot goal-reaching via planning, and transfer more effectively to downstream tasks. On the Ant environment, DADS achieves state-of-the-art zero-shot goal-reaching performance by planning through the learned skill-dynamics model using model predictive control (MPC).

## Core Contributions
- Introduced the conditional MI objective I(s'; z | s) for skill discovery, shifting from state-occupancy diversity to transition diversity
- Simultaneously learned a skill-conditioned dynamics model p_θ(s' | s, z) that enables model-based planning over discovered skills
- Demonstrated zero-shot goal-reaching via model predictive control (MPC) in the learned skill space without any goal-conditioned training
- Showed that continuous skill spaces (z ∈ R^d) outperform discrete spaces for locomotion, enabling smooth interpolation between skills
- Provided theoretical analysis connecting DADS to empowerment and mutual-information-based exploration
- Achieved superior downstream task performance compared to DIAYN through better state-space coverage and skill predictability
- Established a framework where skill discovery and dynamics learning are unified in a single training loop

## Methodology Deep-Dive
DADS maximizes the conditional mutual information I(s'; z | s) = H(s' | s) − H(s' | s, z). The first term H(s' | s) encourages diverse next-state transitions across different skills (high marginal transition entropy), while the second term −H(s' | s, z) ensures that conditioned on a specific skill z, the transitions are predictable (low conditional transition entropy). This is approximated using a variational bound with a learned dynamics model q_ω(s' | s, z) that approximates the true conditional p(s' | s, z).

The intrinsic reward becomes: r(s, z, s') = log q_ω(s' | s, z) − log p(s' | s), where the first term rewards transitions that are well-predicted by the skill-conditioned model, and the second term (often approximated or dropped) penalizes transitions that are too common regardless of skill. In practice, the dynamics model q_ω is parameterized as a Gaussian distribution over Δs = s' − s, predicting the state difference rather than absolute next state for numerical stability.

The skill latent z is sampled from a continuous prior z ~ N(0, I) (in contrast to DIAYN's discrete uniform). Continuous skills enable smooth interpolation: z = 0.5·z_forward + 0.5·z_right yields a diagonal movement skill. The policy π_θ(a | s, z) is trained via SAC with the intrinsic reward, while the dynamics model q_ω(Δs | s, z) is trained via maximum likelihood on collected transitions.

At test time for zero-shot goal-reaching, the learned dynamics model enables model predictive control (MPC): given a goal state s_goal, the planner samples skill sequences {z_1, ..., z_T}, rolls out predicted trajectories using q_ω, and selects the sequence minimizing distance to s_goal. This planning occurs entirely in the skill space without any additional environment interaction or policy fine-tuning. The cross-entropy method (CEM) is used as the MPC optimizer, iterating elite sampling to refine skill sequences.

A critical implementation detail is the choice of state features for the dynamics model. For locomotion, DADS typically uses the center-of-mass x-y position as the prediction target, which biases skills toward diverse locomotion directions. The dynamics model architecture is a 3-layer MLP with 256 hidden units, outputting mean and diagonal covariance of the Gaussian transition distribution.

## Key Results & Numbers
- Ant zero-shot goal-reaching: DADS achieved 80%+ success rate within 3m radius goals using MPC planning, compared to 40% for DIAYN
- Discovered continuous skill manifold on Ant covering 360° of locomotion directions with smooth interpolation
- Half-Cheetah skills exhibited predictable velocity profiles enabling precise speed control via skill selection
- MPC planning with 20-step horizon and 512 CEM samples achieved real-time skill composition
- Dynamics model prediction error < 5% over 10-step rollouts on Ant locomotion
- Downstream task fine-tuning from DADS initialization converged 3× faster than from DIAYN initialization
- State-space coverage measured by trajectory entropy was 1.5× higher for DADS vs. DIAYN on Ant

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
DADS offers dynamics-aware skill discovery for Mini Cheetah, where predictable locomotion primitives are more useful than merely diverse ones. The learned dynamics model could enable model-based planning for navigation — composing trotting, turning, and speed-change skills to reach waypoints without additional training. However, the high-dimensional state space of a real quadruped and sim-to-real transfer challenges may limit the accuracy of the learned dynamics model. The continuous skill space is well-suited for smooth gait interpolation (e.g., blending walk and trot speeds).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
DADS is explicitly used in Project B's Primitives level alongside DIAYN for unsupervised skill discovery on Cassie. The dynamics-aware objective ensures that Cassie's learned locomotion primitives have predictable state transition profiles, which is essential for the Planner level's model-based planning. The learned skill-conditioned dynamics model q_ω(s'|s,z) directly feeds into the Planner's MPC module for composing skills to achieve high-level goals. Understanding the continuous skill space parameterization, dynamics model architecture, and MPC planning procedure is critical for implementing and debugging Project B's hierarchical architecture. The zero-shot composition capability means new tasks can be addressed by re-planning over existing skills rather than retraining.

## What to Borrow / Implement
- Implement the conditional MI objective with a Gaussian dynamics model predicting Δs = s' − s conditioned on (s, z) for skill-aware training on Cassie
- Use continuous skill latents z ~ N(0, I) with d=4 or d=8 dimensions for smooth skill interpolation
- Deploy CEM-based MPC planning over the learned dynamics model for zero-shot goal-reaching at the Planner level in Project B
- Combine DADS and DIAYN objectives as complementary losses: DIAYN for initial skill diversity + DADS for dynamics predictability refinement
- Use x-y-yaw as the dynamics model prediction target for bipedal locomotion to bias skills toward useful navigation primitives

## Limitations & Open Questions
- Dynamics model accuracy degrades for long-horizon rollouts (>15 steps), limiting MPC planning horizon and reliability for complex maneuvers
- Continuous skill spaces can be harder to interpret than discrete skills — requires post-hoc visualization to understand what each region of z-space corresponds to
- The Gaussian dynamics model assumption may not capture multi-modal transitions (e.g., a skill that sometimes turns left and sometimes right depending on terrain)
- Computational cost of MPC planning at test time may be prohibitive for real-time control on physical robots without careful optimization
