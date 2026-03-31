# Diversity Is All You Need: Learning Skills without a Reward Function

**Authors:** Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine
**Year:** 2019 | **Venue:** ICLR
**Links:** [arXiv](https://arxiv.org/abs/1802.06070)

---

## Abstract Summary
DIAYN (Diversity Is All You Need) introduces an information-theoretic framework for learning diverse, distinguishable skills without any extrinsic reward signal. The method maximizes the mutual information between a latent skill variable z (sampled from a fixed prior) and the states visited by the policy conditioned on z. This encourages the agent to learn a repertoire of behaviors where each skill reliably visits distinct regions of the state space.

The key insight is that maximizing I(S; Z) — the mutual information between states and skills — can be decomposed into maximizing the entropy of states H(S) (encouraging exploration) while minimizing the conditional entropy H(S|Z) (ensuring each skill is predictable and distinguishable). A learned discriminator q_φ(z|s) approximates the posterior over skills given states, and the log-probability from this discriminator serves as the intrinsic reward. The resulting skills emerge as diverse locomotion primitives, navigational strategies, and manipulation behaviors entirely from unsupervised interaction.

The paper demonstrates that DIAYN-discovered skills can be used as primitives for hierarchical reinforcement learning, composed for downstream tasks, and fine-tuned with sparse task rewards — all without any reward engineering during the skill discovery phase.

## Core Contributions
- Formalized unsupervised skill discovery as mutual information maximization between a discrete latent variable z and visited states s
- Derived a tractable variational lower bound on I(S; Z) using a learned discriminator q_φ(z|s) as intrinsic reward
- Demonstrated that diverse locomotion gaits (hopping, walking, turning) emerge without any reward function on MuJoCo tasks
- Showed that discovered skills transfer to downstream tasks via hierarchical RL, where a meta-controller selects among learned skills
- Introduced the use of a fixed uniform prior p(z) to encourage balanced skill utilization and prevent mode collapse
- Established DIAYN as a foundational unsupervised RL primitive that decouples skill acquisition from task specification
- Provided analysis connecting DIAYN to options framework, empowerment, and intrinsic motivation literatures

## Methodology Deep-Dive
DIAYN samples a skill index z ~ p(z) = Uniform(1, ..., K) at the start of each episode and conditions the policy π_θ(a|s, z) on this latent throughout. The intrinsic reward at each timestep is r(s, z) = log q_φ(z|s) − log p(z), where q_φ(z|s) is a learned discriminator (a neural network classifier mapping states to skill probabilities). This reward is high when the discriminator can confidently identify which skill z produced state s, incentivizing each skill to produce distinctive state visitations.

The objective decomposes the mutual information I(S; Z) = H(Z) − H(Z|S). Since Z is drawn from a fixed uniform prior, H(Z) is constant and maximized. The optimization therefore focuses on minimizing H(Z|S) by training q_φ(z|s) to accurately predict the active skill from observed states. The variational bound yields: I(S; Z) ≥ E_{z~p(z), s~π(z)} [log q_φ(z|s) − log p(z)], which is exactly the intrinsic reward signal.

Training alternates between: (1) updating the policy π_θ using SAC (Soft Actor-Critic) with the intrinsic discriminator reward, and (2) updating the discriminator q_φ via cross-entropy classification loss on (state, skill) pairs from the replay buffer. The soft actor-critic backbone provides maximum entropy exploration within each skill, further encouraging state-space coverage. The discriminator is conditioned on the full state observation (or a subset, such as x-y position for navigation tasks), which determines the nature of the discovered skills.

A critical design choice is that the discriminator operates on states rather than state-action pairs. This encourages skills to be distinguished by their outcomes (where the agent goes) rather than their mechanisms (how the agent moves), yielding more semantically meaningful and transferable primitives. The number of skills K is a hyperparameter, typically set between 8 and 50, with diminishing returns at very high values due to the difficulty of maintaining discriminability.

For hierarchical composition, a meta-policy π_meta(z|s) is trained on top of the fixed skill policies, selecting which skill z to execute for a fixed number of timesteps before re-selecting. This two-level hierarchy enables long-horizon task completion using the discovered skill repertoire as a discrete action space.

## Key Results & Numbers
- Discovered 8–20 distinct locomotion skills on Half-Cheetah, Ant, and Hopper environments without any reward
- Half-Cheetah skills included forward/backward running at different speeds and directions
- Ant skills covered locomotion in all cardinal directions and spinning behaviors
- Hierarchical DIAYN achieved comparable performance to flat RL on navigation tasks with sparse rewards
- Skill-augmented policy solved sparse-reward Ant navigation in ~50% fewer environment steps than flat SAC
- Discriminator accuracy exceeded 90% across all tested environments, confirming skill distinguishability
- Fine-tuning a discovered skill with task reward reached near-optimal performance 5× faster than training from scratch

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
DIAYN provides a principled framework for discovering diverse locomotion gaits for the Mini Cheetah without manually engineering reward functions for each gait. The discovered skill repertoire could include trotting, bounding, pronking, and turning primitives that form a gait library. However, vanilla DIAYN on high-dimensional quadruped state spaces may suffer from skill degeneracy (many skills converging to similar behaviors) without additional structural priors. The method's strength lies in bootstrapping an initial gait repertoire that can be refined with task-specific rewards.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
DIAYN is explicitly used in Project B's Primitives level for unsupervised skill discovery on Cassie. The discriminator-based intrinsic reward drives the low-level policy to learn diverse bipedal locomotion skills — different walking speeds, turning gaits, and balancing behaviors. The discrete skill index z maps directly to the Primitives module's latent space. Understanding DIAYN's reward formulation, discriminator architecture, and failure modes (skill collapse, state-space coverage issues) is essential for debugging and improving the Primitives level. The hierarchical composition mechanism (meta-policy selecting skills) directly informs the Planner level's skill-selection strategy.

## What to Borrow / Implement
- Implement the discriminator architecture q_φ(z|s) as a 3-layer MLP classifier with softmax output for skill prediction on robot proprioceptive states
- Use SAC as the backbone optimizer with the discriminator log-probability as intrinsic reward: r = log q_φ(z|s) − log p(z)
- Start with K=16 skills for Cassie (Project B) and K=10 for Mini Cheetah (Project A), tuning based on discriminator accuracy
- Apply the state-based (not action-based) discriminator design to ensure skills are distinguished by locomotion outcomes
- Adopt the hierarchical skill composition mechanism for the Planner level in Project B, with a meta-policy selecting skill indices at fixed intervals

## Limitations & Open Questions
- Vanilla DIAYN suffers from skill degeneracy in high-dimensional state spaces — many skills collapse to similar behaviors, especially for complex robots like quadrupeds and bipeds
- The fixed uniform prior p(z) does not account for the varying difficulty or utility of different skills, potentially wasting capacity on trivial distinctions
- Discriminator conditioning on full state may lead to skills distinguished by irrelevant state dimensions rather than meaningful locomotion differences
- No dynamics awareness — skills are not optimized for predictability or composability, limiting model-based planning (addressed by DADS)
