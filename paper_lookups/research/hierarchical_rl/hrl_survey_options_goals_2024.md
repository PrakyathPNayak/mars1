# Hierarchical Reinforcement Learning: From Options to Goal-Conditioned Policies - A Comprehensive Survey

**Authors:** (Survey 2024)
**Year:** 2024 | **Venue:** Survey
**Links:** [RLJ Club](https://rljclub.github.io/posts/hierarchical-reinforcement-learning/)

---

## Abstract Summary
This comprehensive survey traces the evolution of hierarchical reinforcement learning (HRL) from its theoretical foundations in the options framework through feudal networks to modern goal-conditioned and skill-based approaches. The survey covers three decades of HRL research, organizing the field into four paradigms: (1) the options framework and its extensions (Option-Critic, termination critic, deep options), (2) feudal/manager-worker architectures (FeUdal Networks, HIRO, HAM), (3) unsupervised skill discovery (DIAYN, DADS, VIC, successor features), and (4) goal-conditioned HRL (HER, RIG, MEGA, universal value functions). For each paradigm, the survey provides theoretical foundations, algorithmic details, key results, and applications to locomotion, manipulation, and navigation.

A central theme is the trade-off between hierarchy expressiveness and training stability. Deep option architectures (Option-Critic) provide a theoretically grounded end-to-end training signal but suffer from option collapse (all options converge to the same behavior). Feudal architectures provide strong hierarchy but require manual reward shaping at each level. Unsupervised skill discovery produces diverse behaviors but without guarantees of task relevance. Goal-conditioned approaches bridge hierarchy and task specification but require careful goal space design. The survey identifies hybrid approaches — combining skill discovery with goal conditioning, or options with feudal structure — as the most promising direction.

The survey dedicates significant attention to locomotion applications, reviewing how each HRL paradigm has been applied to quadruped and bipedal locomotion. It identifies key challenges specific to locomotion HRL: temporal abstraction at gait-cycle timescales, contact-aware skill boundaries, and the interaction between hierarchical decisions and physical stability constraints. The concluding section identifies open problems including scalable hierarchy learning, multi-task locomotion, and sim-to-real transfer for hierarchical policies.

## Core Contributions
- Comprehensive taxonomy of HRL methods across four paradigms: options, feudal, skill discovery, and goal-conditioned approaches
- Unified mathematical framework connecting options (ω, π_ω, β_ω) to feudal (manager-worker) to goal-conditioned (π(a|s,g)) formulations
- Detailed analysis of Option-Critic architecture including the option-value gradient theorem and termination gradient
- Survey of unsupervised skill discovery methods (DIAYN, DADS, VIC) with theoretical connections to mutual information maximization
- Comparative analysis of HRL applications to locomotion: quadruped gaits, bipedal walking, multi-terrain adaptation
- Identification of option collapse problem and review of mitigation strategies (diversity bonuses, information-theoretic regularization)
- Analysis of open problems: scalable hierarchy, multi-task locomotion, sim-to-real for hierarchical policies

## Methodology Deep-Dive
The options framework defines a semi-MDP where options ω ∈ Ω are temporally extended actions. Each option is a tuple (I_ω, π_ω, β_ω) where I_ω ⊆ S is the initiation set, π_ω(a|s) is the intra-option policy, and β_ω(s) ∈ [0,1] is the termination probability. The Option-Critic architecture (Bacon et al., 2017) learns options end-to-end using two key gradients: (1) the intra-option policy gradient: ∇θ J = E[∇θ log π_ω(a|s) Q_Ω(s, ω, a)], where Q_Ω is the option-value function; (2) the termination gradient: ∇φ J = -E[∇φ β_ω(s') A_Ω(s', ω)], where A_Ω(s', ω) = Q_Ω(s', ω) - V_Ω(s') is the option advantage. The termination gradient encourages option termination when the current option's value is lower than the expected value under re-selection.

The survey identifies option collapse as a critical failure mode: without diversity pressure, all options converge to the same policy because the option-value gradient does not penalize redundancy. Mitigation strategies include: (a) DIAYN-style mutual information bonus I(z; s') encouraging distinguishable options, (b) deliberation cost penalizing frequent option switching, and (c) successor feature diversity requiring options to reach distinct future state distributions.

DIAYN (Diversity is All You Need) learns skills by maximizing I(z; s) = H(z) - H(z|s) where z is a categorical skill variable. The skill policy π(a|s, z) is trained with a pseudo-reward r_DIAYN = log q(z|s) - log p(z), where q(z|s) is a learned discriminator and p(z) is the skill prior (uniform). This encourages each skill z to visit distinctive states that are identifiable by the discriminator. DADS (Dynamics-Aware Discovery of Skills) extends DIAYN by maximizing I(z; s'|s) — skill-conditional next-state mutual information — which produces skills that are not only distinguishable but also predictable in their dynamics, enabling model-based planning over skill sequences.

Goal-conditioned HRL introduces a manager-worker decomposition where the manager sets subgoals g ∈ G and the worker executes goal-conditioned policies π(a|s, g). HIRO (Data-Efficient HRL) trains the manager to set subgoals in observation space using off-policy relabeling: the manager's transitions are relabeled with the goal that would have been most likely given the worker's actual behavior, enabling off-policy training. Universal Value Functions (UVFs) V(s, g) generalize value functions to goal-conditioned settings, enabling zero-shot generalization to new goals at test time.

For locomotion applications, the survey identifies the gait-cycle as a natural temporal abstraction boundary: options/skills correspond to distinct gaits (walk, trot, gallop, bound), and the manager/option-selector operates at the gait-switching timescale (~0.5–2s). Contact events provide natural option termination signals, as the transition from swing to stance phase represents a fundamental change in dynamics. The survey notes that hierarchical locomotion policies outperform flat policies primarily on multi-terrain tasks where different gaits are optimal for different terrain types.

The survey compares training complexity across paradigms: Option-Critic has the lowest implementation complexity but highest collapse risk. Feudal architectures require multi-level reward design but provide stable training. DIAYN/DADS pre-training followed by fine-tuning offers a clean separation of concerns but may discover irrelevant skills. Goal-conditioned HRL with HER provides the best sample efficiency for goal-reaching tasks but requires meaningful goal space design.

## Key Results & Numbers
- Option-Critic: 4–8 options sufficient for locomotion tasks; >8 options leads to increased collapse without diversity pressure
- DIAYN: typically discovers 10–50 distinguishable skills; 60–80% are locomotion-relevant on quadruped platforms
- DADS: dynamics predictability enables 2–5× better planning performance compared to DIAYN skills
- Goal-conditioned HRL (HIRO): 30–50% sample efficiency improvement over flat PPO on multi-terrain locomotion
- Option collapse mitigation with MI bonus: recovers 85–95% of distinct option utilization
- Feudal architectures: require 2–3× more hyperparameter tuning than flat baselines but achieve 20–40% performance improvement on complex tasks
- Successor feature diversity: most theoretically principled collapse prevention but computationally expensive (2× training cost)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For the Mini Cheetah's primarily flat-policy PPO approach, the HRL survey provides context for potential architectural enhancements. If the Mini Cheetah needs to handle multiple terrain types or switch between gaits, an Option-Critic layer above the current PPO policy could enable gait selection based on terrain context. The gait-cycle temporal abstraction aligns with Mini Cheetah's ~0.3–0.5s gait period. However, the current project scope (single PPO policy with domain randomization) may not require full hierarchical control.

The survey's analysis of option collapse and mitigation strategies is relevant if hierarchical extensions are considered, providing a roadmap of known pitfalls and solutions.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This survey is a critical reference for Cassie's 4-level hierarchy. The Option-Critic framework directly underlies the Planner→Primitives level, where the Planner selects among locomotion primitives and each primitive has learned termination conditions. DIAYN/DADS are used for Cassie's unsupervised primitive discovery, and the survey's comparison of these methods (DADS providing dynamics predictability for planning) directly informs the choice between them.

The goal-conditioned HRL concepts apply to Cassie's Planner, which can be viewed as a goal-conditioned manager setting subgoals for the Primitive Selector. The survey's identification of gait-cycle temporal abstraction as a natural hierarchy boundary validates Cassie's architectural design. The option collapse analysis is essential for ensuring Cassie's discovered primitives remain diverse during training.

## What to Borrow / Implement
- Use Option-Critic's termination gradient for learning primitive termination conditions in Cassie's Planner→Primitives interface
- Implement DIAYN with MI bonus for Cassie's unsupervised primitive discovery, with DADS as an alternative for dynamics-predictable skills
- Apply HIRO-style off-policy goal relabeling if Cassie's Planner uses goal-conditioned primitive selection
- Use the 4–8 option guideline for Mini Cheetah if adding a gait-selection layer, with MI diversity bonus to prevent collapse
- Consider successor feature diversity as a principled option collapse prevention mechanism for Cassie's primitive library

## Limitations & Open Questions
- Survey coverage of sim-to-real transfer for hierarchical policies is limited; most HRL results are simulation-only
- The interaction between hierarchical structure and safety constraints (e.g., CBF-QP) is not addressed in the survey
- Scalability of HRL beyond 2–3 hierarchy levels is theoretically discussed but not empirically validated in the surveyed works
- The computational overhead of training multiple hierarchy levels simultaneously (end-to-end) vs. sequentially is not systematically compared
