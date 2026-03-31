# Temporal Abstraction and the Options Framework: How Agents Learn to Think in Subgoals

**Authors:** (Survey 2024)
**Year:** 2024 | **Venue:** Blog/Survey
**Links:** [Muthu's Notes – Temporal Abstraction and Options Framework](https://notes.muthu.co/2026/02/temporal-abstraction-and-the-options-framework-how-agents-learn-to-think-in-subgoals/)

---

## Abstract Summary
This comprehensive survey traces the evolution of the options framework for temporal abstraction in reinforcement learning, from the foundational work of Sutton, Precup, and Singh (1999) through modern extensions including the Option-Critic architecture, interest functions, and deep HRL variants. The survey provides a unified treatment of how agents can learn to decompose complex tasks into reusable subroutines (options) that operate at multiple temporal scales, enabling efficient planning and transfer.

The core thesis is that temporal abstraction through options addresses fundamental limitations of flat RL: long credit assignment horizons, poor exploration in sparse-reward environments, and inability to reuse learned behaviors across tasks. The survey covers the mathematical foundations of Semi-Markov Decision Processes (SMDPs), the call-and-return execution model, option termination conditions, and the policy-over-options framework, providing both theoretical grounding and practical implementation guidance.

Particular attention is paid to the Option-Critic algorithm and its variants, which enable end-to-end learning of all option components (intra-option policies, termination functions, and policy-over-options) from a single reward signal. The survey discusses common failure modes—option collapse, degenerate termination, and the exploration-exploitation trade-off in option selection—and surveys proposed solutions from recent literature. Applications in locomotion are highlighted throughout.

## Core Contributions
- Provides a unified mathematical treatment of the options framework from SMDP foundations through modern deep RL implementations
- Traces the evolution from hand-crafted options (Sutton et al., 1999) to fully learned options (Option-Critic, 2017) to current state-of-the-art
- Catalogues common failure modes in option learning (collapse, degenerate termination, lack of diversity) with proposed solutions
- Surveys interest functions and their role in constraining option initiation for more structured hierarchical behavior
- Covers the relationship between options, macro-actions, skills, and other temporal abstraction formalisms
- Discusses applications of the options framework to continuous control and locomotion tasks
- Provides practical implementation guidance for Option-Critic with PPO-based optimization

## Methodology Deep-Dive
The survey begins with the **formal definition of options**: an option o = (I, π, β) consists of an initiation set I ⊆ S (states where the option can be started), an intra-option policy π: S × A → [0,1] (the action policy executed while the option is active), and a termination function β: S → [0,1] (the probability of terminating the option in each state). Options extend primitive actions by operating over multiple timesteps, creating a Semi-Markov Decision Process (SMDP) where the time between decisions is variable.

The **call-and-return model** governs option execution: when an option o is selected by the policy-over-options πΩ, the intra-option policy π_o generates actions until the termination function β_o triggers (with probability β_o(s) at each state s), at which point control returns to πΩ for new option selection. The value function under options decomposes as: V_Ω(s) = Σ_o πΩ(o|s)·Q_Ω(s,o), where Q_Ω(s,o) accounts for the expected cumulative reward over the option's duration plus the continuation value.

The **Option-Critic architecture** (Bacon, Harb, Precup, 2017) provides policy gradient theorems for learning all option components simultaneously. The intra-option policy gradient is: ∇θ J ∝ E[∇θ log π_o(a|s)·Q_U(s,o,a)], where Q_U is the option-action value function. The termination gradient is: ∇φ J ∝ E[∂β_o(s')/∂φ·(V_Ω(s') - Q_Ω(s',o))], which increases termination probability when switching options would increase value and decreases it otherwise. This gradient can be unstable—a small advantage for switching can cause premature termination.

The survey covers **modern solutions to option learning pathologies**: (1) **Deliberation cost** adds a penalty ξ for switching options, stabilizing termination by requiring a sufficient advantage before switching; (2) **Diversity-promoting objectives** add mutual information terms I(s;o) or entropy bonuses to prevent option collapse; (3) **Interest functions** Io(s) replace binary initiation sets with continuous functions that modulate option availability based on state, enabling more flexible hierarchical structures; (4) **Proximal Policy Option-Critic (PPOC)** combines Option-Critic with PPO's clipping mechanism for more stable updates to all option components.

The **deep RL implementation** section covers practical considerations: shared vs. separate networks for intra-option policies, parameterization of termination functions (sigmoid for continuous, Bernoulli for discrete), batch option updates, and option-conditional experience replay. The locomotion applications section discusses how options map to locomotion skills: different gaits as distinct options, recovery behaviors as safety options, and terrain-adaptive strategies as environment-conditioned options.

## Key Results & Numbers
- Option-Critic achieves competitive or superior performance to flat PPO on MuJoCo locomotion with 4–8 options
- Deliberation cost ξ=0.01 typically stabilizes option duration to 10–50 steps on continuous control tasks
- Diversity-promoting objectives increase option utilization from ~30% to ~80% (fraction of distinct options actively used)
- Interest functions reduce option space complexity by 40–60% through state-dependent option availability
- PPOC improves training stability by 30% (measured by variance of returns across seeds) compared to vanilla Option-Critic
- Options framework enables 2–4× faster transfer to modified tasks compared to flat policies through option reuse
- Sample efficiency of hierarchical option-based methods matches flat methods on simple tasks and exceeds them by 2–5× on complex, multi-phase tasks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The options framework provides the theoretical foundation for structuring Mini Cheetah's locomotion behaviors as a hierarchy of skills. If the project extends beyond single-gait velocity tracking to include multiple behaviors (walking, trotting, bounding, recovery, stair climbing), the options framework offers a principled way to organize these skills with automatic switching. The temporal abstraction reduces the effective planning horizon, potentially improving training efficiency for complex multi-phase tasks.

For the current PPO-based approach, the survey's practical guidance on implementing Option-Critic with PPO (PPOC) provides a concrete upgrade path if hierarchical control becomes necessary. The deliberation cost mechanism is particularly relevant for ensuring gaits persist for complete cycles rather than switching mid-stride.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This survey is foundational for Cassie's Primitives level, which directly uses the Option-Critic framework. The comprehensive coverage of Option-Critic, its failure modes, and modern solutions provides essential background for the project's implementation. The discussion of option collapse is directly relevant—Cassie's locomotion primitives must remain diverse and not all converge to basic walking. The deliberation cost mechanism should be implemented to ensure Cassie's primitives persist for gait-cycle-appropriate durations.

The interest function concept maps naturally to Cassie's safety-aware option initiation: certain primitives should only be available in specific states (e.g., recovery primitives activated only when instability is detected, running primitives only when balance is stable). The SMDP value decomposition provides the mathematical foundation for Cassie's hierarchical value estimation across the Planner→Primitives→Controller hierarchy.

## What to Borrow / Implement
- Implement PPOC (Proximal Policy Option-Critic) as the base algorithm for Cassie's Primitives level, combining PPO stability with option learning
- Add deliberation cost ξ to Cassie's option termination to ensure primitives persist for complete gait cycles
- Implement interest functions to constrain primitive availability based on Cassie's dynamic state (e.g., balance metrics)
- Use diversity-promoting objectives (MI maximization between states and options) alongside DIAYN to prevent primitive collapse
- Apply the SMDP value decomposition for hierarchical value estimation across Cassie's four-level architecture

## Limitations & Open Questions
- Survey format means no novel experimental results—the discussed methods are compiled from other papers
- Coverage of continuous control applications is somewhat limited compared to discrete domains
- The interaction between options and off-policy learning (important for sample efficiency) is underexplored
- Scaling the options framework to very deep hierarchies (4+ levels as in Cassie's architecture) is not well-addressed in the surveyed literature
