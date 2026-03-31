# The Option-Critic Architecture

**Authors:** Pierre-Luc Bacon, Jean Harb, Doina Precup
**Year:** 2017 | **Venue:** AAAI
**Links:** [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10916)

---

## Abstract Summary
The Option-Critic Architecture is a foundational contribution to hierarchical reinforcement learning that enables end-to-end learning of options—temporally extended actions consisting of an initiation set, an intra-option policy, and a termination condition. Prior to this work, options were either hand-designed by domain experts or discovered through separate, often disconnected, procedures. Option-Critic unifies the learning of all option components through gradient-based optimization within a single actor-critic framework.

The architecture derives policy gradient theorems for both intra-option policies and termination functions, enabling simultaneous optimization of: (1) a policy over options (which option to execute), (2) intra-option policies (what primitive actions to take within each option), and (3) termination functions (when to stop executing the current option). This end-to-end learning eliminates the need for predefined skill decompositions and allows the agent to discover temporal abstractions suited to the task.

Experiments demonstrate that Option-Critic discovers meaningful options in both discrete (Atari games, four-rooms) and continuous control domains, with options naturally corresponding to interpretable sub-behaviors. The framework generalizes the standard actor-critic to the options setting and is compatible with any policy gradient method as the base optimizer.

## Core Contributions
- Derivation of the intra-option policy gradient theorem, extending the standard policy gradient to options with proper credit assignment across temporal abstractions
- Derivation of the termination gradient, enabling learned termination conditions that balance option duration with task performance
- End-to-end differentiable architecture that jointly optimizes all three option components (policy-over-options, intra-option policies, termination functions)
- Demonstration that meaningful temporal abstractions emerge from end-to-end learning without explicit sub-goal specification
- Theoretical proof that the option-critic gradient is unbiased and consistent, inheriting convergence properties from the base policy gradient method
- Framework generality: compatible with any policy gradient method (REINFORCE, A2C, PPO) and any function approximator (neural networks)
- Introduction of a deliberation cost regularizer (ξ) to control option switching frequency

## Methodology Deep-Dive
The Options framework (Sutton, Precup, Singh 1999) defines an option ω = (I_ω, π_ω, β_ω) where I_ω ⊆ S is the initiation set, π_ω(a|s) is the intra-option policy, and β_ω(s) ∈ [0,1] is the termination function. A policy over options μ(ω|s) selects which option to execute. The agent follows the selected option's policy until termination, then selects a new option. The Option-Critic Architecture learns π_ω, β_ω, and the option-value function Q_Ω(s,ω) end-to-end.

The intra-option policy gradient theorem states: ∂Q_Ω(s,ω)/∂θ = Σ_{s',ω'} Σ_t γ^t P(s_t=s', ω_t=ω' | s_0=s, ω_0=ω) Σ_a ∂π_{ω'}(a|s')/∂θ · Q_U(s',ω',a), where Q_U is the option-action value upon entering state s' under option ω' and taking action a. This is analogous to the standard policy gradient but accounts for the hierarchical structure of option execution.

The termination gradient theorem provides: ∂Q_Ω(s,ω)/∂ϑ = -Σ_{s',ω'} Σ_t γ^t P(s_t=s', ω_t=ω') ∂β_{ω'}(s')/∂ϑ · A_Ω(s',ω'), where A_Ω(s',ω') = Q_Ω(s',ω') - V_Ω(s') is the advantage of continuing option ω' over re-selecting. The termination function learns to terminate when the current option is no longer advantageous, naturally discovering option boundaries aligned with task structure.

The deliberation cost ξ is added to the termination gradient to discourage excessive option switching: the modified gradient includes a term -ξ · ∂β_ω(s)/∂ϑ that penalizes high termination probabilities. This hyperparameter controls the trade-off between temporal abstraction (longer options) and flexibility (frequent re-selection). In practice, ξ ∈ [0.01, 0.1] produces options lasting 5-50 timesteps.

The architecture uses shared lower layers between option policies with option-specific heads. The critic learns Q_Ω(s,ω) using semi-gradient TD updates with the option-value Bellman equation: Q_Ω(s,ω) = Σ_a π_ω(a|s)[r(s,a) + γ Σ_{s'} P(s'|s,a) U(ω,s')], where U(ω,s') = (1-β_ω(s'))Q_Ω(s',ω) + β_ω(s')V_Ω(s') handles the continuation/termination branching.

Implementation uses a neural network with shared convolutional/MLP layers, branching into N option heads (each outputting π_ω and β_ω) and a critic head outputting Q_Ω(s,ω) for all options. Training alternates between: (1) critic update via TD, (2) intra-option policy gradient update, and (3) termination gradient update. The policy over options μ(ω|s) can be ε-greedy over Q_Ω or a learned softmax policy.

## Key Results & Numbers
- Four-rooms domain: discovers 2-4 meaningful corridor-navigation options from scratch, matching hand-designed options
- Atari games: with 4-8 options, achieves performance comparable to DQN while discovering interpretable temporal abstractions
- Options naturally specialize: in locomotion, different options correspond to different movement phases (stance, swing, turning)
- Deliberation cost ξ=0.01 produces options averaging 15-25 timesteps; ξ=0.1 produces 30-50 timestep options
- Training overhead vs flat A2C: ~20% more computation per step due to option-level gradients, but often fewer total steps to convergence
- Transfer learning: policies with discovered options transfer better to modified environments (20-30% faster adaptation)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For Project A, Option-Critic provides a framework for automatically decomposing Mini Cheetah locomotion into temporal abstractions. Different options could naturally learn to correspond to different gait phases (stance, flight, transition) or different locomotion modes (walking, trotting, galloping). This decomposition could improve training efficiency by enabling the policy to reason at a higher temporal level, reducing the effective horizon.

However, Project A's current flat PPO approach may be sufficient for the locomotion task if the gait variety requirement is limited. Option-Critic becomes more valuable if Project A needs to learn multiple locomotion behaviors (speed changes, direction changes, terrain adaptation) within a single policy, where temporal abstractions can help manage the behavioral complexity.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Option-Critic is explicitly specified as a component of Project B's Primitives level, making this paper essential reading. The Primitives level uses Option-Critic to select among locomotion primitives (walking, turning, speed changes), with each primitive being an option consisting of an intra-option policy and a termination condition. Understanding the gradient derivations is critical for implementing and tuning this component.

The termination gradient is particularly important for Cassie, as premature termination of locomotion primitives can cause instability, while overly long options reduce responsiveness to high-level commands from the Planner. The deliberation cost ξ must be carefully tuned for bipedal balance—too much switching destabilizes the gait, too little switching makes the robot unresponsive. The architecture's compatibility with the Dual Asymmetric-Context Transformer means option policies can leverage the full context window for informed action selection. Integration with DIAYN/DADS at the skill discovery level requires ensuring option diversity, which connects to the attention and variational extensions of Option-Critic.

## What to Borrow / Implement
- Implement the Option-Critic architecture as the foundation for Project B's Primitives level, with locomotion-specific options (walk, turn, speed-up, slow-down, recover)
- Use the deliberation cost regularizer (ξ) tuned for bipedal stability to control option switching frequency on Cassie
- Apply the intra-option policy gradient with PPO as the base optimizer for stable option learning
- Explore shared lower layers with option-specific heads to enable parameter sharing across locomotion primitives
- Use transfer learning capability: pre-train options on simple terrains, then fine-tune on complex environments

## Limitations & Open Questions
- Option degeneracy: without careful regularization, all options may converge to similar policies, defeating the purpose of hierarchical decomposition
- Deliberation cost ξ is a sensitive hyperparameter with no principled selection method; requires manual tuning
- Scalability to large option spaces (>8-10 options) degrades, as the policy-over-options must discriminate among many choices
- No formal guarantees on option interpretability—discovered options may not correspond to human-meaningful behaviors
