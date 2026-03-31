---
## 📂 FOLDER: research/hierarchical_rl/

### 📄 FILE: research/hierarchical_rl/option_critic_architecture_bacon_2017.md

**Title:** The Option-Critic Architecture
**Authors:** Pierre-Luc Bacon, Jean Harb, Doina Precup
**Year:** 2017
**Venue:** AAAI 2017
**arXiv / DOI:** arXiv:1609.05140

**Abstract Summary (2–3 sentences):**
This foundational paper introduces the Option-Critic architecture, which enables end-to-end learning of options (temporally extended actions), their intra-option policies, and termination functions within the options framework. The authors derive policy gradient theorems for option parameters, allowing simultaneous discovery and optimization of hierarchical policies without requiring manual option specification or predefined subgoal structures. The approach is validated on Atari games and continuous control tasks, demonstrating that meaningful temporal abstractions can emerge automatically through gradient-based optimization.

**Core Contributions (bullet list, 4–7 items):**
- Derivation of the intra-option policy gradient theorem enabling gradient-based optimization of option policies
- Derivation of the termination gradient theorem for learning when options should terminate
- End-to-end architecture that simultaneously learns option policies, termination conditions, and the policy-over-options
- Elimination of the need for manual option specification, subgoal definition, or pre-segmented demonstrations
- Theoretical grounding in the options framework of Sutton, Precup, and Singh (1999) with novel gradient results
- Empirical demonstration of meaningful option discovery in both discrete (Atari) and continuous control domains
- Call-and-return execution model compatible with off-policy learning and experience replay

**Methodology Deep-Dive (3–5 paragraphs):**
The Option-Critic architecture is built upon the options framework, where an option ω consists of three components: an initiation set I_ω (states where the option can be started), an intra-option policy π_ω(a|s) (the action selection policy while the option is active), and a termination function β_ω(s) (the probability of terminating the option in a given state). The key innovation is deriving gradient expressions for both the intra-option policy parameters θ and the termination function parameters ϕ, enabling these to be learned simultaneously via stochastic gradient methods. The policy-over-options μ(ω|s) selects which option to execute, the intra-option policy executes primitive actions, and the termination function decides when to return control to the policy-over-options. This call-and-return execution model naturally produces temporally extended behaviors.

The intra-option policy gradient theorem extends the standard policy gradient to the hierarchical setting. The authors show that the gradient of the expected return with respect to the intra-option policy parameters θ decomposes as a sum over option-state pairs, weighted by the discounted occupancy measure under the current option. Crucially, this gradient can be computed using quantities available during online interaction: the advantage function Q_U(s,ω,a) - Q_U(s,ω), where Q_U is the option-value function that accounts for the possibility of option termination and continuation. This means that standard actor-critic methods can be adapted to learn intra-option policies by conditioning on the currently active option. The practical algorithm uses a critic that estimates Q_U(s,ω,a) and an actor for each option that updates its policy parameters in the direction of the advantage-weighted policy gradient.

The termination gradient theorem addresses learning when options should terminate, which is equally important as learning what actions to take within options. The gradient of the expected return with respect to the termination parameters ϕ is proportional to the advantage of continuing the current option versus terminating and selecting a new one: A_Ω(s,ω) = Q_U(s,ω) - V_Ω(s), where V_Ω(s) is the value of state s under the policy-over-options. If continuing the current option has higher value than the expected value of switching, the termination probability should decrease, and vice versa. The authors add a deliberation cost η that regularizes against too-frequent option switching, encouraging temporally extended options. Without this regularization, the degenerate solution of single-step options (equivalent to a flat policy) is a valid optimum, so the deliberation cost is essential for obtaining meaningful temporal abstractions.

The complete Option-Critic algorithm alternates between three updates: (1) the critic update, which learns Q_Ω(s,ω) and Q_U(s,ω,a) using temporal difference methods; (2) the intra-option policy update using the intra-option policy gradient; and (3) the termination function update using the termination gradient. All three updates use quantities computed from the same stream of experience, making the algorithm efficient and compatible with online learning. The architecture supports both tabular and function approximation settings, with the deep version using neural networks for all components. In the deep version, the network shares early layers across all options (processing visual or state inputs) and branches into option-specific heads for the intra-option policies and termination functions, with a separate head for the policy-over-options.

The authors also discuss the relationship between Option-Critic and other hierarchical RL approaches. Unlike feudal methods (FeUdal Networks, HIRO) that use explicit goal-conditioned sub-policies, Option-Critic discovers temporal abstractions without requiring a goal space. Unlike skill discovery methods based on mutual information (DIAYN, VIC), Option-Critic optimizes options directly for task performance rather than diversity. The framework is compatible with both on-policy (A2C-style) and off-policy (DQN-style) learning, with the paper presenting results for both variants. The theoretical results hold for general parameterized policies and termination functions, making the framework applicable to discrete and continuous action spaces.

**Key Results & Numbers:**
- Successfully learns meaningful options (room-navigation subroutines) in the four-rooms gridworld domain
- Achieves competitive or superior performance to DQN on several Atari games when using 4–8 options
- Options specialize to interpretable behaviors (e.g., navigating corridors, avoiding enemies) without supervision
- Deliberation cost η effectively controls option duration, with higher values producing longer-lasting options
- Continuous control experiments show smooth option switching and stable multi-modal behaviors
- Training overhead compared to flat policies is minimal (approximately 10-15% additional computation per step)

**Relevance to Project A (Mini Cheetah):** MEDIUM — Provides the theoretical foundation for a potential hierarchical Mini Cheetah controller where different options correspond to different locomotion modes (walking, trotting, galloping). However, the current Mini Cheetah project uses flat PPO, so Option-Critic would represent an architectural change rather than a direct improvement to the existing approach.

**Relevance to Project B (Cassie HRL):** HIGH — The Option-Critic architecture is a direct foundation for the Option-Critic-based primitives level in the Cassie HRL system. The intra-option policy gradient theorem provides the mathematical basis for training locomotion primitives, the termination gradient enables learning when to switch between gaits/behaviors, and the policy-over-options maps to the higher-level planner's decision-making. The deliberation cost concept is directly applicable to preventing overly frequent primitive switching.

**What to Borrow / Implement:**
- Intra-option policy gradient theorem for training locomotion primitive policies within the options framework
- Termination gradient for learning when locomotion primitives should hand off control
- Deliberation cost mechanism to encourage temporally coherent locomotion behaviors
- Shared feature extraction layers across options to reduce parameter count for multiple primitives
- Call-and-return execution model for structured primitive execution and switching

**Limitations & Open Questions:**
- Option degeneration problem: without sufficient deliberation cost, options collapse to single-step actions
- Number of options must be specified a priori; no mechanism for growing or pruning the option set
- Initiation sets are typically set to the full state space (all options available everywhere), losing potential specialization
- Scalability to high-dimensional continuous action spaces (like full robot joint control) requires careful network architecture design
- The relationship between learned options and human-interpretable skills is not guaranteed—options may not correspond to meaningful behaviors
- Credit assignment across long option executions can be challenging, potentially slowing learning for options with long durations
---
