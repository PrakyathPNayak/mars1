# Constrained Decision Transformer for Offline Safe Reinforcement Learning

**Authors:** Zuxin Liu, Zijian Guo, Yihang Yao, Zhepeng Cen, Wenhao Yu, Tingnan Zhang, Ding Zhao
**Year:** 2023 | **Venue:** ICML
**Links:** https://proceedings.mlr.press/v202/liu23m.html

---

## Abstract Summary
Constrained Decision Transformer (CDT) extends the Decision Transformer framework to handle safety constraints in offline reinforcement learning. While the original DT conditions only on desired returns, CDT introduces a dual-conditioning mechanism that simultaneously targets both task reward and safety cost budgets. This enables the learned policy to navigate the Pareto frontier between performance and safety at deployment time without retraining, making it highly suitable for real-world robotic applications where safety margins must be adjustable.

The key innovation is a multi-objective optimization formulation within the sequence modeling paradigm. CDT models trajectories as sequences of (return-to-go, cost-to-go, state, action) tuples, where the cost-to-go represents the anticipated cumulative safety violations. By conditioning on both targets, the agent can dynamically trade off between maximizing task performance and minimizing constraint violations based on deployment-time specifications.

CDT is evaluated on the DSRL (D4RL Safety) benchmark suite, which includes safety-constrained versions of HalfCheetah, Hopper, Walker2d, and other continuous control tasks. It consistently outperforms prior safe offline RL methods including BCQ-Lagrangian, BEAR-Lagrangian, and CPQ in terms of both reward performance and constraint satisfaction. The approach demonstrates particular robustness when the safety threshold is varied at test time, showing smooth interpolation between conservative and aggressive policies.

## Core Contributions
- Extends Decision Transformer to the constrained MDP setting with simultaneous return-to-go and cost-to-go conditioning
- Enables deployment-time adjustment of safety-performance trade-offs without retraining the policy
- Introduces a cost augmentation strategy that enriches offline datasets with diverse safety-performance trade-off demonstrations
- Achieves state-of-the-art performance on DSRL benchmarks, outperforming CPQ, BCQ-Lagrangian, and BEAR-Lagrangian
- Provides theoretical analysis showing that the dual-conditioning mechanism recovers the Pareto-optimal policy under certain data coverage conditions
- Demonstrates robustness to varying safety thresholds, enabling smooth policy interpolation between conservative and aggressive behaviors
- Shows particular suitability for real-world robotic deployment where safety budgets must be adaptable to changing conditions

## Methodology Deep-Dive
CDT extends the Decision Transformer's input representation by appending a cost-to-go token Ĉ_t alongside the return-to-go R̂_t at each timestep. The trajectory representation becomes (R̂_t, Ĉ_t, s_t, a_t) tuples over a context window of K timesteps. The cost-to-go Ĉ_t = Σ_{t'=t}^{T} c_{t'} represents the cumulative safety cost from timestep t to the episode end, where c_t is the instantaneous safety cost (e.g., joint torque violation, contact force exceeding limits, falling).

The architecture maintains the GPT-2 causal transformer backbone but expands the embedding layer to accommodate the additional cost modality. Each of the four token types (return-to-go, cost-to-go, state, action) is projected into the transformer's hidden dimension via separate linear layers. Positional embeddings are shared within each timestep across all four tokens, preserving temporal alignment.

Training employs a two-phase strategy. First, the offline dataset is augmented with synthetic trajectories that span diverse safety-performance trade-off levels. This is achieved by relabeling existing trajectories with different cost budgets through a hindsight relabeling procedure—analogous to Hindsight Experience Replay but for constraint budgets. The augmented dataset ensures the model observes a wide range of (return, cost) conditioning pairs. Second, standard supervised learning minimizes the action prediction MSE, identical to the original DT training procedure.

A critical design choice is the cost normalization scheme. Raw safety costs can vary dramatically across environments and episodes, making conditioning unstable. CDT normalizes costs to [0, 1] based on the empirical distribution in the offline dataset, ensuring consistent conditioning behavior. Similarly, returns are normalized, preventing scale mismatches between the two conditioning signals.

At deployment, the operator specifies a desired return target R̂_target and a maximum cost budget Ĉ_max. The policy generates actions that aim to achieve R̂_target while keeping cumulative costs below Ĉ_max. During execution, both conditioning values are decremented: R̂_{t+1} = R̂_t − r_t and Ĉ_{t+1} = Ĉ_t − c_t. If the cost budget is nearly exhausted, the model naturally shifts to more conservative actions. This dynamic adaptation is emergent—it arises from the training distribution rather than explicit safety logic.

## Key Results & Numbers
- HalfCheetah-Safe: CDT achieves 95.2% of unconstrained return while maintaining <5% constraint violations, vs. CPQ at 87.3%
- Hopper-Safe: CDT return 78.4 (normalized) with 2.1% violations, vs. BCQ-Lag 65.2 with 8.7% violations
- Walker2d-Safe: CDT outperforms all baselines in both reward (82.1) and safety (1.8% violations)
- Smooth Pareto interpolation: varying cost budget from 0 to 0.5 produces monotonically increasing returns with monotonically increasing (but controlled) violations
- CDT reduces constraint violations by 60-80% compared to unconstrained DT while sacrificing only 5-15% reward performance
- Training converges within 200K gradient steps across all environments
- Hindsight cost relabeling improves data efficiency by 3-5x compared to training on raw datasets alone

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
CDT is highly relevant to Mini Cheetah deployment where safety is paramount. The quadruped operates with real hardware subject to joint torque limits, thermal constraints, and contact force budgets. CDT's ability to condition on safety budgets at deployment time allows operators to dynamically adjust the trade-off—e.g., running more aggressively on flat terrain and more conservatively on uneven surfaces—without retraining the policy. This is especially valuable during iterative sim-to-real transfer, where initial real-world tests should be conservative and progressively relaxed as confidence grows.

The offline nature of CDT means it can leverage existing Mini Cheetah trajectory datasets (from simulation or prior controllers) to learn safe policies without online interaction, reducing the risk of hardware damage during training. The cost-to-go conditioning naturally encodes safety specifications that are critical for quadruped hardware protection.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
CDT is directly applicable to Cassie's Safety level in the 4-level hierarchical architecture. The Safety level must ensure constraint satisfaction (joint limits, balance stability, contact forces) while allowing the lower levels (Primitives, Controller) to optimize locomotion performance. CDT's dual-conditioning mechanism maps naturally to this: the return-to-go conditions the locomotion objective from the Planner level, while the cost-to-go conditions the safety budget from the Safety level.

The integration with the Learned Control Barrier Function (LCBF) paradigm is particularly promising. CDT could serve as the policy backbone within the Safety level, with the LCBF providing the safety cost signal c_t. The deployment-time adjustability means the safety margin can be tightened for challenging terrains (stairs, slopes) and relaxed for flat ground, matching the adaptive safety requirements of the Cassie project. The hindsight cost relabeling technique could also enhance the diversity of the safety-annotated trajectory dataset used to train the LCBF.

## What to Borrow / Implement
- Implement dual (return-to-go, cost-to-go) conditioning in the Cassie DACT architecture for integrated safety-performance optimization
- Adopt hindsight cost relabeling to augment offline datasets with diverse safety-performance trade-off examples
- Use CDT's cost normalization scheme for consistent safety conditioning across different terrain types and locomotion modes
- Deploy CDT's dynamic budget decrementing mechanism at the Safety level for real-time adaptive constraint enforcement
- Leverage deployment-time safety threshold tuning for progressive sim-to-real transfer on both robots

## Limitations & Open Questions
- Offline-only training limits adaptation to novel safety scenarios not represented in the training data; online fine-tuning extensions are needed
- Cost-to-go conditioning assumes safety costs are well-defined and measurable, which may not hold for all real-world safety specifications (e.g., soft subjective comfort constraints)
- The hindsight relabeling strategy may introduce distribution shift if synthetic cost labels are far from the true data distribution
- Scalability to high-dimensional safety specifications (multiple simultaneous constraints) is not thoroughly evaluated; Cassie's project may need multi-dimensional cost-to-go vectors
