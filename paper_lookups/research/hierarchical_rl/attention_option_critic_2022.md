# Attention Option-Critic

**Authors:** Mariano Phielipp, Alejandro H. Toselli, Łukasz Kaiser
**Year:** 2022 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2201.02628)

---

## Abstract Summary
Attention Option-Critic (AOC) addresses a fundamental limitation of the standard Option-Critic architecture: option degeneracy, where all learned options collapse to near-identical policies despite being initialized differently. This degeneracy eliminates the benefits of hierarchical decomposition, as the agent effectively operates with a flat policy disguised as multiple options. AOC integrates attention mechanisms into the option framework, forcing different options to attend to different subsets of the observation space, thereby promoting functional diversity among learned options.

The key insight is that different sub-behaviors naturally require attending to different features of the environment. For locomotion, a balancing option should attend to IMU and center-of-mass data, while a stepping option should attend to foot contact sensors and terrain features. By parameterizing each option with a distinct attention mask over the observation space, AOC encourages this natural specialization. The attention weights are learned end-to-end alongside the option policies and termination functions.

Experiments across Atari games and continuous control tasks demonstrate that AOC produces significantly more diverse and interpretable options compared to standard Option-Critic, while achieving equal or better task performance. The attention masks provide built-in interpretability, revealing which observation features each option considers most important.

## Core Contributions
- Identification of the option degeneracy problem: formal analysis showing why standard Option-Critic options collapse to similar policies under gradient-based optimization
- Integration of learned attention masks into option-critic, where each option has a unique soft attention over observation dimensions
- Diversity regularization through attention orthogonality loss, encouraging different options to attend to non-overlapping observation subsets
- Demonstration that attention-based options are significantly more diverse (measured by policy divergence metrics) than standard Option-Critic options
- Built-in interpretability: attention weights reveal which observation features each option prioritizes, enabling human understanding of learned hierarchies
- Compatible with both discrete and continuous action spaces, and with any base policy gradient method
- Reduced option switching frequency as a natural consequence of diverse options (options switch because context changes, not because they are interchangeable)

## Methodology Deep-Dive
AOC modifies the Option-Critic architecture by introducing a per-option attention module. For option ω with index i, the attention mask α_i(s) ∈ [0,1]^d is computed as α_i = σ(W_i · s + b_i), where W_i ∈ R^{d×d} and b_i ∈ R^d are learnable parameters, σ is the sigmoid function, and d is the observation dimensionality. The masked observation is z_i = α_i ⊙ s (element-wise product), and the intra-option policy operates on z_i instead of s directly: π_ω(a|z_i). This forces different options to develop policies based on different observation subsets.

To prevent all attention masks from converging to identical all-ones patterns (which would recover standard Option-Critic), AOC introduces an attention orthogonality loss: L_ortho = Σ_{i≠j} ||α_i^T · α_j||² / (||α_i||² · ||α_j||²). This cosine-similarity-based penalty encourages attention masks to be as orthogonal as possible, ensuring different options attend to different features. The orthogonality loss is weighted by a hyperparameter λ_ortho ∈ [0.01, 0.1] and added to the total loss.

The full optimization objective combines four terms: (1) the critic TD loss for Q_Ω(s,ω), (2) the intra-option policy gradient (from standard Option-Critic), (3) the termination gradient (from standard Option-Critic), and (4) the attention orthogonality loss. Gradients flow through the attention masks via the reparameterization trick, enabling end-to-end training of attention parameters alongside policy and termination parameters.

For continuous control tasks (locomotion), the attention mechanism naturally discovers meaningful feature groupings. In quadruped locomotion experiments, option 1 might attend heavily to joint positions and velocities of the front legs (attention weights > 0.8), while option 2 attends to rear leg states and body orientation. This specialization emerges without explicit supervision, driven only by the orthogonality loss and the task reward.

AOC also introduces an entropy bonus over the policy-over-options μ(ω|s) to ensure all options are utilized: L_entropy = -H(μ(·|s)). This prevents the degenerate case where a single option dominates all states. Combined with the attention orthogonality loss, this produces a set of well-utilized, diverse options with distinct observational focus.

The architecture uses a shared observation encoder (2-3 MLP layers), followed by N parallel branches (one per option), each containing an attention module, a policy head, and a termination head. The critic shares the encoder but has a separate value head outputting Q_Ω(s,ω) for all options simultaneously.

## Key Results & Numbers
- Option diversity (measured by average JS divergence between option policies): AOC achieves 0.45 vs 0.08 for standard Option-Critic (5.6x improvement)
- Atari: AOC matches or exceeds standard Option-Critic performance while producing interpretable options (Pong: options specialize in offense vs defense)
- Continuous control (MuJoCo HalfCheetah): AOC discovers 4 distinct movement phases as options, achieving 5% higher reward than flat SAC baseline
- Option switching frequency reduced by 30-40% compared to standard Option-Critic, as options switch based on context rather than degeneracy
- Attention orthogonality effectively prevents mask collapse: average cosine similarity between option masks < 0.15 (vs > 0.85 for standard OC)
- Training overhead: ~10% more computation than standard Option-Critic due to attention computation and orthogonality loss
- Interpretability analysis: attention weights correlate >0.8 with manually identified feature relevance for known sub-tasks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
AOC provides a mechanism for learning diverse locomotion sub-behaviors for the Mini Cheetah that naturally specialize based on different sensor modalities. For example, one option might specialize in balancing (attending to IMU data) while another handles forward propulsion (attending to leg joint states). This could be valuable if Project A aims to learn multiple gaits within a single policy.

The attention masks offer interpretability benefits for debugging Mini Cheetah policies—understanding which sensors each sub-behavior relies on can inform sensor failure robustness analysis and help identify unnecessary observation dimensions. However, if Project A uses a flat PPO policy without hierarchical structure, AOC's benefits are limited to potential future extensions.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
AOC is highly relevant to Project B's Primitives level, which uses Option-Critic for locomotion primitive selection. The option degeneracy problem is a real risk for Cassie—if all locomotion primitives collapse to similar policies, the hierarchical structure provides no benefit. AOC's attention mechanism directly addresses this by ensuring different primitives attend to different aspects of Cassie's observation space.

The attention mechanism complements Project B's Dual Asymmetric-Context Transformer architecture. While the Transformer processes temporal context, AOC's attention operates over observation dimensions within each timestep, providing a complementary form of attention. The attention orthogonality loss can be integrated with DIAYN/DADS diversity objectives to ensure skill diversity from both observation-space and behavior-space perspectives. For Cassie's specific observation space, attention masks could naturally separate: (1) hip-level states for balance primitives, (2) ankle-level states for foot placement primitives, (3) global orientation for heading control primitives, and (4) contact force states for terrain adaptation primitives.

## What to Borrow / Implement
- Integrate attention masks into Project B's Option-Critic implementation at the Primitives level to prevent option degeneracy on Cassie
- Use the attention orthogonality loss (λ_ortho ~ 0.05) as a regularizer alongside DIAYN/DADS diversity objectives
- Leverage attention weight visualization for debugging and interpreting learned locomotion primitives on both platforms
- Apply the entropy bonus over policy-over-options to ensure all locomotion primitives are utilized during Cassie training
- Use attention masks as a feature importance diagnostic to identify which sensor inputs matter most for each locomotion phase

## Limitations & Open Questions
- Soft attention masks may not produce sufficiently sharp specialization; hard attention (Gumbel-Softmax) might produce more distinct options but introduces gradient estimation challenges
- Orthogonality loss with many options (>6-8) may over-constrain the system, as observation dimensions must be shared among many options
- No evaluation on real robotic hardware; attention-based options may be sensitive to observation noise that affects mask computation
- Interaction between attention mechanisms and domain randomization unexplored—randomized observations may confuse learned attention patterns
