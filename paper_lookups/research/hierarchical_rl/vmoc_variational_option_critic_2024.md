# Learning Variational Temporal Abstraction Embeddings in Option-Induced MDPs (VMOC)

**Authors:** (2024)
**Year:** 2024 | **Venue:** OpenReview
**Links:** [OpenReview](https://openreview.net/forum?id=eM3Wzs6Unt)

---

## Abstract Summary
VMOC (Variational Mixture of Option-Critic) introduces a variational inference framework for learning options in hierarchical reinforcement learning. The core innovation lies in reformulating option discovery as a variational inference problem, where option embeddings are learned in a continuous latent space rather than being discrete indices. This enables smoother option transitions, better scalability to large option sets, and more stable training dynamics compared to the standard Option-Critic architecture.

The method augments the Option-Critic framework with entropy-based rewards that encourage diverse and well-separated option behaviors, addressing the well-known problem of option collapse where all options converge to similar behaviors. By learning option embeddings through a variational autoencoder-like objective, VMOC can represent options as points in a continuous latent space, enabling interpolation between options and more principled option selection through the policy-over-options.

Experiments on MuJoCo locomotion tasks (HalfCheetah, Walker2d, Ant, Humanoid) demonstrate that VMOC consistently outperforms standard Option-Critic, PPOC (Proximal Policy Option-Critic), and other HRL baselines. The learned options show diverse, interpretable behaviors with stable termination patterns, and the variational embedding enables effective transfer to new task configurations.

## Core Contributions
- Reformulates option discovery as variational inference, learning continuous option embeddings instead of discrete option indices
- Introduces entropy-augmented reward shaping that prevents option collapse and encourages behavioral diversity across options
- Demonstrates stable option termination learning, addressing a key instability in standard Option-Critic where options either never terminate or terminate immediately
- Achieves state-of-the-art performance on MuJoCo locomotion benchmarks among HRL methods
- Shows that continuous option embeddings enable smooth interpolation between learned behaviors and improve transfer to new tasks
- Provides scalability to larger option sets (16–64 options) without degradation, unlike discrete Option-Critic which struggles beyond 4–8 options
- Includes thorough ablations on entropy coefficient, embedding dimension, and number of options

## Methodology Deep-Dive
VMOC builds upon the Option-Critic architecture but introduces three key modifications: variational option embeddings, entropy-augmented option rewards, and a modified termination gradient. In standard Option-Critic, options are indexed by discrete integers and the policy-over-options selects among them using a softmax distribution. VMOC instead represents each option as a learned embedding vector zₖ ∈ ℝᵈ, where d is the embedding dimension (typically 8–32). The policy-over-options outputs parameters of a distribution over the continuous embedding space, and the closest option embedding is selected (or a soft mixture is used).

The variational objective combines the task reward with a KL divergence term that regularizes the option embedding distribution. Specifically, the loss includes: L = L_task + β·KL(q(z|s,a) ‖ p(z)), where q(z|s,a) is the posterior option assignment given states and actions, and p(z) is a prior (typically standard Gaussian). This variational term serves dual purposes: it regularizes the embedding space for smoothness and prevents degenerate solutions where all options collapse to the same embedding region. The β coefficient controls the diversity-performance trade-off.

The entropy-augmented reward is applied at the intra-option level: R_aug = R_task + α·H(π_z(·|s)), where H is the entropy of the intra-option policy. This encourages each option to maintain diverse action distributions within its behavioral repertoire, preventing premature convergence to deterministic behaviors. The entropy coefficient α is annealed during training, starting high for exploration and decaying to allow specialization.

The termination function β_z(s) is parameterized as a sigmoid network conditioned on both the state s and the option embedding z. VMOC modifies the Option-Critic termination gradient by adding a deliberation cost term that penalizes excessive switching: ∂L_term/∂θ_β = (Q(s,z') - Q(s,z) + ξ)·∂β_z(s)/∂θ_β, where ξ > 0 is the switching cost. This stabilizes termination learning by creating a bias toward continuing the current option unless a different option offers sufficiently higher value.

Training uses PPO as the base optimizer for all components (intra-option policies, policy-over-options, termination functions, and option embeddings) with shared feature extraction layers. The option embeddings are updated via both the variational objective and backpropagation from the policy-over-options selection.

The architecture enables scaling to many options because the continuous embedding space allows options to specialize in nearby regions without requiring separate network heads for each option—a single conditioned network generates behavior for any option embedding.

## Key Results & Numbers
- Outperforms standard Option-Critic by 20–35% in average return on MuJoCo locomotion tasks
- Outperforms PPOC by 10–20% while maintaining similar sample efficiency
- Stable training with 16–64 options, whereas standard Option-Critic degrades beyond 8 options
- Option utilization: 85–95% of options are actively used (vs. 30–50% in standard Option-Critic due to option collapse)
- Embedding dimension d=16 found optimal; d=8 too constrained, d=32 no additional benefit
- Entropy coefficient α=0.01 annealed to 0.001 provides best diversity-performance trade-off
- Switching cost ξ=0.01 produces options with average duration of 15–30 steps on locomotion tasks
- Transfer experiments show 40–60% faster adaptation to modified tasks compared to training from scratch

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
VMOC provides an advanced option discovery mechanism that could enhance Mini Cheetah's skill learning beyond basic PPO. The continuous option embeddings could represent different locomotion skills (various gaits, recovery behaviors, turning strategies) in a smooth latent space, enabling interpolation between skills for smoother transitions. The entropy-augmented rewards would help discover diverse behaviors during curriculum learning.

However, for the Mini Cheetah project's primary focus on robust locomotion with PPO, VMOC may introduce unnecessary complexity. The option framework is most beneficial when the task requires temporal abstraction and skill composition, which may not be the primary requirement for velocity-tracking locomotion. It becomes more relevant if the project expands to include multiple distinct behaviors.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
VMOC is directly relevant to Cassie's Primitives level, which uses Option-Critic for primitive selection and execution. VMOC addresses the known limitations of standard Option-Critic that will likely manifest in Cassie's training: option collapse (all primitives converge to similar walking behavior), termination instability (primitives either persist forever or terminate every step), and scaling issues (difficulty maintaining diverse primitives as the number increases).

The variational option embeddings provide a principled replacement for discrete primitive indices, enabling smooth transitions between locomotion primitives that are critical for bipedal balance. The deliberation cost mechanism directly addresses the need for appropriate primitive duration—Cassie's walking primitives should persist for complete gait cycles rather than switching mid-stride. The continuous embedding space also provides a natural interface with the GATv2 Planner, which can output target embeddings rather than discrete primitive selections, allowing for more nuanced locomotion commands.

## What to Borrow / Implement
- Replace discrete Option-Critic with VMOC's variational option embeddings at Cassie's Primitives level for smoother primitive transitions
- Implement the entropy-augmented intra-option reward to prevent primitive collapse and ensure diverse locomotion skills
- Adopt the deliberation cost (switching cost ξ) to stabilize primitive termination at gait-cycle-appropriate durations
- Use the continuous option embedding space as the Planner→Primitives interface, allowing GATv2 to output target embeddings
- Apply the variational KL regularization to ensure smooth, interpolable primitive space for Cassie's locomotion

## Limitations & Open Questions
- Computational overhead of the variational objective and embedding learning adds ~20% training time compared to standard Option-Critic
- The continuous embedding space may make it harder to interpret individual options compared to discrete indices
- Optimal number of options and embedding dimension require tuning per task—no automatic method for determining these
- Real-robot validation is absent; all experiments are in MuJoCo simulation
