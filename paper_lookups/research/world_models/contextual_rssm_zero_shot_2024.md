# Dreaming of Many Worlds: Learning Contextual World Models Aids Zero-Shot Generalization

**Authors:** (AutoML 2024)
**Year:** 2024 | **Venue:** AutoML/arXiv
**Links:** AutoML 2024

---

## Abstract Summary
This paper introduces the contextual Recurrent State-Space Model (cRSSM), an extension of the standard RSSM (as used in Dreamer/DreamerV3) that conditions the world model on observable context variables representing environment parameters such as payload mass, terrain type, gravity, or friction coefficients. By making the world model explicitly aware of the context in which transitions occur, the cRSSM learns a family of dynamics models indexed by context, enabling zero-shot generalization to unseen combinations of context variables without retraining.

The standard RSSM learns a single dynamics model that averages over all environment variations encountered during training (e.g., from domain randomization). This averaging produces a model that is adequate for no specific scenario but optimal for none. The cRSSM instead conditions each component of the RSSM (prior, posterior, reward predictor, decoder) on the context vector c, learning context-specific dynamics. During imagination-based policy training, the policy is optimized across imagined rollouts with diverse context values, learning to be robust to environment variation. At deployment, the context can be either observed directly (known payload mass), estimated from proprioceptive data, or inferred by the world model's posterior.

The paper validates the cRSSM on continuous control tasks with varying dynamics parameters (CartPole with varying pole length/mass, HalfCheetah with varying friction/gravity, and a simulated quadruped with varying terrain properties). The cRSSM achieves 15-40% higher return on unseen parameter combinations compared to the standard RSSM, and matches the performance of oracle models that are separately trained on each parameter setting.

## Core Contributions
- Introduces the contextual RSSM (cRSSM) that conditions all world model components on observable context variables, learning a family of dynamics models rather than a single averaged model
- Demonstrates zero-shot generalization to unseen context combinations: training on 10 terrain types enables deployment on novel mixed terrains without retraining
- Proposes context-conditioned imagination: during policy training, imagined rollouts sample diverse context values, naturally producing robust policies without explicit domain randomization of the policy
- Shows that cRSSM separates context-dependent and context-independent dynamics in its latent space, learning a factored representation where context modulates dynamics without corrupting state estimation
- Provides theoretical analysis showing that cRSSM's prediction error scales as O(1/sqrt(n_c)) where n_c is the number of context values seen per state, compared to O(1) for standard RSSM that cannot reduce context-related error
- Introduces a context inference module that estimates c from proprioceptive history when context is not directly observable, enabling deployment without explicit context measurement
- Achieves 15-40% improvement over standard RSSM on zero-shot generalization benchmarks while adding only 5-10% computational overhead

## Methodology Deep-Dive
The cRSSM modifies the standard RSSM by concatenating a context vector c to the inputs of all learned components. The deterministic state update becomes h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}, c), where c modulates the dynamics through the GRU's gating mechanism. The stochastic posterior becomes q(z_t | h_t, o_t, c), and the prior becomes p(z_t | h_t, c). The decoder and reward predictor are similarly conditioned: p(o_t | h_t, z_t, c) and p(r_t | h_t, z_t, c).

The context vector c can represent any environment parameter that affects dynamics: continuous values (mass, friction coefficient, gravity) are directly used; categorical variables (terrain type, robot morphology) are one-hot encoded; and multi-dimensional contexts (terrain heightmap statistics, full dynamics parameter sets) are projected through a learned embedding layer before concatenation. The paper explores context dimensions from 1 (single scalar) to 32 (rich context embedding).

Training follows the DreamerV3 paradigm with modification. During data collection, each episode is associated with a context vector c_i (either observed or recorded from the simulator). The replay buffer stores (o, a, r, c) tuples. The world model is trained on batches where each sequence has a consistent context but different sequences may have different contexts. The KL loss aligns the context-conditioned prior with the context-conditioned posterior: L_KL = KL[q(z_t|h_t,o_t,c) || p(z_t|h_t,c)], ensuring that the prior can predict stochastic states from context and deterministic history alone.

Context-conditioned imagination is the key innovation for policy training. Standard DreamerV3 imagines rollouts from replay buffer states using the prior. The cRSSM imagines rollouts with sampled context values: for each starting state s_0 from the buffer, multiple imagined trajectories are generated with different c values sampled from a context distribution p(c). The policy is trained to maximize the average return across these diverse imagined scenarios, naturally producing a robust policy. This is more efficient than domain randomization of the actual environment because imagination is orders of magnitude faster than simulation.

The context inference module addresses the deployment scenario where context is not directly observable. An inference network q_inf(c | o_{t-T:t}) maps a window of proprioceptive observations to a context estimate. This is trained alongside the world model using the cRSSM's reconstruction loss: if the inferred context enables accurate world model predictions, it must be correct. The inference module is trained with a combination of reconstruction-based loss and an optional supervised loss when ground-truth context is available during simulation training.

A key architectural detail is the FiLM (Feature-wise Linear Modulation) conditioning mechanism. Rather than simple concatenation, the context vector modulates intermediate features of the GRU and decoder via learned scale and bias: feature_out = gamma(c) * feature_in + beta(c), where gamma and beta are linear projections of c. FiLM conditioning provides more expressive context modulation than concatenation, as it allows multiplicative interactions between context and features. The paper shows FiLM conditioning improves zero-shot generalization by 8-12% over concatenation.

The factored latent space analysis reveals that the cRSSM naturally separates context-dependent dynamics (encoded in the GRU's context-modulated gates) from context-independent state (encoded in the stochastic variable z_t). This factorization means that the stochastic latent z_t represents the robot's state in a context-invariant way, while the dynamics model adapts to the context through the GRU's deterministic pathway. This is verified by showing that z_t distributions are similar across contexts for the same physical state, while h_t distributions differ.

## Key Results & Numbers
- CartPole with varying pole length (0.5-2.0x): cRSSM achieves 95% of oracle performance on unseen lengths, vs 71% for standard RSSM
- HalfCheetah with varying friction (0.2-2.0): cRSSM achieves 88% oracle performance on unseen friction values, vs 62% for standard RSSM
- Quadruped with varying terrain: cRSSM matches oracle on 8/10 trained terrains and achieves 83% oracle performance on 5 unseen terrains, vs 54% for standard RSSM
- Zero-shot to combined contexts (novel friction + novel terrain): cRSSM achieves 76% oracle, standard RSSM achieves 41%
- FiLM conditioning improves generalization by 8-12% over concatenation-based context conditioning
- Context inference from proprioceptive history achieves 91% classification accuracy for terrain type and 0.85 R-squared for continuous parameters (friction, mass)
- Computational overhead: 5-10% additional training time (FiLM adds minimal parameters); inference speed identical to standard RSSM once context is determined
- Latent space factorization: mutual information between z_t and c is 0.02 nats (near-independent), confirming clean separation

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The cRSSM directly addresses a key challenge in Mini Cheetah training: making the world model aware of domain randomization parameters. In standard DreamerV3 training with domain randomization, the world model must learn an averaged dynamics that performs acceptably across all randomized conditions. With cRSSM, the world model explicitly conditions on the randomization parameters (friction, mass, motor strength, terrain height), learning a sharper, more accurate model for each condition.

For Mini Cheetah's sim-to-real transfer, the context inference module is crucial: during real-world deployment, the context (true friction, terrain type) is unknown, but the inference module estimates it from proprioceptive history, enabling the world model to adapt to real-world conditions. The FiLM conditioning mechanism ensures that the context modulation is expressive enough to capture the nonlinear effects of different physical parameters on Mini Cheetah's dynamics.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The cRSSM directly enhances the RSSM component in Project B's Controller level. By conditioning the world model on context from the CPTE (terrain latent) and the Primitives level (current locomotion mode), the Controller's world model becomes context-aware, enabling more accurate dynamics prediction and better imagination-based policy training.

Specific integration points: (1) The CPTE output (terrain latent vector z_terrain) serves as the context vector c for the cRSSM. This means the Controller's world model learns terrain-specific dynamics, predicting how Cassie's joints respond to motor commands differently on slopes vs stairs vs flat ground. (2) The Primitives level's current primitive index (walk, run, climb, recover) can be included as a categorical context variable, enabling the world model to switch between dynamics regimes for different locomotion modes. (3) The FiLM conditioning mechanism allows nonlinear interaction between terrain context and dynamics, which is important because terrain effects on Cassie's dynamics are highly nonlinear (e.g., foot slip on ice creates fundamentally different dynamics, not just scaled dynamics).

The context-conditioned imagination is especially powerful for Project B: the Planner level can request imagined rollouts under different hypothetical terrain conditions (what if the terrain becomes icy? what if there are stairs ahead?), enabling anticipatory planning through the Dual Asymmetric-Context Transformer. This essentially gives the Planner a terrain-conditioned simulator for fast mental rehearsal.

## What to Borrow / Implement
- Replace the standard RSSM in Project B's Controller level with a cRSSM conditioned on CPTE terrain latent and Primitives mode index via FiLM conditioning
- Use context-conditioned imagination for terrain-aware policy training in the Controller level: sample diverse terrain contexts during imagination to produce robust motor policies
- Implement the context inference module for real-world Cassie deployment where terrain parameters are not directly observable
- For Project A, condition the world model on domain randomization parameters during training and use the context inference module for sim-to-real transfer
- Adopt FiLM conditioning over simple concatenation for all context-dependent modules in both projects

## Limitations & Open Questions
- The context vector must be specified at training time; learning what constitutes relevant context from data alone is not addressed
- Scalability to very high-dimensional context (e.g., full terrain heightmap) may require dimensionality reduction that loses task-relevant information
- The interaction between cRSSM and hierarchical RL architectures is not explored; how context should flow between hierarchy levels is an open question
- Context inference from proprioceptive history introduces a delay (T timesteps) that may be too slow for rapidly changing terrains or dynamic obstacles
