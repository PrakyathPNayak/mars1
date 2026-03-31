# Hidden Parameter Recurrent State Space Models for Changing Dynamics Scenarios (HiP-RSSM)

**Authors:** Various
**Year:** 2022 | **Venue:** ICLR 2022
**Links:** https://openreview.net/forum?id=ds8yZOUsea

---

## Abstract Summary
HiP-RSSM augments the RSSM latent space with hidden parameter variables that capture environment-specific dynamics variations (e.g., changing mass, friction). This enables rapid adaptation to non-stationary environments and generalization across related tasks. The model infers hidden parameters from a short context window and uses them to condition the dynamics model.

## Core Contributions
- Augmented the RSSM with explicit hidden parameter variables that encode environment-specific dynamics (mass, friction, damping, etc.)
- Demonstrated rapid adaptation to changing dynamics by inferring hidden parameters from a short context window (~10-50 timesteps)
- Achieved generalization across task families where dynamics vary, outperforming meta-learning baselines (MAML, RL²)
- Provided a principled separation between shared dynamics structure (RSSM backbone) and environment-specific variations (hidden parameters)
- Showed that hidden parameters are interpretable — they correlate with actual physical quantities like mass and friction
- Enabled zero-shot transfer to new dynamics configurations within the training distribution

## Methodology Deep-Dive
The core idea of HiP-RSSM is that many real-world dynamics variations can be captured by a low-dimensional set of hidden parameters. When a robot moves from tile to grass, the underlying dynamics equations don't change fundamentally — only a few parameters (friction coefficient, compliance) shift. HiP-RSSM formalizes this by factoring the RSSM's transition model into a shared structure and a set of task-specific hidden parameters.

The architecture extends the standard RSSM with an additional inference network that estimates hidden parameters ψ from a context window of recent observations and actions. Given a sequence of (o_{t-k:t}, a_{t-k:t}), the hidden parameter encoder produces a distribution q(ψ | context) from which ψ is sampled. This hidden parameter then conditions the RSSM's transition model: the GRU dynamics become h_{t+1} = f_θ(h_t, z_t, a_t, ψ), where ψ modulates the dynamics. The rest of the RSSM (stochastic latent, observation decoder, reward predictor) remains unchanged.

Training uses an amortized variational inference approach. The model is trained across multiple environments with different dynamics parameters (e.g., carts with different masses, surfaces with different frictions). The hidden parameter encoder learns to infer ψ from context, while the dynamics model learns to use ψ to adjust its predictions. A KL regularization term on ψ prevents the model from encoding observation-specific information in the hidden parameters, ensuring they capture only dynamics-relevant factors. During deployment, the model collects a short context window and infers ψ, which then remains fixed (or is periodically updated) for the current environment.

A key insight is that the hidden parameter space is much lower-dimensional than the full latent space. While the RSSM latent state might be 200+ dimensional, the hidden parameter ψ is typically 5-20 dimensional, capturing only the essential dynamics variations. This low dimensionality makes inference fast and robust, requiring only a short context window for accurate estimation. The interpretability of ψ is a bonus: plotting the inferred hidden parameters against true physical quantities shows strong correlation, even though no supervision on ψ is provided.

The training procedure involves sampling different dynamics configurations, collecting episodes, and training the full model end-to-end. At test time, the model encounters a new dynamics configuration, collects a short context window (~10-50 steps), infers ψ, and immediately adapts its predictions and policy. This is much faster than meta-learning approaches that require gradient updates, making it suitable for real-time adaptation.

## Key Results & Numbers
- Fast adaptation to changing dynamics within 10-50 timesteps of context (no gradient updates needed)
- Generalizes across dynamics families: tested on varying mass (0.5x-2x), friction (0.1-1.0), and damping
- Outperforms meta-learning baselines (MAML, RL²) on adaptation speed and final performance
- Hidden parameter dimensionality: 5-20 variables sufficient for capturing dynamics variations
- Zero-shot transfer to new dynamics within training distribution
- Interpretable hidden parameters show >0.9 correlation with true physical quantities
- Context window of 10-50 steps sufficient for accurate hidden parameter inference

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Directly applicable to domain randomization for Mini Cheetah sim-to-real transfer. During training in MuJoCo, the policy encounters randomized mass, friction, and motor parameters. HiP-RSSM could learn to infer these hidden parameters from a short context window on the real robot, enabling the world model (and consequently the policy) to adapt to the true physical parameters without knowing them explicitly. This is more principled than hoping the domain randomization distribution covers the real robot — instead, the model actively infers where in the distribution the real robot falls. The 10-50 step context window (~20-100ms at 500 Hz) is fast enough for real-time adaptation.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Hidden parameters could capture terrain-dependent dynamics shifts, enabling adaptation without retraining when Cassie encounters new surfaces. The planner's RSSM world model would benefit from conditioning on inferred terrain parameters (friction, compliance, slope) rather than trying to encode everything in the general latent state. This aligns with the CPTE (Contrastive Terrain Encoder) module — the hidden parameters provide a complementary, dynamics-focused terrain characterization that could be fused with the CPTE's visual terrain embedding. For the hierarchical architecture, hidden parameters inferred at the planner level could be propagated down to the controller and safety layers, enabling terrain-aware behavior at all levels.

## What to Borrow / Implement
- Augment the RSSM planner with hidden parameter inference for terrain-dependent dynamics adaptation
- Use the context-window inference approach (10-50 steps) for real-time dynamics estimation on both robots
- Implement the factored transition model: shared dynamics structure + hidden parameter conditioning
- Train across a diverse set of dynamics configurations in simulation (mass, friction, terrain type) to learn a general hidden parameter space
- Combine hidden parameter inference with domain randomization: randomize during training, infer during deployment
- Propagate inferred hidden parameters through the hierarchical architecture (planner → primitives → controller)
- Validate interpretability by correlating learned hidden parameters with known physical quantities

## Limitations & Open Questions
- Assumes dynamics variations are low-dimensional and can be captured by a small parameter vector
- May struggle with qualitative dynamics changes (e.g., rigid vs. deformable terrain) that can't be parameterized smoothly
- Context window inference assumes stationarity within the window — may fail during rapid terrain transitions
- Training requires access to diverse dynamics configurations, which may be limited for real-world data
- Interaction with other adaptation mechanisms (domain randomization, adversarial training) not fully explored
- Hidden parameter space may not generalize to out-of-distribution dynamics configurations
- Unclear how to handle multiple simultaneous dynamics changes (e.g., mass shift + terrain change)
