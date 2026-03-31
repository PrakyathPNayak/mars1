# Learning Control Barrier Functions and Their Application in Reinforcement Learning: A Survey

**Authors:** Survey Authors (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv:2404.16879](https://arxiv.org/abs/2404.16879)

---

## Abstract Summary
This comprehensive survey examines the rapidly growing intersection of Control Barrier Functions (CBFs) and deep reinforcement learning, cataloguing data-driven approaches to learning safety certificates for autonomous systems. The paper systematically reviews neural network-based CBF formulations, covering both end-to-end learning paradigms where the CBF is jointly optimized with the policy, and modular approaches where CBFs are trained separately and composed with pre-existing controllers via quadratic programming (QP) safety filters.

The survey identifies three major thrusts in learned CBF research: (1) supervised learning of CBFs from expert demonstrations or known safe sets, (2) self-supervised and Lyapunov-inspired training losses that enforce the CBF decrease condition along trajectories, and (3) reinforcement learning frameworks that embed CBF constraints directly into the policy optimization loop. Verification of learned CBFs remains a central open challenge, as neural networks lack the closed-form guarantees of hand-crafted CBFs, prompting research into SMT-based verification, Lipschitz bounds, and interval bound propagation.

The authors also discuss latent-space safety—learning CBFs in a compressed representation space rather than the full state space—which is particularly relevant for high-dimensional robotic systems where defining explicit safe sets is intractable. Applications span autonomous driving, bipedal and quadruped locomotion, manipulation, and multi-agent coordination, with a growing emphasis on sim-to-real transfer of learned safety guarantees.

## Core Contributions
- Provides the first unified taxonomy of learned CBF methods, categorizing approaches by learning paradigm (supervised, self-supervised, RL-integrated), network architecture, and verification strategy
- Identifies the fundamental tension between CBF expressiveness (complex safe sets) and verifiability (formal guarantees), mapping the Pareto frontier of existing methods
- Reviews integration mechanisms for CBFs in deep RL: CBF-QP safety filters, Lagrangian relaxation of CBF constraints, reward shaping with CBF penalties, and differentiable CBF layers
- Surveys verification techniques for neural CBFs including SMT solvers, Lipschitz-based certificates, abstract interpretation, and probabilistic guarantees
- Catalogs failure modes of learned CBFs: distribution shift, adversarial inputs, sim-to-real degradation, and conservatism from over-approximation
- Identifies open problems including scalability to high-dimensional systems, multi-agent safety composition, and online adaptation of CBFs
- Benchmarks computational costs of CBF-QP solvers at inference time across different robotic platforms

## Methodology Deep-Dive
The survey structures the CBF learning problem around the standard definition: given a dynamical system ẋ = f(x) + g(x)u, a CBF h(x) defines a safe set C = {x : h(x) ≥ 0} with the condition that ḣ(x,u) + α(h(x)) ≥ 0 for some extended class-K function α. Neural CBFs parameterize h as a deep network hθ(x) and learn θ to satisfy the CBF conditions.

For supervised approaches, the survey reviews methods that train hθ from labeled safe/unsafe state pairs using binary cross-entropy or hinge losses, plus trajectory-level losses that enforce the Lie derivative condition ∇hθ · (f + gu) + αhθ ≥ 0 along sampled rollouts. The key technical challenge is that the Lie derivative depends on system dynamics (f, g), leading to three sub-approaches: known-dynamics (analytical Lie derivatives), learned-dynamics (neural ODE or Gaussian process models), and model-free (finite-difference approximations of ḣ from trajectory data).

For RL-integrated approaches, the survey categorizes methods by how CBF constraints enter the optimization. CBF-QP safety filters solve min‖u - π(s)‖² s.t. ḣ(x,u) ≥ -αh(x) at each timestep, projecting the RL policy's action onto the safe set. Lagrangian methods add λ · max(0, -ḣ - αh) to the RL loss and co-optimize the multiplier λ. Differentiable CBF layers (e.g., BarrierNet) embed the QP as a differentiable module enabling gradient flow through the safety constraint.

The verification section is particularly thorough, reviewing exact methods (branch-and-bound for ReLU networks, MILP formulations) and approximate methods (interval bound propagation, Lipschitz estimation, randomized smoothing). The survey notes that exact verification scales exponentially with network size, motivating research into architecture choices (shallow ReLU networks, input-convex networks) that admit more efficient verification.

Latent-space CBF methods are reviewed as an emerging direction where an encoder maps high-dimensional observations (images, point clouds) to a low-dimensional latent space, and the CBF is learned in this latent space. The survey discusses the challenge of ensuring that safety in latent space implies safety in the original state space, reviewing approaches based on decoder error bounds and Lipschitz continuity of the encoder.

## Key Results & Numbers
- Neural CBFs with 2-3 hidden layers (64-256 units) achieve >99% empirical safety rate on tabletop manipulation and planar navigation benchmarks
- CBF-QP safety filters add 0.5-5ms overhead per control step depending on state/action dimensionality and QP solver choice
- Exact verification of ReLU CBFs with >1000 neurons is computationally intractable (hours to days); Lipschitz-based methods scale to 10,000+ neurons but with conservatism
- End-to-end learned CBFs (differentiable QP layer) reduce constraint violations by 40-70% compared to post-hoc CBF-QP filters in locomotion tasks
- Sim-to-real transfer of learned CBFs shows 15-30% degradation in safety rate without domain randomization; with randomization, degradation reduces to 3-8%
- Latent-space CBFs reduce safe set computation from O(n³) in full state space to O(k³) where k << n is the latent dimension

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The survey provides a useful overview of CBF integration strategies for the Mini Cheetah deployment pipeline. For sim-to-real transfer of learned locomotion policies, a CBF-QP safety filter could prevent dangerous joint configurations or excessive ground reaction forces during real-world testing. The computational overhead analysis (0.5-5ms per step) confirms feasibility for Mini Cheetah's 50Hz control loop.

However, Project A's primary focus is on learning robust locomotion via PPO with domain randomization and curriculum learning in MuJoCo, where safety is primarily enforced through reward shaping and episode termination rather than formal CBF constraints. The survey's insights on reward shaping with CBF penalties could be incorporated as an auxiliary training signal without requiring a full CBF-QP layer.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This survey is a critical reference for Project B's Safety level, which uses a Learned Control Barrier Function (LCBF) with QP-based action correction. The taxonomy of learning approaches directly informs the LCBF training strategy: the self-supervised trajectory-level loss (enforcing ḣ + αh ≥ 0 along rollouts) is the most applicable paradigm since Project B's safety level operates on the output of the Controller level rather than direct motor commands.

The survey's analysis of differentiable CBF layers (BarrierNet-style) is essential for Project B's architecture, as the LCBF must allow gradient flow from the safety correction back to the upstream policy for end-to-end fine-tuning. The discussion of latent-space CBFs is also relevant given that the Controller level operates in a learned latent representation. The verification section provides a roadmap for validating the LCBF before Cassie hardware deployment.

## What to Borrow / Implement
- Adopt the self-supervised CBF training loss (trajectory-level Lie derivative enforcement) for LCBF training in Project B's Safety level
- Use the differentiable QP formulation (OptNet/cvxpylayers) to enable gradient flow through the CBF-QP safety filter
- Implement domain randomization of dynamics parameters during CBF training to improve sim-to-real robustness (survey reports 3-8% degradation vs 15-30% without)
- Consider Lipschitz-bounded network architectures for the LCBF to enable scalable approximate verification
- Apply reward shaping with CBF penalty terms as an auxiliary training signal for Project A's PPO training

## Limitations & Open Questions
- The survey focuses primarily on continuous-time CBF formulations; discrete-time CBF theory (more relevant to RL's discrete timesteps) receives less attention
- Scalability of verification methods to the state dimensions of real legged robots (30+ DoF for Cassie) remains largely untested
- The interaction between learned CBFs and hierarchical RL architectures (like Project B's 4-level hierarchy) is not addressed
- Sim-to-real transfer analysis is limited to relatively simple systems; complex legged locomotion with contact dynamics presents unique challenges not covered
