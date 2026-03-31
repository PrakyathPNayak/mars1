# CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning

**Authors:** Elliot Chane-Sane, Pierre-Alexandre Leziart, Thomas Flayols, Olivier Stasse, Philippe Souères, Nicolas Mansard
**Year:** 2024 | **Venue:** ICRA 2024
**Links:** https://constraints-as-terminations.github.io/

---

## Abstract Summary
CaT proposes treating safety constraints as stochastic termination conditions in RL rather than using Lagrangian penalties. When a constraint is violated, the episode terminates with a probability proportional to the violation severity. This naturally encourages the policy to stay within safe regions without complex dual optimization and was validated on real quadruped robots crossing challenging terrain.

## Core Contributions
- Reformulates safety constraints as probabilistic episode terminations, avoiding Lagrangian dual optimization entirely
- Demonstrates that termination probability proportional to constraint violation severity provides a natural curriculum for learning safe behaviors
- Eliminates the need for dual variable tuning, simplifying hyperparameter selection compared to constrained RL baselines
- Validates the approach on real quadruped robots traversing challenging terrain including gaps and stairs
- Shows improved constraint satisfaction rates over standard Lagrangian methods across multiple benchmarks
- Provides theoretical justification linking stochastic terminations to constraint satisfaction in the CMDP framework

## Methodology Deep-Dive
The core idea behind CaT is elegantly simple: instead of adding constraint violation costs to the reward function or using Lagrangian multipliers to enforce constraints, the method treats each constraint violation as a potential episode termination. The termination is stochastic — the probability of ending the episode scales with how severely the constraint is violated. Small violations have a low chance of termination, while large violations almost certainly end the episode.

This formulation transforms the constrained MDP into a standard unconstrained MDP with modified dynamics. The agent learns to avoid constraint violations not because of explicit penalty terms, but because violations shorten the expected episode length and thus reduce cumulative reward. This creates an implicit trade-off between task performance and safety without requiring any dual optimization.

The implementation is straightforward: at each timestep, the constraint function is evaluated. If a violation occurs, a Bernoulli random variable determines whether the episode terminates. The termination probability is computed as a sigmoid or linear function of the violation magnitude. This means the agent experiences a soft boundary rather than a hard cutoff, enabling smoother policy optimization.

Training uses standard PPO without any modifications to the policy gradient algorithm. The only change is in the environment's termination logic. This makes CaT a drop-in replacement for existing RL training pipelines — no new loss terms, no additional networks, and no Lagrangian multiplier schedules to tune.

The real-world experiments deploy the trained policies on quadruped robots navigating gaps, stairs, and uneven terrain. The sim-to-real transfer is facilitated by the fact that the constraint-as-termination formulation does not introduce additional sim-to-real gaps beyond those already present in the dynamics model.

## Key Results & Numbers
- Outperforms Lagrangian-based methods in constraint satisfaction rate across multiple locomotion benchmarks
- Achieves comparable or better task performance while maintaining stricter safety margins
- Real-world deployment on quadruped robots crossing gaps and stairs with zero constraint violations during testing
- Training stability significantly improved — no oscillations from dual variable updates
- Implementation requires only ~10 lines of code change to standard PPO training loops
- No hyperparameter sensitivity to constraint thresholds, unlike Lagrangian methods which require careful tuning of multiplier learning rates

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
CaT is directly applicable to Mini Cheetah's PPO training pipeline. The method can enforce safety constraints (joint limits, torque limits, body orientation bounds) during training without modifying the PPO algorithm itself. Since Mini Cheetah uses standard PPO at 500 Hz PD control, CaT's termination-based approach integrates seamlessly — constraint violations during MuJoCo simulation simply trigger stochastic episode resets. This is far simpler than implementing Lagrangian penalties or CBF layers for training-time safety. Domain randomization and curriculum learning remain fully compatible since CaT only modifies termination conditions.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
CaT offers a compelling alternative or complement to the LCBF (Learned Control Barrier Function) safety layer in Project B's hierarchy. Rather than learning a separate CBF network and solving a QP at runtime, CaT could enforce safety constraints directly during PPO training of the controller level. For the multi-level hierarchy (Planner→Primitives→Controller→Safety), CaT could replace or simplify the Safety level by baking constraint satisfaction into the Controller's policy during training. This is particularly relevant since Project B uses PPO and Option-Critic — both are compatible with CaT's termination-based approach. The method could also complement the Adversarial Curriculum by adding constraint-aware terminations during adversarial training.

## What to Borrow / Implement
- Implement stochastic termination wrapper for MuJoCo environments in both projects
- Replace or augment Lagrangian constraint handling with CaT for training-time safety in PPO
- Use CaT for joint limit, torque limit, and orientation constraints during Mini Cheetah training
- Evaluate CaT as a simpler alternative to LCBF for Cassie's safety enforcement
- Combine CaT with domain randomization — termination probabilities can be randomized as part of sim-to-real transfer
- Test hybrid approach: CaT during training + LCBF at deployment for runtime safety guarantees

## Limitations & Open Questions
- Stochastic terminations can reduce effective episode length, potentially slowing learning of long-horizon tasks
- No formal safety guarantees at deployment time — only training-time constraint encouragement
- Unclear how well the approach scales to many simultaneous constraints (>10)
- The termination probability function (sigmoid vs. linear) requires some tuning
- May not be sufficient for hard safety constraints where any violation is catastrophic
- Interaction with curriculum learning needs investigation — early curriculum stages may have different constraint requirements
