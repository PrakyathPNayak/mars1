# Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion

**Authors:** Yunho Kim et al.
**Year:** 2024 | **Venue:** IEEE Transactions on Robotics
**Links:** https://arxiv.org/abs/2308.12517

---

## Abstract Summary
This paper proposes replacing the many manually-tuned reward terms typically used in RL for legged locomotion with explicit constraints, dramatically simplifying the training process. The authors show that tuning a single reward coefficient while handling other objectives as constraints is sufficient for robust locomotion control. The approach is validated across multiple real and simulated robots on difficult terrain.

## Core Contributions
- Reformulation of multi-objective RL locomotion as constrained optimization with a single reward term
- Dramatic simplification of reward engineering: one tunable coefficient instead of dozens
- Constraint-based formulation naturally handles conflicting objectives (speed vs. energy vs. stability)
- Validated on multiple robot platforms including real-world deployment
- Robust locomotion on difficult terrain without extensive reward tuning
- Principled approach to balancing locomotion objectives using Lagrangian relaxation
- Demonstrates generality across quadruped and bipedal platforms

## Methodology Deep-Dive
Traditional RL for legged locomotion requires designing a complex reward function with many weighted terms: velocity tracking, energy minimization, foot clearance, smoothness, contact penalties, orientation targets, and more. Each term needs careful weighting, and these weights interact in complex ways. The authors argue that most of these terms are better expressed as constraints rather than rewards.

The key insight is that objectives like "don't use too much energy" or "maintain body height within a range" are naturally constraints (inequalities or equalities) rather than rewards to be maximized. By formulating them as constraints, the optimizer automatically determines the appropriate trade-off rather than requiring manual weight tuning. The single remaining reward term captures the primary objective—typically velocity tracking or task completion.

The constrained optimization is solved using a primal-dual approach based on Lagrangian relaxation. Each constraint is associated with a dual variable (Lagrange multiplier) that is automatically adjusted during training. When a constraint is violated, its multiplier increases, pushing the policy to satisfy it; when it's satisfied with margin, the multiplier decreases, allowing the policy to focus on the primary reward. This self-tuning mechanism replaces manual reward weight adjustment.

The implementation builds on PPO with an augmented Lagrangian method. The policy gradient is computed on the Lagrangian (reward minus weighted constraint violations), and the dual variables are updated based on constraint satisfaction. The constraint thresholds themselves are the main hyperparameters, but these are much more interpretable than reward weights—e.g., "torque should stay below 30 Nm" is easier to specify than "torque penalty weight should be 0.003."

Experiments span multiple platforms: Unitree A1 and Go1 quadrupeds, ANYmal, and bipedal robots, both in simulation and on real hardware. The constraint-based approach matches or exceeds hand-tuned reward functions while requiring far less tuning effort. Ablations show that converting any single reward term to a constraint improves robustness, and converting all auxiliary terms provides the best results.

## Key Results & Numbers
- Single reward coefficient replaces 10-20+ manually tuned reward weights
- Comparable or superior locomotion quality to extensively hand-tuned baselines
- Validated on Unitree A1, Go1, ANYmal, and bipedal platforms
- Real-world deployment on rough terrain, stairs, and slopes
- Constraint satisfaction rates > 95% during deployment
- Training time comparable to standard PPO (no significant overhead from dual updates)
- Reduced hyperparameter sensitivity: performance robust across wide range of constraint thresholds
- Transfer across robot morphologies with minimal re-tuning

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper directly addresses one of the most time-consuming aspects of RL training for Mini Cheetah: reward function design and tuning. The current PPO training pipeline likely uses multiple weighted reward terms for velocity tracking, energy, smoothness, and stability. Converting auxiliary objectives to constraints would dramatically reduce tuning effort and improve robustness. The constraint formulation is compatible with the existing PPO training in MuJoCo and can be implemented by augmenting the reward computation with Lagrangian multipliers. The 500 Hz PD control loop is unaffected—constraints operate at the RL training level. Domain randomization benefits from the constraint approach, as constraint thresholds are more invariant to dynamics changes than reward weights.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The constraint-based formulation aligns naturally with Project B's safety-focused design philosophy. The LCBF (Learned Control Barrier Function + QP) already operates on a constraint-satisfaction paradigm; extending this philosophy to the RL training itself creates a unified constraint-based framework. Each level of the 4-level hierarchy could have its own constraints: the Planner level constrains path feasibility, the Primitives level constrains transition smoothness, the Controller level constrains joint limits and energy, and the Safety level enforces stability via CBFs. The Lagrangian relaxation approach could be integrated with the Option-Critic framework at the Primitives level, where constraints ensure safe option transitions. For Cassie specifically, bipedal balance constraints (CoM within support polygon, capture point within reachable set) are natural fits for the constraint formulation.

## What to Borrow / Implement
- Replace auxiliary reward terms with constraints in both Project A and B's PPO training
- Implement Lagrangian relaxation with automatic dual variable updates in the training loop
- Define interpretable constraint thresholds for joint torques, energy, body orientation, and foot clearance
- Use the constraint framework to encode safety requirements at each level of Project B's hierarchy
- Integrate constraint-based training with the adversarial curriculum to ensure constraints hold under perturbation
- Benchmark single-reward + constraints vs. current multi-reward approach on Mini Cheetah

## Limitations & Open Questions
- Constraint thresholds still require some domain knowledge to set appropriately
- Lagrangian relaxation may oscillate if learning rates for dual variables are not tuned
- Hard constraints cannot be guaranteed during training exploration—only soft enforcement
- Interaction between multiple constraints can lead to infeasible regions during early training
- How to handle time-varying constraints (e.g., different constraints for different terrain types)?
- Scaling to very large numbers of constraints (dozens) may slow convergence
