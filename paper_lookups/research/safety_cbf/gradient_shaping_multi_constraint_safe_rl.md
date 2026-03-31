# Gradient Shaping for Multi-Constraint Safe Reinforcement Learning

**Authors:** Yihang Yao et al.
**Year:** 2024 | **Venue:** L4DC 2024 (PMLR)
**Links:** https://proceedings.mlr.press/v242/yao24a.html

---

## Abstract Summary
This paper addresses training instability in multi-constraint safe RL by modifying gradient directions during policy updates. Gradient Shaping (GradS) projects conflicting constraint gradients to find feasible update directions that satisfy all constraints simultaneously. The method improves both training efficiency and constraint satisfaction rates compared to standard Lagrangian approaches, particularly in high-DoF robotic systems.

## Core Contributions
- Identifies gradient conflict as a key source of training instability in multi-constraint safe RL
- Proposes Gradient Shaping (GradS), which projects conflicting gradients onto a feasible cone satisfying all constraints
- Demonstrates that GradS outperforms standard Lagrangian methods when multiple constraints are active simultaneously
- Provides theoretical analysis showing GradS guarantees descent in a direction satisfying all constraints when feasible
- Scales effectively to high-DoF robotic systems with many simultaneous safety constraints
- Compatible with standard policy gradient algorithms (PPO, TRPO) as a drop-in gradient modifier

## Methodology Deep-Dive
The fundamental problem addressed is gradient conflict in multi-constraint optimization. When multiple constraints are active, their gradient directions may oppose each other — satisfying one constraint's gradient may worsen another. Standard Lagrangian methods handle this by forming a weighted sum of constraint gradients, but the resulting direction may not actually improve all constraints simultaneously. This leads to oscillating training dynamics where the policy repeatedly violates different constraints.

Gradient Shaping resolves this by projecting the task gradient onto the intersection of half-spaces defined by each constraint gradient. If the task gradient already satisfies all constraint gradients (i.e., it has non-negative inner product with each constraint's improvement direction), no modification is needed. Otherwise, GradS finds the closest direction to the original task gradient that lies within the feasible cone. This is formulated as a quadratic program over the gradient space.

The QP for gradient projection has a closed-form solution when only two constraints conflict, and is efficiently solvable for moderate numbers of constraints using standard QP solvers. The computational overhead is minimal since it operates on the gradient vector (same dimension as the policy parameters) and only needs to be solved once per policy update step.

Implementation with PPO is straightforward: after computing the policy gradient for the task reward and the gradients for each constraint cost, GradS projects the combined gradient before the PPO update step. The constraint gradients are obtained by differentiating the constraint cost functions with respect to the policy parameters. The PPO clipping mechanism operates on the shaped gradient, maintaining the trust region properties.

The paper validates GradS on multi-constraint locomotion tasks where the robot must simultaneously satisfy joint angle limits, torque constraints, velocity bounds, and contact force limits. Standard Lagrangian methods show significant oscillation between constraints, while GradS maintains steady improvement across all constraints. The improvement is most pronounced when constraints are tightly coupled (e.g., joint angle and torque limits on the same joint).

## Key Results & Numbers
- Reduces constraint violation rate by 40-60% compared to standard Lagrangian methods in multi-constraint settings
- Training convergence speed improves by 25-35% due to elimination of gradient conflict oscillations
- Computational overhead of gradient projection is <5% of total training time
- Scales to 10+ simultaneous constraints without degradation in performance
- Maintains task reward within 5% of unconstrained baselines while satisfying all constraints
- Outperforms PCGrad (multi-task gradient method) when applied to multi-constraint safe RL

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Mini Cheetah training with PPO can benefit from GradS when multiple safety objectives are imposed simultaneously (12 joint limits, 12 torque limits, body orientation constraints, foot slip constraints). The gradient projection approach avoids the oscillation issues common with multi-constraint Lagrangian training. Implementation is straightforward as a wrapper around PPO's gradient computation. The 12-DoF system with PD control at 500 Hz generates many coupled constraints that would benefit from GradS's conflict resolution.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Project B's hierarchy imposes multiple simultaneous constraints at the controller and safety levels: CBF constraints from the LCBF layer, joint angle limits, torque bounds, balance constraints (Differentiable Capture Point), and gait phase constraints. These constraints are tightly coupled — CBF constraints on leg angles directly conflict with torque limits during aggressive maneuvers. GradS can resolve these conflicts during PPO training of the controller level, providing stable gradients that satisfy all constraints simultaneously. The QP formulation in GradS is analogous to the QP in the LCBF safety filter, suggesting a unified optimization framework. For the Adversarial Curriculum, GradS prevents the adversary from exploiting gradient conflicts between safety constraints.

## What to Borrow / Implement
- Implement GradS as a gradient modifier wrapper for PPO in both projects
- Use GradS when training with more than 2 simultaneous safety constraints
- Apply to Mini Cheetah's joint limit + torque limit + orientation constraints
- Integrate with Project B's LCBF layer — use GradS during training and LCBF QP during deployment
- Benchmark GradS against PID-Lagrangian (from Paper 3) for multi-constraint scenarios
- Use the gradient conflict detection mechanism to diagnose training instabilities in existing pipelines

## Limitations & Open Questions
- QP for gradient projection becomes expensive with very large numbers of constraints (>50)
- When no feasible direction exists (all constraints conflict with the task), GradS defaults to minimizing constraint violation — may stall task progress
- The approach assumes constraint gradients are accurate, which may not hold with noisy gradient estimates from small batch sizes
- Interaction with domain randomization (which changes constraint boundaries) is unexplored
- No theoretical guarantees on convergence rate, only on descent direction feasibility
- Has not been tested specifically on bipedal locomotion tasks with balance constraints
