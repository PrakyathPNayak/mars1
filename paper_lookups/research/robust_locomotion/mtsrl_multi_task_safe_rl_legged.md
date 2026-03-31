# MTSRL: Multi-Task Safe Reinforcement Learning for Legged Robots

**Authors:** Various
**Year:** 2023 | **Venue:** arXiv / Workshop 2023
**Links:** https://sucro-legged.github.io/MTSRL/

---

## Abstract Summary
MTSRL presents a multi-task safe RL framework for legged robots that uses task-specific critics to reduce negative transfer between objectives. Each task (walking, base height control, pitch, roll) maintains its own critic while sharing a single actor network, enabling robust multi-task locomotion policies. The approach achieves direct sim-to-real transfer on both quadruped and biped hardware.

## Core Contributions
- Multi-task RL architecture with shared actor and task-specific critics, eliminating negative transfer between locomotion objectives
- Safe RL integration ensuring constraint satisfaction (joint limits, contact forces, body orientation) during multi-task training
- Demonstrated sim-to-real transfer on both quadruped and biped hardware platforms without real-world fine-tuning
- Systematic analysis of negative transfer in multi-task locomotion and how task-specific critics mitigate it
- Scalable framework that adds new tasks without degrading performance on existing ones
- Reward decomposition strategy that separates competing objectives into manageable sub-problems
- Comparison against single-critic multi-task baselines showing significant performance improvements

## Methodology Deep-Dive
The key insight of MTSRL is that multi-task locomotion suffers from negative transfer when a single critic must estimate value for competing objectives simultaneously. For example, maximizing forward velocity may conflict with minimizing energy consumption, and a single critic struggles to provide accurate gradients for both. MTSRL addresses this by assigning each task its own critic while maintaining a shared actor that must satisfy all objectives.

The shared actor receives proprioceptive observations and outputs joint-level actions. Each task-specific critic receives the same observations plus the actor's actions and estimates the task-specific return. During policy updates, the actor gradient is computed as a weighted sum of gradients from all task-specific critics. The weighting scheme is adaptive — tasks where the policy currently underperforms receive higher weights, ensuring balanced progress across all objectives. This prevents any single task from dominating the policy update and degrading others.

Safety constraints are integrated through a constrained optimization formulation. Hard constraints on joint position limits, joint velocity limits, contact forces, and body orientation are enforced via a Lagrangian relaxation approach. The Lagrange multipliers are learned alongside the policy, automatically adjusting the constraint strictness based on current violation rates. This ensures that the policy remains safe throughout training without requiring manual tuning of constraint weights.

The task decomposition is domain-informed. For legged locomotion, the tasks are: (1) velocity tracking — following commanded linear and angular velocities; (2) base height control — maintaining desired standing height; (3) body pitch regulation — keeping the torso level in the sagittal plane; and (4) body roll regulation — keeping the torso level in the frontal plane. Each task has a clear physical interpretation and can be evaluated independently, making the multi-critic architecture natural.

Domain randomization is applied during training to ensure sim-to-real transferability. Physical parameters (mass, friction, motor strength, joint damping) are randomized, and each task-specific critic must generalize across these variations. The authors find that task-specific critics are more robust to domain randomization than single critics, as each critic can specialize in understanding how physical parameter changes affect its specific objective.

## Key Results & Numbers
- Multi-task performance improvement of 20-35% over single-critic baselines across locomotion objectives
- Eliminated negative transfer between velocity tracking and energy minimization tasks
- Successful sim-to-real transfer on quadruped hardware (Unitree A1/Go1) with zero fine-tuning
- Successful sim-to-real transfer on biped hardware demonstrating cross-platform generality
- Safety constraint violation rate reduced by >60% compared to unconstrained multi-task baselines
- Adding new tasks (e.g., foot clearance) causes <5% performance degradation on existing tasks
- Training converges 15-25% faster than single-critic alternatives due to reduced gradient interference

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The multi-task framework is directly applicable to Mini Cheetah's diverse locomotion objectives. Project A requires simultaneous velocity tracking, body orientation control, energy efficiency, and gait pattern regulation — exactly the kind of competing objectives where MTSRL excels. The task-specific critics architecture can replace a single monolithic critic, preventing negative transfer between these objectives. The demonstrated sim-to-real transfer on quadruped hardware (similar scale to Mini Cheetah) validates the approach. The safety constraints (joint limits, contact forces) directly map to Mini Cheetah's operational constraints. The PPO-compatible training procedure integrates with Project A's existing training infrastructure.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The multi-critic architecture is directly relevant to Cassie's multi-objective optimization across hierarchy levels. At the controller level, Cassie must balance velocity tracking, balance maintenance, energy efficiency, and gait quality — objectives that benefit from task-specific critics. The safe RL formulation connects to Project B's LCBF safety layer, providing an alternative or complementary approach to constraint enforcement. The demonstrated biped transfer validates the approach for Cassie's morphology. The scalable task addition aligns with Project B's need to incorporate new skills (primitives) without degrading existing capabilities. The multi-critic structure could be applied at multiple levels of the hierarchy — the planner level could use task-specific critics for navigation objectives, while the controller level uses them for locomotion objectives.

## What to Borrow / Implement
- Implement the multi-critic architecture (shared actor, task-specific critics) for both projects' PPO training
- Adopt the adaptive task weighting scheme to balance competing locomotion objectives
- Use the Lagrangian safety constraint formulation as a complement to Project B's LCBF layer
- Apply the task decomposition (velocity, height, pitch, roll) to Mini Cheetah's reward structure
- Use the negative transfer analysis methodology to diagnose optimization issues in current training
- Consider the task-specific critic architecture at multiple levels of Project B's hierarchy

## Limitations & Open Questions
- Task decomposition requires domain expertise; poorly decomposed tasks may not benefit from multi-critic approach
- Adaptive weighting scheme adds hyperparameters (update rates, initial weights) that need tuning
- Safety constraints are enforced in expectation, not pointwise — occasional violations can still occur
- Scalability to very large numbers of tasks (>10) is not thoroughly evaluated
- How does the multi-critic approach interact with hierarchical RL where different hierarchy levels have different objectives?
- Can the task-specific critics be combined with learned world models for sample-efficient multi-task training?
- Limited evaluation on highly dynamic tasks (running, jumping, recovery) where task interactions are more complex
