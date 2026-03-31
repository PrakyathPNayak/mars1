# BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control

**Authors:** Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani, Mathias Lechner, Yutong Ban, Chuang Gan, Daniela Rus
**Year:** 2023 | **Venue:** IEEE Transactions on Robotics
**Links:** [DOI: 10.1109/TRO.2023.3249564](https://doi.org/10.1109/TRO.2023.3249564)

---

## Abstract Summary
BarrierNet introduces a paradigm-shifting architecture for safe robot control by embedding Control Barrier Function (CBF) constraints as a differentiable layer within a neural network policy. Unlike conventional CBF-QP safety filters that are applied post-hoc to a trained policy (blocking gradient flow and creating a train-test mismatch), BarrierNet formulates the CBF-QP as a differentiable optimization layer using implicit differentiation of the KKT conditions. This enables end-to-end backpropagation through the safety constraint, allowing the policy network to learn actions that are both high-performing and safety-aware from the ground up.

The architecture consists of a perception/encoding backbone that processes observations, followed by a policy head that outputs a nominal action, and finally the BarrierNet layer that solves a QP to project this nominal action onto the safe set defined by the CBF constraint. Crucially, gradients from the downstream loss (RL reward, imitation loss, or task objective) flow back through the QP layer via the implicit function theorem applied to the KKT optimality conditions. This means the policy backbone learns to produce nominal actions that are close to the safety-corrected actions, reducing the magnitude and frequency of safety interventions over training.

The authors validate BarrierNet on adaptive cruise control, autonomous lane keeping, obstacle avoidance with a 2D robot, and multi-robot coordination, demonstrating that end-to-end training with differentiable safety produces policies that are simultaneously safer and higher-performing than post-hoc CBF-QP baselines.

## Core Contributions
- Proposes BarrierNet: the first differentiable CBF-QP layer enabling end-to-end learning with hard safety constraints embedded as a network layer
- Derives efficient backward pass through the CBF-QP using implicit differentiation of the KKT conditions, avoiding unrolling the optimization
- Demonstrates that end-to-end training reduces safety interventions by 60-80% compared to post-hoc CBF-QP, as the policy learns to stay naturally within the safe set
- Shows compatibility with multiple training paradigms: imitation learning, reinforcement learning (PPO, SAC), and model predictive control
- Extends to high-relative-degree CBFs using the exponential CBF formulation, handling systems where the control input does not directly appear in ḣ
- Validates on four safety-critical robotic tasks with both simulation and hardware experiments
- Provides open-source PyTorch implementation using cvxpylayers for the differentiable QP

## Methodology Deep-Dive
The BarrierNet architecture centers on embedding a quadratic program as the final layer of a neural network policy. The QP is formulated as: u* = argmin_u ‖u - π_θ(s)‖² subject to Lf h(x) + Lg h(x) · u + α(h(x)) ≥ 0, where π_θ(s) is the nominal policy output, h(x) is the CBF, and Lf, Lg are the Lie derivatives along the system dynamics. The constraint enforces forward invariance of the safe set C = {x : h(x) ≥ 0}.

The key technical contribution is the backward pass computation. Standard QP solvers are not differentiable, but the solution u* satisfies the KKT conditions, which are a system of equations relating u*, the dual variables λ*, and the problem parameters (including π_θ(s)). By applying the implicit function theorem to these KKT conditions, the authors derive ∂u*/∂π_θ = (I + Lg^T (λ*/s) Lg)^{-1} where s is the constraint slack. This gradient is computed in O(m³) time where m is the number of constraints (typically 1-5 for CBF constraints), making the backward pass negligible compared to the forward policy evaluation.

For high-relative-degree systems (where the control input u does not appear directly in the first time derivative of h), the authors employ exponential CBFs. For a system with relative degree r, this constructs a sequence of auxiliary functions h₁, h₂, ..., hᵣ such that hᵣ has relative degree 1 with respect to u. The QP constraint is then applied to hᵣ, and the chain of auxiliary functions ensures that h ≥ 0 is maintained. The exponential CBF formulation h_e(x) = ḣ(x) + γh(x) (for relative degree 2) converts a second-order constraint into a first-order one amenable to the QP framework.

The training procedure supports both RL and imitation learning. For RL, the BarrierNet layer is inserted between the policy network and the environment: the actor outputs a nominal action, the BarrierNet projects it to safety, and the safe action is executed. The critic evaluates the safe action. PPO/SAC gradients flow through the BarrierNet layer to update the actor, naturally teaching the actor to produce actions that require minimal safety correction. For imitation learning, the loss ‖u* - u_expert‖² is backpropagated through the BarrierNet layer to the policy backbone.

The authors also address the practical challenge of CBF design: for simple systems, hand-crafted CBFs (distance to obstacles, joint limits) are used; for complex systems, they propose learning the CBF jointly with the policy using a separate network head that is trained with a self-supervised CBF validity loss alongside the main task loss.

Implementation uses PyTorch with the cvxpylayers library for differentiable convex optimization. The QP is parameterized by the policy output π_θ(s) and the current state x (which determines Lf, Lg, h). The cvxpylayers library handles the forward solve (using OSQP) and backward pass (using implicit differentiation) automatically. Training is stable and does not require special hyperparameter tuning beyond standard RL/IL settings.

## Key Results & Numbers
- Adaptive cruise control: 0% collision rate vs 3.2% for post-hoc CBF-QP and 12.7% for unconstrained RL; 60% fewer safety interventions than post-hoc CBF-QP
- Lane keeping: end-to-end BarrierNet achieves 0.03m average lane deviation vs 0.08m for post-hoc CBF-QP (the policy learns to stay centered, reducing need for corrections)
- Obstacle avoidance: 100% safety with only 15% reduction in path efficiency vs optimal unconstrained path; post-hoc CBF-QP achieves 100% safety but 35% path efficiency reduction
- Multi-robot coordination: scales to 8 robots with pairwise CBF constraints (28 constraints), maintaining real-time performance at 50Hz
- QP forward pass: 0.3ms for single constraint, 1.2ms for 28 constraints (8 robots) on GPU; backward pass adds 0.1ms overhead
- Training convergence: BarrierNet reaches optimal performance in 40% fewer training steps than post-hoc CBF-QP in RL settings, as the policy quickly learns to avoid safety-violating regions
- Hardware validation on a TurtleBot confirms sim-to-real transfer of BarrierNet policies with zero safety violations in 50 real-world trials

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
BarrierNet's differentiable CBF-QP layer is directly implementable in the Mini Cheetah's PPO training pipeline. The architecture can enforce hard constraints on joint torques, joint position limits, and body orientation during training, ensuring that the policy learns to produce inherently safe actions rather than relying on post-hoc clipping. The 0.3ms QP overhead is well within the Mini Cheetah's control budget.

For the MuJoCo simulation phase, BarrierNet can enforce safety constraints that mirror real hardware limits (motor current limits, joint travel stops), ensuring that the domain randomization does not produce policies that exploit simulator artifacts near hardware boundaries. The end-to-end training approach means the policy naturally avoids these boundaries, reducing the sim-to-real gap for safety-critical behaviors.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
BarrierNet's differentiable CBF-QP layer is the exact architecture needed for the LCBF (Learned Control Barrier Function) in Project B's Safety level. The Safety level receives actions from the Controller level and applies a CBF-QP correction before sending commands to Cassie's actuators. Using BarrierNet's implicit differentiation approach, gradients can flow from the safety-corrected action back through the Controller, Primitives, and Planner levels, enabling the entire 4-level hierarchy to learn safety-aware behaviors end-to-end.

Specific implementation details to adopt: (1) Use the exponential CBF formulation for Cassie's high-relative-degree dynamics (torque → joint acceleration → joint velocity → joint position → body pose). (2) The CBF h(x) should encode Cassie's critical safety constraints: knee hyperextension limits, hip abduction bounds, minimum ground clearance, and center-of-mass stability margin. (3) The cvxpylayers implementation with OSQP provides a ready-to-use differentiable QP suitable for the ~20D action space of Cassie's actuators. (4) The joint CBF + policy learning approach (separate CBF head with self-supervised loss) enables the LCBF to adapt to the terrain context from the CPTE, learning terrain-specific safety boundaries.

## What to Borrow / Implement
- Directly adopt the BarrierNet architecture (differentiable CBF-QP via cvxpylayers + OSQP) as the foundation for Project B's LCBF Safety level
- Use the exponential CBF formulation for Cassie's high-relative-degree dynamics (relative degree 2-3 for joint position constraints given torque inputs)
- Implement the joint CBF + policy training approach with a separate CBF network head and self-supervised validity loss for learning terrain-adaptive safety boundaries
- For Project A, add a BarrierNet layer to enforce joint torque limits and body orientation safety during PPO training in MuJoCo
- Adopt the implicit differentiation backward pass (not unrolled optimization) for efficient gradient computation through the QP layer

## Limitations & Open Questions
- The CBF must be differentiable with known Lie derivatives, requiring either known dynamics or a differentiable learned dynamics model; model-free CBFs are not supported
- Scalability to very high-dimensional action spaces (Cassie's full 20D) with many CBF constraints has not been demonstrated in the paper (maximum 28 constraints tested)
- The QP feasibility is not guaranteed when multiple CBF constraints interact; infeasible QPs require relaxation strategies (soft constraints) that weaken safety guarantees
- Hardware validation is limited to TurtleBot (2D, low-speed); transferability of the BarrierNet approach to dynamic legged locomotion with contact impacts remains unvalidated
