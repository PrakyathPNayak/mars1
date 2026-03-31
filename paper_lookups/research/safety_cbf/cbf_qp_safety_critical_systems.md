# Control Barrier Functions for Safety Critical Systems: Theory and Applications

**Authors:** Aaron Ames, Xiangru Xu, Jessy Grizzle, Paulo Tabuada
**Year:** 2019 | **Venue:** IEEE Transactions on Automatic Control / Survey
**Links:** Survey/Tutorial Paper (IEEE TAC / CDC)

---

## Abstract Summary
This paper provides a comprehensive treatment of Control Barrier Functions (CBFs) as a framework for ensuring safety in dynamical systems. Safety is formalized as the forward invariance of a safe set C = {x : h(x) ≥ 0}, where h is a continuously differentiable function defining the safety boundary. A CBF enforces this invariance by constraining the time derivative of h to satisfy ḣ(x,u) + α(h(x)) ≥ 0, where α is an extended class-K function. This constraint is encoded in a Quadratic Program (QP) that minimally modifies a nominal control input to ensure safety, enabling real-time safety filtering.

The paper develops the theory from first principles, starting with Lyapunov-based safety and progressing to CBFs, high-order CBFs (HOCBFs) for systems with relative degree > 1, exponential CBFs (ECBFs) for more aggressive safety enforcement, and the integration of CBFs with Control Lyapunov Functions (CLFs) for simultaneous safety and stability guarantees. The CBF-QP formulation is derived as: min_u ||u - u_nom||² subject to L_f h(x) + L_g h(x)·u + α(h(x)) ≥ 0, where L_f and L_g are Lie derivatives along the system dynamics f and control matrix g.

Applications span adaptive cruise control, bipedal robot walking (demonstrated on the MABEL robot, a precursor to Cassie), multi-robot collision avoidance, and robotic manipulation. The bipedal walking application is particularly detailed, showing how CBFs maintain walking stability by enforcing constraints on the robot's center of pressure and angular momentum.

## Core Contributions
- Rigorous mathematical formulation of safety as forward invariance of a safe set, with necessary and sufficient conditions via CBFs
- CBF-QP framework: real-time optimization that minimally modifies control inputs to guarantee safety, with closed-form solutions for single-constraint cases
- High-Order CBFs (HOCBFs) extending the framework to systems with relative degree > 1, critical for mechanical systems where safety constraints on position require controlling acceleration
- Exponential CBFs (ECBFs) that enforce exponential convergence to the safe set boundary, providing tunable safety margins
- Unified CBF-CLF-QP framework for simultaneous safety (CBF constraint) and stability (CLF constraint), with formal guarantees on both
- Application to bipedal robot walking with safety constraints on center of pressure, angular momentum, and step placement
- Compatibility analysis: conditions under which safety (CBF) and performance (CLF) constraints are simultaneously feasible

## Methodology Deep-Dive
The foundational formulation considers a control-affine system ẋ = f(x) + g(x)u, where x ∈ R^n is the state, u ∈ R^m is the control input, f and g are locally Lipschitz. The safe set is C = {x ∈ R^n : h(x) ≥ 0} for a continuously differentiable h : R^n → R. A function h is a CBF if there exists α ∈ K_ext such that sup_u [L_f h(x) + L_g h(x)·u + α(h(x))] ≥ 0 for all x ∈ C, where L_f h = ∂h/∂x · f(x) and L_g h = ∂h/∂x · g(x) are Lie derivatives. The key theorem states: if h is a CBF for C, then any Lipschitz controller u(x) satisfying L_f h(x) + L_g h(x)·u + α(h(x)) ≥ 0 renders C forward invariant.

For systems with relative degree r > 1 (e.g., position constraints in mechanical systems where ḧ depends on u but ḣ does not), HOCBFs introduce a sequence of barrier functions: ψ_0 = h, ψ_1 = ψ̇_0 + α_1(ψ_0), ..., ψ_r = ψ̇_{r-1} + α_r(ψ_{r-1}). The final constraint ψ_r ≥ 0 involves u and can be enforced via QP. Each α_i defines a convergence rate at its respective order. For bipedal robots, joint position limits have relative degree 2 (position → velocity → acceleration/torque), requiring second-order CBFs with two class-K functions α_1, α_2 to tune.

The CBF-QP for safety filtering is: min_{u ∈ R^m} (1/2)||u - u_ref||² subject to A_cbf · u ≥ b_cbf, where A_cbf = L_g h(x) and b_cbf = -L_f h(x) - α(h(x)). For multiple safety constraints {h_i}_{i=1}^k, each adds a row to the QP. The QP is convex (quadratic objective, linear constraints) and solvable in microseconds using active-set or interior-point methods, enabling real-time control at 1 kHz or higher.

The CBF-CLF-QP extends this to: min_{u,δ} (1/2)||u - u_ref||² + p·δ² subject to CBF constraints and CLF constraint L_f V + L_g V·u + γ·V ≤ δ, where V is a CLF, γ is a convergence rate, and δ is a relaxation variable (weighted by p) that allows CLF constraint violation when it conflicts with safety. This prioritizes safety over performance: the CBF constraint is hard (never violated), while the CLF constraint is soft (relaxed when needed).

For bipedal robots, the paper demonstrates safety constraints including: (1) Center of Pressure (CoP) within support polygon: h_cop = x_cop_max - |x_cop| ≥ 0, (2) Angular momentum bounds: h_L = L_max - |L| ≥ 0, (3) Step placement within feasible region: h_step = r_max - ||p_foot - p_target||² ≥ 0. These constraints are evaluated at the current state and enforced through the QP modifying joint torque commands. On the MABEL bipedal robot, this framework prevented falls during perturbation experiments while allowing the nominal walking controller to operate with minimal interference.

## Key Results & Numbers
- CBF-QP solving time: < 100 μs for single constraint, < 500 μs for 5+ constraints on standard hardware (suitable for 1 kHz control)
- MABEL bipedal robot: zero falls during 50+ perturbation tests with CBF-QP safety filter, vs 8/50 falls without filter
- Safety constraint satisfaction: 100% (by construction, given correct model), with constraint violations < 10^-6 numerically
- Control input modification: average ||u - u_ref|| < 5% of nominal torque, indicating minimal interference with performance controller
- Adaptive cruise control: maintains safe following distance within 2% of theoretical minimum at all speeds
- CBF-CLF-QP feasibility: infeasible in < 0.1% of timesteps across all bipedal walking experiments (resolved by CLF relaxation δ)
- Exponential CBF with convergence rate λ=5 provides safety margin recovery in ~0.2 seconds after perturbation

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
CBF-QP provides a principled safety layer for Mini Cheetah deployment. Safety constraints for a quadruped include: joint position/velocity limits, body orientation bounds (anti-flip), foot ground clearance during swing, and maximum ground reaction forces. A CBF-QP filter on top of the learned RL policy would ensure these constraints are never violated during real-world operation, even if the policy outputs unsafe actions.

For Project A, the relative degree 2 CBFs (HOCBFs) are needed since Mini Cheetah's safety constraints on joint positions require torque-level control. The computational cost is manageable—Mini Cheetah runs at ~1 kHz with 12 actuators, and the CBF-QP with 20-30 constraints solves well within the control loop. However, if the RL policy is well-trained with sufficient domain randomization, the safety filter may rarely activate, making it a deployment safeguard rather than a training component.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The CBF-QP framework is THE foundational theory for Project B's Safety level (LCBF - Learned Control Barrier Function). Understanding this paper is essential for implementing the LCBF component. Cassie's safety constraints include: angular momentum bounds for balance, center of pressure within support polygon, joint position/velocity/torque limits, minimum foot clearance during swing, and ground reaction force limits.

The HOCBF formulation is critical since Cassie's position-level safety constraints (e.g., knee angle limits, hip extension bounds) have relative degree 2 with respect to torque inputs. The CBF-CLF-QP framework enables simultaneous safety and tracking: the CLF ensures the RL policy's desired joint trajectories are tracked, while the CBF ensures safety constraints are maintained. The MABEL bipedal application in this paper provides direct precedent for Cassie—both are planar bipeds with similar dynamics. The LCBF in Project B extends this framework by learning the CBF function h(x) from data rather than hand-designing it, but the QP formulation and constraint enforcement remain identical.

## What to Borrow / Implement
- Implement the CBF-QP as the core safety filter for Project B's Safety level, with HOCBF for relative degree 2 position constraints on Cassie
- Define safety constraints for Cassie: angular momentum bounds, CoP limits, joint limits, and foot clearance using the h(x) ≥ 0 formulation
- Use the CBF-CLF-QP for combined safety and tracking at the Controller level, with CLF relaxation for feasibility
- Apply the MABEL bipedal walking safety constraints as a starting template for Cassie-specific constraints
- Implement real-time QP solving using OSQP or qpOASES for 1 kHz safety filtering on both platforms

## Limitations & Open Questions
- CBF formulation requires an accurate dynamic model f(x), g(x); model errors can lead to constraint violations despite theoretical guarantees
- Safe set design (choosing h(x)) requires domain expertise and may be overly conservative if not carefully crafted
- Multiple CBF constraints can conflict, requiring priority ordering or relaxation strategies not fully addressed in the basic framework
- Extension to learned/neural CBFs (as in Project B's LCBF) introduces approximation errors not covered by the classical theory
