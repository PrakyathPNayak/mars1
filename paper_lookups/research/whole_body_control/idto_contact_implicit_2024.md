# IDTO: Inverse Dynamics Trajectory Optimization for Contact-Rich Tasks

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [Project Page](https://idto.github.io/)

---

## Abstract Summary
IDTO presents a fast, open-source contact-implicit trajectory optimization framework built on inverse dynamics rather than the conventional forward dynamics formulation. By optimizing in the space of joint accelerations and contact forces (inverse dynamics) rather than joint torques and states (forward dynamics), the method exploits the sparsity and linearity of the inverse dynamics constraints to achieve order-of-magnitude speedups in trajectory optimization for contact-rich manipulation and locomotion tasks.

The key innovation is that the inverse dynamics formulation: τ = M(q)q̈ + C(q,q̇)q̇ + g(q) − Jᵀλ makes the relationship between decision variables (q̈, λ) and joint torques τ linear for a given configuration (q, q̇). This linearity, combined with the sparsity of the mass matrix and contact Jacobian, enables extremely efficient computation of the optimization's KKT system. No explicit contact scheduling is required—the complementarity constraints between contact distances and forces are handled through a relaxation scheme.

IDTO supports both offline trajectory planning (generating gaits from scratch) and real-time MPC (tracking and adapting trajectories online). The framework is validated on legged robot locomotion tasks, demonstrating autonomous gait generation where the optimizer discovers walking, trotting, and bounding gaits purely from the dynamics and a simple forward-velocity objective, without any gait-specific prior.

## Core Contributions
- Inverse dynamics formulation for contact-implicit trajectory optimization, exploiting linearity and sparsity for computational efficiency
- Open-source implementation enabling reproducibility and community adoption
- Autonomous gait discovery: walking, trotting, and bounding emerge from optimization without gait templates or contact scheduling
- Support for both offline planning and real-time MPC modes within a unified framework
- Order-of-magnitude speedup over forward-dynamics CI-TO approaches, enabling real-time deployment
- Contact complementarity handling via log-barrier relaxation with warm-started continuation
- Comprehensive benchmarking against existing CI-TO methods (forward dynamics, direct collocation)

## Methodology Deep-Dive
The trajectory optimization is formulated over a horizon of N timesteps with decision variables consisting of joint positions qₜ, joint velocities q̇ₜ, joint accelerations q̈ₜ, and contact forces λₜ at each timestep. The inverse dynamics constraint: τₜ = M(qₜ)q̈ₜ + c(qₜ, q̇ₜ) − J(qₜ)ᵀλₜ serves as an equality constraint that implicitly defines the joint torques. The integration constraints: qₜ₊₁ = qₜ + q̇ₜΔt + ½q̈ₜΔt² and q̇ₜ₊₁ = q̇ₜ + q̈ₜΔt enforce temporal consistency (using semi-implicit Euler or higher-order schemes).

The contact physics are encoded through complementarity constraints: φᵢ(qₜ) ≥ 0 (signed distance non-negative), λₙ,ᵢ ≥ 0 (normal force non-negative), φᵢ(qₜ) · λₙ,ᵢ = 0 (complementarity), and ‖λₜ,ᵢ‖ ≤ μ λₙ,ᵢ (friction cone), where φᵢ is the signed distance for contact point i, λₙ,ᵢ and λₜ,ᵢ are normal and tangential forces, and μ is the friction coefficient. The complementarity is relaxed via a log-barrier: −κ(log φᵢ + log λₙ,ᵢ), where κ is progressively reduced in a continuation scheme.

The critical computational advantage comes from the structure of the KKT system. In the inverse dynamics formulation, the Hessian of the Lagrangian has a block-tridiagonal structure (from temporal coupling) with block sizes proportional to the number of degrees of freedom. The mass matrix M(q) is sparse (banded for serial chains, tree-structured for branched robots), and the contact Jacobian J(q) is sparse (each contact point depends only on the kinematic chain from root to that point). The authors exploit this combined sparsity using a custom structured linear algebra solver that performs the KKT solve in O(N · n³) time, where N is the horizon length and n is the number of DoFs.

For autonomous gait discovery, the cost function consists of: (1) a forward velocity tracking term: ‖v_com,x − v_desired‖², (2) a torque regularization: ‖τ‖²_R, (3) a joint limit penalty, and (4) a posture regularization toward a default standing configuration. Remarkably, this simple cost function, combined with the contact-implicit dynamics, produces natural-looking gaits. The optimizer discovers that alternating contact is more efficient than dragging all feet, and the specific gait pattern (walk, trot, bound) depends on the desired velocity—matching the biological Froude number relationship.

The real-time MPC mode uses the trajectory optimizer in a receding-horizon fashion. At each MPC iteration, the previous solution is shifted forward in time and extended with a nominal terminal condition. The warm-start from the previous solution typically enables convergence in 1–3 Newton iterations (vs. 10–20 from cold start). The MPC handles disturbances by re-optimizing the trajectory: when a push displaces the robot from its planned trajectory, the optimizer naturally adjusts the contact schedule and forces to recover.

The open-source implementation is built on Drake and supports URDF/SDF robot descriptions. The contact model uses point contacts with Coulomb friction, and the signed distance computations use Drake's collision detection infrastructure. The solver is implemented in C++ with Python bindings for ease of use.

## Key Results & Numbers
- Offline trajectory optimization: full gait cycle (1 second, 50 timesteps) computed in 0.2–0.5 seconds for a quadruped (12 DoF)
- Real-time MPC: 5–15 ms per iteration with warm-starting, enabling 65–200 Hz MPC rates
- 10–30× speedup over forward-dynamics CI-TO (e.g., ContactImplicit.jl) on equivalent problems
- Autonomous gait discovery: walk at 0.3 m/s, trot at 0.8 m/s, bound at 1.5 m/s, matching biological Froude number predictions
- Torque cost reduction of 15–25% compared to hand-designed gait schedules, demonstrating efficiency of discovered gaits
- Disturbance recovery from 20–35% body weight lateral pushes in MPC mode
- Open-source release with support for quadruped, biped, and manipulation models

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
IDTO is directly applicable to the Mini Cheetah as both a trajectory generation tool and a real-time MPC controller. The autonomous gait discovery capability can generate diverse training trajectories for the PPO policy without manual gait specification—the optimizer finds natural gaits that the RL policy can then learn to replicate. The inverse dynamics formulation is particularly relevant because the Mini Cheetah's torque-controlled actuators can directly execute the optimized torque trajectories. For curriculum learning, IDTO can generate reference trajectories at increasing speeds and on varying terrains, providing a principled progression. The open-source Drake-based implementation facilitates integration with the MuJoCo training environment (via trajectory export). The 12-DoF Mini Cheetah is well within the computational budget for real-time MPC, enabling model-based backup control for safety-critical situations.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
IDTO is critically relevant to multiple levels of Cassie's hierarchical architecture. At the Planner level, IDTO's autonomous gait discovery can generate the reference walking gaits and gait transitions that the Planner must produce. The contact-implicit nature means no pre-specification of heel-strike/toe-off timing is needed—the optimizer discovers these naturally. At the Controller level, IDTO provides a whole-body trajectory optimizer that computes dynamically consistent joint trajectories and contact forces, directly executable as torque commands. The inverse dynamics formulation is especially valuable for Cassie because it naturally handles the underactuation (Cassie's passive spring joints) by treating the unactuated joint torques as zero-constrained. The real-time MPC mode (5–15 ms) is fast enough for the Controller level's update rate. For the Primitives level's DIAYN/DADS skill discovery, IDTO can generate diverse motion primitives by varying the cost function, providing physics-consistent skill demonstrations. The open-source release with biped support means Cassie-specific models can be directly implemented.

## What to Borrow / Implement
- Use IDTO for offline generation of diverse reference trajectories for both platforms, feeding the RL training pipeline with dynamically consistent demonstrations
- Deploy IDTO's real-time MPC mode as the whole-body controller at Cassie's Controller level, replacing hand-tuned PD controllers
- Leverage the autonomous gait discovery for curriculum learning: generate training references at increasing difficulty levels
- Adopt the inverse dynamics formulation's computational advantages for online trajectory re-planning within the safety filter
- Use IDTO's open-source implementation as the foundation for building a contact-implicit planning module compatible with both platforms

## Limitations & Open Questions
- Point contact model may be insufficient for Cassie's flat foot contacts, where pressure distribution matters for balance
- The log-barrier relaxation can cause numerical issues when contact distances approach zero, requiring careful parameter tuning
- Computational cost scales cubically with DoF, which may limit applicability to very high-DoF humanoid models
- The autonomous gait discovery produces locally optimal gaits that depend on initialization; global optimality of discovered gaits is not guaranteed
