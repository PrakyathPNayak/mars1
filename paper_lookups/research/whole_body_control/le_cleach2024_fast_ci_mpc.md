# Fast Contact-Implicit Model Predictive Control

**Authors:** Le Cleac'h, S., Howell, T., Pan, Z., Manchester, Z., Schwager, M., Posa, M. (Stanford University)
**Year:** 2024 | **Venue:** IEEE Transactions on Robotics (TRO) / arXiv
**Links:** [PDF](https://msl.stanford.edu/papers/le_cleach_fast_2024.pdf)

---

## Abstract Summary
This paper presents a structure-exploiting interior-point solver specifically designed for real-time contact-implicit model predictive control (CI-MPC) on legged robots. Unlike traditional approaches that require pre-scheduled contact modes and timing, this method discovers contact sequences online as part of the optimization, enabling highly reactive locomotion behaviors.

The key insight is that contact-implicit formulations naturally encode complementarity constraints between contact forces and distances, and the authors exploit the sparsity structure of these constraints within a custom interior-point method. The solver achieves computation times compatible with real-time MPC loops (on the order of milliseconds), which was previously considered intractable for contact-implicit planning. The approach is validated on quadruped walking tasks with dynamic disturbance recovery, demonstrating that the robot can autonomously adjust its contact schedule in response to unexpected perturbations.

The work bridges the gap between the expressiveness of contact-implicit trajectory optimization and the real-time requirements of model predictive control, making it practical for deployment on hardware platforms like the Mini Cheetah.

## Core Contributions
- Custom interior-point solver that exploits the block-sparse structure of contact-implicit dynamics, achieving order-of-magnitude speedups over general-purpose solvers
- Real-time CI-MPC framework that discovers contact timing and sequences online without pre-scheduled contact modes
- Smooth relaxation of complementarity constraints enabling gradient-based optimization through contact events
- Demonstration of dynamic disturbance recovery on quadruped platforms where the robot autonomously re-plans contact sequences
- Warm-starting strategy that leverages temporal structure of MPC to reduce solve times across successive iterations
- Comprehensive benchmarking against existing contact-scheduling approaches showing superior reactivity and robustness

## Methodology Deep-Dive
The foundation of this work is the contact-implicit formulation where the dynamics are expressed as a complementarity problem. Given a robot with generalized coordinates q and velocities v, the discrete-time dynamics incorporate contact forces λ through: M(q)(v⁺ − v) = h(q,v)Δt + J(q)ᵀλ, subject to the complementarity condition 0 ≤ φ(q) ⊥ λ ≥ 0, where φ represents signed distances and J is the contact Jacobian. The friction cone constraints are handled via a polyhedral approximation.

The interior-point solver replaces the complementarity constraints with a barrier formulation: min Σ cost(xₜ,uₜ) − κ Σ (log φᵢ + log λᵢ), where κ is a barrier parameter that is progressively reduced. The authors identify that the KKT system of this barrier problem has a specific block-banded structure arising from the temporal coupling in MPC and the local nature of contact constraints. They design a custom linear algebra backend that solves the KKT system in O(N·(n+m)³) time, where N is the horizon length, n is the state dimension, and m is the number of contact points.

A critical algorithmic detail is the use of a smooth contact model during the interior-point iterations. Rather than handling the non-smooth complementarity directly, the barrier relaxation provides a smooth approximation that becomes exact as κ→0. This enables standard Newton steps with reliable convergence. The authors also implement a filter line-search to ensure global convergence properties.

The warm-starting strategy shifts the previous MPC solution forward in time and extrapolates the final step. For contact-implicit problems, this is non-trivial because the contact mode may change between MPC iterations. The solver handles this gracefully because it does not commit to a fixed contact schedule—the warm start provides a good initial guess, and the solver naturally adjusts the contact forces and timing.

Disturbance recovery experiments demonstrate pushes of up to 40% body weight applied to the quadruped's torso. The CI-MPC autonomously adjusts stance timing, takes corrective steps, and modulates ground reaction forces—all without any external contact schedule planner.

## Key Results & Numbers
- Solve times of 2–8 ms per MPC iteration on a standard CPU, enabling 100+ Hz control rates
- 10–50× speedup over general-purpose interior-point solvers (e.g., IPOPT) on equivalent contact-implicit problems
- Successful disturbance recovery from lateral pushes up to 40% body weight on quadruped simulation
- Horizon lengths of 20–40 steps (0.4–0.8 seconds look-ahead) sufficient for reactive locomotion
- Contact sequence discovery matches or outperforms hand-designed gait schedulers in dynamic scenarios
- Convergence typically in 5–15 interior-point iterations with warm-starting

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This work is directly relevant to the Mini Cheetah platform as it demonstrates real-time contact-implicit MPC on quadrupeds. The structure-exploiting solver could serve as a model-based planning component that complements the RL policy—either as a trajectory reference generator for the PPO policy to track, or as a safety-critical backup controller. The disturbance recovery capabilities are particularly valuable for sim-to-real transfer, where unexpected perturbations are common. The contact-implicit formulation eliminates the need for a separate gait scheduler, which simplifies the overall control architecture. Integration with domain randomization could involve randomizing the contact model parameters (friction, restitution) within the MPC formulation.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The contact-implicit MPC approach is critically relevant to Cassie's hierarchical architecture at multiple levels. At the Planner level, CI-MPC can serve as a physics-aware trajectory generator that accounts for Cassie's underactuated dynamics and contact constraints. At the Controller level, the real-time solve times make it feasible as the low-level whole-body controller, directly computing joint torques that respect contact physics. The automatic contact discovery is especially valuable for Cassie's bipedal locomotion where heel-strike and toe-off timing significantly affects stability. The Capture Point computation in the safety layer could be informed by the CI-MPC's predicted contact forces and timing. The complementarity-based contact model provides a principled foundation for the CBF-QP safety filter.

## What to Borrow / Implement
- Adopt the contact-implicit formulation as the physics model within the MPC-based controller layer for both platforms
- Use the warm-starting strategy for real-time performance when integrating MPC with the RL training loop
- Implement the smooth complementarity relaxation technique for differentiable contact simulation in the training environment
- Leverage the disturbance recovery framework to generate recovery demonstrations for imitation learning components
- Integrate the structure-exploiting solver as a real-time safety verification layer that checks RL policy commands against contact-feasibility

## Limitations & Open Questions
- Computational cost may still be prohibitive for very long planning horizons (>1 second) needed for complex terrain
- Polyhedral friction cone approximation introduces artifacts at high tangential velocities
- Limited to rigid-body contact models; soft terrain or deformable surfaces are not addressed
- The approach assumes known contact geometry—handling perception uncertainty in contact locations remains open
