# Improving Gradient Computation for Differentiable Physics Simulation with Contacts

**Authors:** Zhong et al.
**Year:** 2023 | **Venue:** arXiv
**Links:** [arXiv:2305.00092](https://arxiv.org/abs/2305.00092)

---

## Abstract Summary
Zhong et al. address a fundamental problem in differentiable physics simulation: gradient computation through contact events produces inaccurate or uninformative gradients when contact normals are not fixed. Standard approaches to differentiable contact assume that contact geometry (normals, tangent planes, contact points) remains constant during gradient computation, which is valid for simple geometries (sphere on plane) but breaks down for complex shapes where the contact configuration changes with the optimization variables.

The paper introduces the Time-of-Impact (TOI) Velocity method, a principled approach to computing gradients that correctly accounts for the time at which contact occurs and the velocity state at that instant. By analytically differentiating through the time-of-impact computation, the method provides gradients that reflect how changes in pre-contact state affect post-contact dynamics, including the effect on contact normal direction and contact timing. This yields significantly more accurate gradients for optimal control problems involving contact-rich dynamics.

The TOI Velocity method is validated on a range of contact-rich benchmarks, from simple billiard-type problems to multi-body locomotion scenarios. In all cases, it produces gradients that are closer to finite-difference ground truth than existing methods, and these improved gradients translate to faster convergence and better solutions in downstream optimization tasks. The method is particularly impactful for locomotion, where foot-ground contacts involve complex geometry and varying contact configurations.

## Core Contributions
- Identifies the fixed contact normal assumption as a key source of gradient inaccuracy in differentiable simulation
- Proposes the Time-of-Impact (TOI) Velocity method for accurate gradient computation through contact events
- Provides analytical derivatives through the time-of-impact computation itself, capturing contact timing sensitivity
- Demonstrates 5–50× gradient accuracy improvement over standard methods on contact-rich benchmarks
- Shows that improved gradients lead to faster convergence in optimal control and policy optimization
- Applicable to rigid-body contacts with arbitrary geometry (not limited to convex shapes)
- Integrates with existing differentiable simulation frameworks via a modular contact gradient module

## Methodology Deep-Dive
The core insight is that when two bodies approach contact, three quantities change simultaneously: (1) the contact time t_c (when contact first occurs), (2) the contact point p_c (where on the surface contact occurs), and (3) the contact normal n_c (the direction of the contact impulse). Standard differentiable simulators treat t_c, p_c, and n_c as constants during backpropagation, computing ∂s_{post}/∂s_{pre} with these values fixed. This is only valid when the contact geometry is trivial (e.g., a sphere falling on an infinite plane, where n = [0,0,1] regardless of state). For general geometries—a foot contacting uneven ground, a robot palm grasping a curved object—this assumption introduces systematic gradient bias.

The TOI Velocity method computes the exact derivative dt_c/ds_{pre} (how contact time changes with pre-contact state) and dn_c/ds_{pre} (how the contact normal changes). For a pair of bodies with positions q_A, q_B and velocities v_A, v_B, the time of impact satisfies φ(q_A + v_A·t_c, q_B + v_B·t_c) = 0, where φ is the signed distance function. Differentiating implicitly: dt_c/dv_A = -∂φ/∂q_A · t_c / (∂φ/∂q_A · v_A + ∂φ/∂q_B · v_B), and similarly for other state variables. The contact normal derivative dn_c/ds_{pre} follows from differentiating n_c = ∇φ/||∇φ|| at the contact configuration.

The post-contact velocity is computed from the Newton-Coulomb impact model: v_{post} = v_{pre} - (1+e)·(v_n)·n_c - μ·|v_n|·(v_t/||v_t||), where e is the coefficient of restitution, μ is friction, and v_n, v_t are normal and tangential velocity components. The full gradient ∂v_{post}/∂s_{pre} now includes terms from ∂n_c/∂s_{pre} and ∂t_c/∂s_{pre}, which are zero in standard methods but non-negligible in practice.

Implementation uses a custom backward pass injected into the differentiable simulator's contact resolution step. The forward simulation proceeds normally (with standard contact detection and resolution), but during backpropagation, the custom backward pass replaces the standard contact gradient with the TOI Velocity gradient. This modular design allows integration with existing frameworks: the authors demonstrate integration with Brax, Warp, and a custom JAX-based simulator.

For locomotion applications, the method handles the multi-contact case (multiple feet simultaneously in contact) by treating each contact independently and summing gradients. The per-contact TOI computation requires solving a root-finding problem (binary search or Newton's method on φ), which adds ~10–20% computational overhead per contact compared to standard gradient computation.

## Key Results & Numbers
- Gradient accuracy (cosine similarity to finite-difference): TOI Velocity 0.95–0.99 vs. standard methods 0.3–0.8
- Gradient accuracy improvement: 5–50× depending on contact geometry complexity
- Convergence speedup in trajectory optimization: 2–5× fewer iterations to reach target cost
- Computational overhead: 10–20% per contact event relative to standard differentiable simulation
- Locomotion benchmark: quadruped trajectory optimization converges in 80 iterations (TOI) vs. 350 (standard)
- Billiard benchmark: TOI achieves target configuration in 15 optimization steps vs. >100 for standard
- Method correctly handles edge contacts, vertex contacts, and sliding-to-rolling transitions
- Applicable to meshes with up to ~1000 vertices per collision body without significant slowdown

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The TOI Velocity method improves gradient quality for Mini Cheetah's MuJoCo simulation, particularly during foot-ground contact events where contact normals shift as the foot rolls or slides. While the current PPO-based pipeline does not use simulation gradients directly, adopting differentiable simulation (per Papers 2 and 3 in this collection) would benefit substantially from TOI Velocity gradients. The 2–5× convergence speedup in trajectory optimization could accelerate gait optimization experiments.

The method is most impactful for scenarios involving complex contact geometry: the Mini Cheetah walking on rough terrain with varying surface normals, or performing dynamic maneuvers where foot contact configurations change rapidly. For flat-terrain walking, the improvement over standard methods is more modest.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Accurate contact gradients are essential for Cassie's differentiable capture point (DCP) computation. The DCP module needs to backpropagate through foot-ground contact dynamics to compute balance-maintaining actions, and the quality of these gradients directly determines the DCP's effectiveness. Cassie's foot contacts involve complex geometry (curved sole on uneven ground), making the fixed contact normal assumption particularly problematic.

The TOI Velocity method provides the gradient accuracy needed for the DCP to produce reliable balance signals. The 10–20% computational overhead per contact is acceptable given the DCP operates at a lower frequency (10–50 Hz) than the low-level Controller (200+ Hz). The modular integration design means the TOI gradient module can be added to Cassie's differentiable simulation backend without restructuring the existing pipeline.

## What to Borrow / Implement
- Integrate the TOI Velocity gradient module into the differentiable simulation backend for both platforms
- Use TOI gradients specifically for the Cassie DCP module's contact-through-backpropagation
- Adopt the gradient cosine similarity metric for validating contact gradient quality during development
- Apply TOI Velocity to trajectory optimization for Mini Cheetah gait generation as a differentiable-sim experiment
- Use the modular backward-pass injection pattern for clean integration with Brax or MJX

## Limitations & Open Questions
- Computational overhead (10–20% per contact) accumulates with many simultaneous contacts; quadruped with 4 feet adds ~50–80% total overhead
- Root-finding for time-of-impact can fail to converge for degenerate contact configurations (parallel surfaces, grazing contacts)
- Method assumes rigid-body contacts; deformable terrain (soft ground, sand) requires extension to deformable contact models
- Limited validation on real-world tasks; gradient accuracy improvements may not always translate to better final policy performance
