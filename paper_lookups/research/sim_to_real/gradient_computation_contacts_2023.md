# Improving Gradient Computation for Differentiable Physics Simulation with Contacts

**Authors:** Zhong, Y. D., Qin, J., Yang, Y., Dolan, J. M., Shi, G.
**Year:** 2023 | **Venue:** PMLR (Conference on Robot Learning / CoRL)
**Links:** [arXiv:2305.00092](https://arxiv.org/abs/2305.00092)

---

## Abstract Summary
This paper introduces the TOI-Velocity method, a technique for computing accurate gradients through rigid-body contact dynamics in differentiable physics simulations. The central problem addressed is that standard automatic differentiation through contact-handling code produces unreliable gradients due to the discontinuous nature of contact events—when a contact normal moves across a surface, the gradient with respect to the pre-contact state can be wildly inaccurate. The authors trace this inaccuracy to the time-stepping discretization: contacts are detected and resolved at discrete time steps, but the actual contact geometry changes continuously, creating a mismatch between the discrete gradient and the true continuous gradient.

The TOI-Velocity method resolves this by using continuous collision detection (CCD) to compute the exact time-of-impact (TOI) for each contact event, then computing gradients with respect to the pre-contact velocity at the TOI rather than at the discrete time step boundary. This produces gradient estimates that are consistent with the underlying continuous physics, eliminating the spurious gradient artifacts that plague naive differentiation. The method is demonstrated on trajectory optimization tasks involving multi-body rigid systems with frequent contact transitions.

Results show that TOI-Velocity gradients enable successful trajectory optimization in scenarios where standard differentiable simulation gradients fail completely—for example, optimizing a throwing trajectory where the object must bounce off a surface at a specific angle. The method is compatible with existing differentiable simulation frameworks and adds modest computational overhead.

## Core Contributions
- Identification and formal analysis of gradient inaccuracy sources in differentiable contact simulation: moving contact normals across discrete time steps
- TOI-Velocity method using continuous collision detection to compute physically consistent gradients through contact events
- Mathematical proof that TOI-Velocity gradients converge to the true continuous-time gradient as time step approaches zero
- Demonstration on trajectory optimization tasks where standard differentiable simulation gradients fail
- Compatibility with existing differentiable simulation frameworks (minimal code changes required)
- Analysis of computational overhead: ~10–30% increase in backward pass time
- Extension to multi-contact scenarios with simultaneous contact resolution

## Methodology Deep-Dive
The gradient inaccuracy problem arises from a fundamental mismatch in standard differentiable simulation. Consider a ball bouncing off a surface: at time step t, the ball is above the surface; at time step t+1, the contact is detected and resolved. The contact normal is computed based on the penetration geometry at t+1, which depends discontinuously on the ball's trajectory. If we perturb the ball's initial velocity slightly, the contact geometry at t+1 can change significantly (e.g., the ball contacts a different face of a polyhedron), causing a large, discontinuous jump in the gradient. This is not a numerical precision issue—it is a structural problem with the time-stepping approach.

The TOI-Velocity method addresses this in three steps. First, continuous collision detection (CCD) is performed to compute the exact time-of-impact τ ∈ [t, t+1] when contact first occurs. CCD uses conservative advancement or root-finding on the signed distance function to locate τ to high precision. Second, the pre-contact state is computed by advancing the simulation from time t to τ using the ballistic (contact-free) dynamics. Third, the contact resolution (impulse computation, velocity update) is applied at the TOI, and the post-contact state is advanced from τ to t+1. Gradients are computed by differentiating through this three-phase computation rather than the single-step contact resolution.

The key mathematical insight is that the contact normal at the TOI is a smooth function of the pre-contact velocity (small velocity changes shift the TOI slightly but don't change the contact geometry), whereas the contact normal at the discrete time step is a discontinuous function. By anchoring the gradient computation to the TOI, the method transforms a discontinuous derivative into a smooth one.

For multi-contact scenarios (e.g., a quadruped with four feet contacting the ground simultaneously), the method computes TOIs for all active contacts and processes them sequentially in chronological order. When contacts occur nearly simultaneously (within a tolerance), they are grouped and resolved together using a multi-contact impulse solver, with gradients flowing through the grouped resolution.

The implementation extends a standard differentiable physics engine (DiffTaichi or Brax-like) by adding a CCD module and modifying the backward pass to use the three-phase gradient computation. The forward pass is unchanged—the simulator still uses standard time-stepping for performance—but the backward pass recomputes the TOI to obtain accurate gradients. This "forward-fast, backward-accurate" design minimizes the computational overhead of the method.

The authors validate the gradient accuracy using finite-difference comparisons: TOI-Velocity gradients agree with finite-difference estimates to within 1% relative error, while standard differentiable simulation gradients can deviate by 100–1000% in contact-heavy scenarios. This is tested on systems ranging from a single bouncing ball to a 12-body articulated robot.

## Key Results & Numbers
- Gradient accuracy: <1% relative error vs. finite-difference ground truth (standard method: 100–1000% error in contact scenarios)
- Trajectory optimization success: TOI-Velocity succeeds in 95% of multi-contact optimization tasks vs. 30% for standard gradients
- Computational overhead: 15–25% increase in backward pass wall-clock time
- Tested on systems with up to 12 rigid bodies and 20+ simultaneous contacts
- CCD computation adds <5% overhead to the forward pass
- Bouncing ball task: TOI-Velocity optimization converges in 50 iterations vs. non-convergence for standard method
- Quadruped trajectory optimization: 40% lower final cost using TOI-Velocity gradients

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The gradient computation improvements are relevant if differentiable simulation is adopted for Mini Cheetah training (as proposed in Papers 1 and 2). Mini Cheetah locomotion involves four simultaneous ground contacts during trotting stance phases, with rapid contact transitions during swing phases—exactly the scenario where standard differentiable simulation gradients are unreliable. TOI-Velocity could serve as the gradient backend for AHAC or SHAC training of the Mini Cheetah policy.

However, if the Mini Cheetah pipeline stays with PPO (model-free), this paper's contributions are not directly applicable since PPO does not use simulation gradients. The relevance is conditional on adopting a differentiable simulation approach. The 15–25% backward pass overhead is acceptable for the gradient quality improvement, especially given that PPO alternatives claim 10× sample efficiency gains.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
Similar to Project A, the relevance depends on whether differentiable simulation is integrated into Cassie's training pipeline. Bipedal locomotion has particularly challenging contact dynamics due to the extended single-support phases where only one foot is grounded, and the rapid transitions during double-support and flight phases. The TOI-Velocity method's handling of moving contact normals is relevant because Cassie's foot geometry (small contact patch on a compliant ankle) makes contact normal computation sensitive to state perturbations.

For the Neural ODE Gait Phase module at the Controller level, accurate contact gradients could improve the learning of phase-dependent controllers that must synchronize with the contact schedule. The RSSM at the Planner level could also benefit from more accurate dynamics gradients if the learned dynamics model is trained on differentiable simulation rollouts. However, the higher levels of the hierarchy (Primitives, Planner) operate on abstract state spaces where contact gradients are less directly relevant.

## What to Borrow / Implement
- If adopting differentiable simulation for either project, use TOI-Velocity as the gradient computation backend to ensure reliable optimization through contacts
- Apply the CCD-based gradient correction to improve any trajectory optimization or shooting-based planning module that operates through contacts
- Use the gradient accuracy diagnostic (comparison with finite-difference) as a validation tool when implementing or debugging differentiable simulation pipelines
- Adopt the "forward-fast, backward-accurate" design pattern for computational efficiency in gradient-based training
- Extend the multi-contact handling to match the specific contact patterns of Mini Cheetah (4-foot trotting) and Cassie (1-2 foot bipedal)

## Limitations & Open Questions
- Computational overhead of CCD in the backward pass may compound over long training horizons, especially with many contacts
- The method assumes rigid-body contacts; soft or deformable contacts (as in rubber-footed robots) may require different treatment
- Multi-contact grouping heuristic (temporal tolerance for "simultaneous" contacts) is a tunable parameter that may need per-robot adjustment
- Integration with existing MuJoCo-based pipelines is non-trivial; MuJoCo's contact solver is not natively compatible with TOI-Velocity without significant modification
