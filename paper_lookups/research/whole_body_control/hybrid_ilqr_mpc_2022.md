# Hybrid iLQR Model Predictive Control for Contact Implicit Stabilization on Legged Robots

**Authors:** (2022)
**Year:** 2022 | **Venue:** IEEE IROS
**Links:** [arXiv](https://arxiv.org/abs/2207.04591)

---

## Abstract Summary
This paper extends the iterative Linear Quadratic Regulator (iLQR) to handle hybrid dynamical systems with mode-switching dynamics, operating within a model predictive control (MPC) framework for legged robot stabilization. Traditional iLQR assumes smooth dynamics, which breaks down at contact events where the system undergoes discontinuous state transitions. The authors introduce a hybrid iLQR formulation that explicitly accounts for mode boundaries, saltation matrices at contact transitions, and the possibility that the actual contact sequence may differ from the reference.

The key technical contribution is handling the mismatch between the reference contact schedule and the actual events during execution. In practice, perturbations cause the robot to make or break contact at different times than planned. The hybrid iLQR incorporates event-time sensitivity through saltation matrices—linear maps that propagate perturbations through discontinuities—enabling the backward pass to correctly account for timing variations. This makes the MPC robust to contact schedule deviations.

The approach is validated on legged robot stabilization tasks where external disturbances push the robot off its nominal trajectory, causing contact timing to shift. The hybrid iLQR MPC maintains stability by correctly propagating cost-to-go information through the shifted contact events, outperforming standard iLQR that ignores mode boundaries.

## Core Contributions
- Extension of iLQR to hybrid dynamical systems with explicit saltation matrix computation at mode boundaries
- MPC framework that handles mismatch between planned and actual contact sequences
- Event-time sensitivity analysis enabling the optimizer to adjust contact timing within the iLQR backward pass
- Robust stabilization under perturbations that cause contact schedule deviations
- Comparison with standard iLQR showing improved stability margins under disturbances
- Computational analysis demonstrating real-time feasibility for legged robot control

## Methodology Deep-Dive
The hybrid dynamical system is modeled as a collection of smooth vector fields {fₘ} indexed by mode m, with guard functions gₘ→ₘ' that trigger transitions between modes, and reset maps Δₘ→ₘ' that map pre-transition states to post-transition states. For a legged robot, modes correspond to different contact configurations (e.g., stance phase, flight phase), guards are contact/liftoff conditions (foot height = 0), and reset maps encode impact dynamics (velocity jumps).

The standard iLQR backward pass computes the value function gradient and Hessian by propagating through the linearized dynamics: δxₜ = Aₜδxₜ₋₁ + Bₜδuₜ. At a mode boundary occurring at time t*, the linearization must account for the saltation matrix S = Δ'(x*) + (f⁺ − Δ'f⁻)/(∂g/∂x · f⁻)ᵀ · (∂g/∂x), where f⁻ and f⁺ are the pre- and post-transition dynamics, Δ' is the reset map Jacobian, and g is the guard function. The saltation matrix captures how perturbations δx before the transition map to perturbations after, including the effect of shifted transition timing.

The hybrid iLQR backward pass inserts the saltation matrix at the mode boundary: Vₓ(t*⁻) = Sᵀ Vₓ(t*⁺) and Vₓₓ(t*⁻) = Sᵀ Vₓₓ(t*⁺) S + higher-order terms. This ensures that the quadratic approximation of the value function correctly reflects the cost of perturbations that shift the contact event. The forward pass then naturally produces trajectories where contact timing adapts to the current state.

In the MPC loop, the reference trajectory includes a nominal contact sequence with expected transition times. At each MPC iteration, the current state may indicate that contact events have occurred early or late relative to the reference. The hybrid iLQR handles this by: (1) detecting the current mode from sensor data, (2) aligning the reference mode sequence with the detected mode, (3) computing the hybrid backward pass with saltation matrices at the (potentially shifted) boundaries, and (4) executing the feedforward + feedback policy until the next MPC update.

The contact detection component uses a combination of ground reaction force thresholds and foot position estimates. When a mode mismatch is detected (e.g., the robot is still in swing when the reference says it should be in stance), the MPC re-aligns its plan by either extending the current mode or inserting an additional transition.

## Key Results & Numbers
- MPC update rate of 50–100 Hz on standard hardware, sufficient for legged robot control
- Stability maintained under perturbations causing up to 15% shift in contact timing relative to reference
- Standard iLQR (without saltation matrices) fails to stabilize when contact timing shifts by >5%
- 30–40% improvement in disturbance rejection compared to fixed-schedule MPC approaches
- Validated on planar biped and simplified quadruped models with 2–4 contact points
- Saltation matrix computation adds <0.5 ms overhead per mode boundary in the backward pass

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The hybrid iLQR MPC provides a principled model-based controller that could serve as a baseline or teacher for the Mini Cheetah's RL policy. The contact timing adaptation is valuable for robust locomotion on uneven terrain where foot contacts occur at unexpected times. The saltation matrix formulation could be integrated into differentiable simulation for improved gradient computation during PPO training. For sim-to-real, the ability to handle contact schedule mismatches directly addresses a common failure mode where the simulated contact timing differs from reality. However, iLQR requires a reasonably accurate dynamics model, which may limit its standalone applicability compared to model-free RL.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The mode-switching dynamics handling is critical for Cassie's bipedal locomotion, where gait transitions between stance, swing, and double-support phases define the locomotion behavior. The saltation matrix formulation can be incorporated into the Controller level to ensure stable tracking through contact transitions. For the Planner level, the hybrid iLQR provides a template for generating reference trajectories that explicitly account for Cassie's hybrid dynamics. The event-time sensitivity analysis is directly relevant to the Option-Critic framework, where options correspond to locomotion modes and transitions between options must be handled carefully. The robustness to contact schedule mismatches is essential for Cassie's real-world deployment where ground contact timing is uncertain.

## What to Borrow / Implement
- Implement saltation matrix computation for Cassie's contact transitions to improve gradient accuracy in the RL training loop
- Use the hybrid iLQR as a trajectory generator for the Planner level, providing dynamically consistent reference gaits
- Adopt the mode-mismatch handling strategy for robust MPC execution at the Controller level
- Integrate event-time sensitivity into the curriculum learning framework to gradually increase contact timing uncertainty
- Apply the guard function formulation to define option termination conditions in the Option-Critic architecture

## Limitations & Open Questions
- Validated primarily on simplified planar models; extension to full 3D dynamics with many contact points increases computational cost significantly
- Requires explicit enumeration of possible mode sequences, which grows combinatorially with the number of contact points
- Saltation matrix computation assumes instantaneous (rigid) contact; compliant contacts require modified formulations
- The approach assumes known terrain geometry for guard function evaluation; integrating with terrain perception is not addressed
