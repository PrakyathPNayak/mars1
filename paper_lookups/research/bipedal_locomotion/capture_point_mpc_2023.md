# A Model Predictive Capture Point Control Framework for Robust Humanoid Balancing and Walking

**Authors:** Lim et al.
**Year:** 2023 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2307.13243)

---

## Abstract Summary
This paper presents a Model Predictive Control (MPC) framework centered on Capture Point (CP) tracking for robust humanoid balancing and walking. The Capture Point, derived from the Linear Inverted Pendulum Model (LIPM), represents the ground position where the robot must place its foot to come to a complete stop. By formulating foot placement and step timing as an MPC optimization over capture point trajectories, the framework achieves real-time push recovery and robust walking on humanoid robots. The key advantage over traditional ZMP-based methods is that CP-MPC explicitly optimizes for balance recovery rather than maintaining a static stability margin.

The framework operates in two modes: a steady-state walking mode where the MPC tracks a reference CP trajectory for periodic gait, and a recovery mode where the MPC reactively adjusts foot placement and step timing to reject disturbances. The transition between modes is seamless, governed by the deviation of the actual CP from the reference trajectory. When the deviation exceeds a threshold, the controller prioritizes balance recovery over trajectory tracking, temporarily adjusting step length, step width, and step timing.

Experiments on a full-size humanoid robot demonstrate robust push recovery from forces up to 150N applied for 0.3 seconds, as well as stable walking on flat and mildly uneven surfaces. The MPC formulation is computationally efficient, solving in under 2ms per control cycle, enabling 500Hz control rates suitable for real-time deployment.

## Core Contributions
- **Capture Point MPC formulation** that unifies steady-state walking and push recovery in a single optimization framework, with smooth transitions between modes
- **Joint step placement and step timing optimization** — unlike prior CP controllers that fix step timing and only adjust placement, this framework co-optimizes both for maximum recovery capability
- **LIPM-based prediction model** with centroidal momentum compensation that accounts for full-body dynamics effects not captured by the simple pendulum model
- **Real-time feasibility** with solve times under 2ms, enabling deployment at 500Hz on standard humanoid computing hardware
- **Robust push recovery** demonstrated with forces up to 150N × 0.3s on a full-size humanoid, exceeding typical disturbance rejection capabilities
- **Analytical gradient computation** for the CP-MPC problem, enabling warm-starting and fast convergence within 3-5 QP iterations

## Methodology Deep-Dive
The Capture Point is formally defined as the point on the ground where the robot's Center of Mass (CoM) would converge if the robot placed its foot there and applied zero control (under the LIPM dynamics). Mathematically, the CP is computed as: ξ = x + ẋ/ω₀, where x is the CoM horizontal position, ẋ is the CoM velocity, and ω₀ = √(g/z₀) is the natural frequency of the inverted pendulum with height z₀ and gravitational acceleration g. The CP completely characterizes the balance state under LIPM assumptions: if the CP lies within the support polygon, the robot can recover balance; if it lies outside, the robot must step.

The MPC formulation optimizes over a prediction horizon of N steps (typically N=3-5, covering 1-2 seconds of future motion). The decision variables are: foot placement positions (x_f, y_f) for each future step, step timing (T_step) for each step, and desired CP trajectory parameters. The cost function balances: (1) CP trajectory tracking — deviation of predicted CP from reference periodic gait, (2) foot placement regularity — deviation from nominal step length/width, (3) step timing regularity — deviation from nominal step duration, and (4) control effort — acceleration of the CoM. The optimization is subject to kinematic constraints (maximum step length/width based on leg reachability), timing constraints (minimum/maximum step duration), and friction constraints (foot placement must be within the friction cone).

The LIPM prediction model is extended with centroidal momentum compensation to account for angular momentum effects from arm swing and trunk rotation. The centroidal angular momentum is estimated from the full-body state and injected as an additive disturbance in the LIPM prediction. This significantly improves prediction accuracy during dynamic motions and push recovery where the assumption of zero angular momentum is violated.

The optimization is cast as a Quadratic Program (QP) and solved using an active-set solver. The analytical gradient of the CP dynamics with respect to foot placement and step timing enables efficient warm-starting from the previous solution. The solver typically converges in 3-5 iterations, taking less than 2ms total. For the recovery mode, the QP constraints are relaxed to allow larger steps and faster stepping than the nominal gait, trading kinematic comfort for balance recovery.

A whole-body controller translates the MPC foot placement and CoM trajectory commands into joint-level torques using inverse kinematics and task-space control. The priority hierarchy assigns highest priority to balance maintenance (CoM height and CP tracking), followed by foot placement tracking, and lowest priority to posture regulation.

## Key Results & Numbers
- **Push recovery**: withstands forces up to **150N for 0.3s** (45 N·s impulse) from front, side, and rear directions
- **Recovery step latency**: first recovery step initiated within **120ms** of disturbance detection
- **MPC solve time**: **<2ms** per control cycle, enabling 500Hz control rate on Intel i7 onboard computer
- **Walking speed**: stable walking at up to **0.6 m/s** on flat ground with 20cm step length
- **Step timing adaptation**: step duration varies from **0.3s to 0.8s** (nominal 0.5s) during recovery, demonstrating timing optimization benefit
- **CP tracking error**: RMS error of **1.2cm** during steady-state walking, rising to **4.5cm** during peak push recovery
- **Comparison**: CP-MPC recovers from **35% larger pushes** than fixed-timing CP controller and **60% larger** than ZMP-based controller
- **Prediction horizon analysis**: N=3 steps sufficient for most disturbances; N=5 required for near-limit pushes

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
The Capture Point concept is designed for bipedal balance where the support polygon is small and foot placement is critical. For the Mini Cheetah quadruped with its wide support polygon and four contact points, the Capture Point is less directly relevant. However, the general concepts of MPC-based balance control and disturbance rejection could inform the Mini Cheetah's push recovery capability. The real-time QP formulation and analytical gradient computation are broadly applicable optimization techniques.

The concept of co-optimizing step placement and timing could be adapted to the Mini Cheetah's gait planning, where foot placement timing affects stability during transitions between gaits or on challenging terrain.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critical to Project B as it directly informs Cassie's Differentiable Capture Point module. The CP-MPC formulation provides the mathematical foundation for Cassie's foot placement optimization. The key insight — co-optimizing step placement and timing — should be incorporated into Cassie's capture point computation. The LIPM-based prediction with centroidal momentum compensation is directly applicable to Cassie's reduced-order dynamics model.

The differentiable aspect of Cassie's capture point module can build on this paper's analytical gradient computation for CP dynamics. Making the CP-MPC differentiable would enable end-to-end learning through the capture point computation, allowing the RL policy to backpropagate through the balance controller. The recovery mode with relaxed constraints maps to Cassie's Safety level intervention logic: when balance is threatened, the safety controller overrides normal locomotion to execute recovery steps. The 2ms solve time validates that QP-based capture point control is feasible at the control rates needed for Cassie's real-time operation.

## What to Borrow / Implement
- **Differentiable CP-MPC formulation** — adapt the analytical gradient computation to create a differentiable capture point module for Cassie's architecture, enabling backpropagation through balance control
- **Joint step placement + timing optimization** — implement co-optimization of foot position and step duration in Cassie's Controller level
- **Centroidal momentum compensation** — extend the basic LIPM capture point with angular momentum correction for more accurate balance prediction during dynamic maneuvers
- **Recovery mode trigger and constraint relaxation** — implement the CP deviation threshold for triggering Cassie's Safety level, with relaxed kinematic constraints during recovery
- **QP warm-starting strategy** — use the previous solution as initialization for the next control cycle's QP to maintain real-time feasibility

## Limitations & Open Questions
- **LIPM simplification** — the Linear Inverted Pendulum Model neglects many dynamics effects (foot roll, ankle torque limits, leg mass) that are significant for Cassie's specific morphology
- **Flat ground assumption** — the CP formulation assumes a flat walking surface; extension to uneven terrain requires modifying the pendulum height dynamically
- **No learning component** — the framework is purely model-based; integrating RL for parameter tuning or policy learning could improve adaptiveness
- **Humanoid-specific tuning** — the constraint values (step length, timing bounds) are tuned for a specific humanoid; significant re-tuning needed for Cassie's different leg kinematics
