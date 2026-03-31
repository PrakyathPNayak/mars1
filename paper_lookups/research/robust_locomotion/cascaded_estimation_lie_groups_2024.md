# Real-Time Cascaded State Estimation Framework on Lie Groups for Legged Robots

**Authors:** (2024)
**Year:** 2024 | **Venue:** MDPI Biomimetics
**Links:** [MDPI](https://www.mdpi.com/2313-7673/10/8/527)

---

## Abstract Summary
This paper proposes a cascaded state estimation framework for legged robots that chains two complementary filters in sequence: a Generalized Momentum-based Kalman Filter (GM-KF) followed by an Error-State Kalman Filter (ESKF), both formulated on Lie groups. The first filter stage estimates contact forces and detects foot-ground impacts using generalized momentum observers, while the second stage fuses these contact estimates with IMU and kinematic data to produce the full floating-base state estimate (position, velocity, orientation).

The cascade architecture enables tight coupling between contact force estimation and state estimation while maintaining computational efficiency. The GM-KF stage provides real-time contact force estimates that serve dual purposes: (1) detecting impact events that trigger mode switches in the state estimator, and (2) providing ground reaction force measurements that improve the ESKF's velocity estimation during stance phases. The Lie group formulation ensures geometric consistency throughout both filter stages, preventing the representation singularities and inconsistencies that afflict Euclidean formulations.

A key contribution is the handling of foot-ground impacts—the transient high-force events at heel strike and toe-off that cause large velocity discontinuities. The cascaded framework detects these impacts in the GM-KF stage and injects appropriate state resets in the ESKF stage, preventing the filter from diverging during the impact transients.

## Core Contributions
- Two-stage cascaded estimation: GM-KF for contact force/impact estimation followed by ESKF for full-state estimation
- Generalized momentum-based contact force observer that avoids acceleration measurements (which are noisy and require differentiation)
- Impact detection and handling via momentum discontinuity analysis, triggering state resets in the ESKF
- Lie group (SE₂(3)) formulation for both filter stages, ensuring geometric consistency across the cascade
- Real-time tightly coupled estimation handling foot-ground impacts without requiring hardware force sensors
- Validation on real legged robot data with multiple gaits including dynamic impacts
- Analysis showing improved estimation during impact transients compared to single-filter approaches

## Methodology Deep-Dive
The Generalized Momentum (GM) observer estimates contact forces without requiring joint acceleration measurements. The generalized momentum is defined as p = M(q)q̇, where M is the mass matrix. Its time derivative satisfies: ṗ = τ + Jᵀλ − C(q,q̇)q̇ − g(q) = τ + Jᵀλ − n(q,q̇), where τ are joint torques, λ are contact forces, J is the contact Jacobian, and n(q,q̇) captures Coriolis and gravity terms. The observer estimates the contact wrench as: λ̂ = (Jᵀ)† [p(t) − p(0) − ∫₀ᵗ (τ + Jᵀλ̂ − n) ds + Kᵢ ∫₀ᵗ (p − p̂) ds], where Kᵢ is the observer gain. This is formulated as a Kalman filter (GM-KF) by treating the contact forces as the state and the momentum equation as the process model: λₖ₊₁ = λₖ + wλ (random walk), with observation: p_measured − p_predicted = Jᵀ λₖ Δt + vp.

Impact detection is performed by monitoring the generalized momentum residual r(t) = p(t) − p̂(t). During smooth contact, ‖r(t)‖ remains small. At impact, the momentum undergoes a discontinuous jump Δp = Jᵀ F_impact · Δt_impact, causing a spike in the residual. When ‖r(t)‖ exceeds a learned threshold (calibrated per gait), an impact event is declared. The impact time and magnitude are estimated from the residual peak characteristics.

The ESKF second stage operates on the SE₂(3) Lie group. The error state ξ = [δθ; δv; δp; δbₐ; δbω] ∈ ℝ¹⁵ includes orientation, velocity, and position errors plus accelerometer and gyroscope biases. The nominal state propagation uses IMU measurements: R̂ₖ₊₁ = R̂ₖ Exp((ωₘ − b̂ω)Δt), v̂ₖ₊₁ = v̂ₖ + (R̂ₖ(aₘ − b̂ₐ) + g)Δt, p̂ₖ₊₁ = p̂ₖ + v̂ₖΔt + ½(R̂ₖ(aₘ − b̂ₐ) + g)Δt². The error-state dynamics linearize around the nominal: ξₖ₊₁ = Fₖξₖ + Gₖwₖ, where Fₖ and Gₖ depend only on the IMU measurements and the nominal state.

The ESKF measurement update uses two sources from the GM-KF: (1) the zero-velocity constraint for feet detected in contact (λ̂ > threshold), providing: v̂_foot = J(q)q̇ + ω × r_foot = 0, which constrains the body velocity; (2) the estimated ground reaction forces λ̂ themselves, which provide an independent velocity measurement through Newton's second law: Mv̇ = Σλ̂ᵢ + Mg. The contact force measurements are particularly valuable during double-support phases where both feet are on the ground and the force distribution provides additional information.

When an impact is detected by the GM-KF, the ESKF performs an impact reset: the velocity estimate is corrected based on the estimated impact impulse, and the covariance is inflated to account for the modeling uncertainty during the impact transient. This prevents the ESKF from trusting its pre-impact prediction too much, allowing the post-impact measurements to quickly correct the state.

The Lie group formulation ensures that the rotation error δθ is always small (a property of the error-state formulation on SO(3)), even during large rotations. This contrasts with Euler angle representations where gimbal lock can cause singularities, or quaternion representations where the double-cover issue requires careful handling. The Exp and Log maps between the Lie group and Lie algebra are computed using the Rodrigues formula for SO(3) and the BCH formula for SE₂(3).

## Key Results & Numbers
- Impact detection latency of <2 ms from impact event to detection, enabling prompt state reset
- Position RMSE of 2.8 cm over 30 m traversal, 30% improvement over single ESKF without impact handling
- Velocity estimation RMSE of 0.03 m/s during steady locomotion, degrading to 0.08 m/s during impact transients (vs. 0.15 m/s for single ESKF)
- Contact force estimation RMSE of 5.2 N (out of typical 50–100 N ground reaction forces), sufficient for contact detection
- Total cascaded filter computation time of 0.35 ms per cycle (0.15 ms GM-KF + 0.20 ms ESKF)
- Robustness validated across walking (0.3 m/s), trotting (1.0 m/s), and bounding (1.5 m/s) gaits
- Impact reset reduces post-impact convergence time from 50 ms (without reset) to 8 ms

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The cascaded estimation framework addresses a critical challenge for the Mini Cheetah: accurate state estimation during dynamic gaits with frequent and forceful foot-ground impacts. The GM-based contact force observer eliminates the need for unreliable hardware force sensors on the Mini Cheetah's feet. The impact detection and state reset mechanism is essential for aggressive gaits (bounding, galloping) that the RL policy may learn during training. The cascaded architecture provides both contact force estimates (for the RL observation space) and floating-base state estimates (for the control loop) from a single unified framework. The 0.35 ms computation time leaves substantial margin within the 1 kHz control loop. For domain randomization, the GM observer gains and impact detection thresholds can be varied to simulate estimation uncertainty.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The cascaded framework is directly applicable to Cassie's hierarchical architecture. The GM-KF first stage provides contact force estimates and impact detection that feed the Planner level's gait phase estimation and the CBF-QP safety filter's stability constraints. The ESKF second stage provides the floating-base state for the Controller level's whole-body tracking. The impact handling is crucial for Cassie's bipedal walking, where heel-strike impacts are more severe than in quadruped gaits due to the higher per-foot loading. The Lie group formulation (SE₂(3)) is consistent with the geometric framework used throughout Cassie's estimation pipeline. The contact force estimates from the GM-KF can be directly used to compute the Zero Moment Point (ZMP) and Center of Pressure (CoP), which are fundamental to bipedal balance control and inform the Capture Point computation. The cascaded information flow (forces → state) mirrors the hierarchical architecture's top-down planning and bottom-up estimation structure.

## What to Borrow / Implement
- Implement the two-stage cascaded architecture for both platforms: GM-KF for contact estimation followed by ESKF for state estimation
- Use the generalized momentum observer for sensorless contact force estimation, eliminating hardware force sensor dependencies
- Adopt the impact detection and state reset mechanism for handling heel-strike events in Cassie's walking gait
- Integrate the estimated ground reaction forces into the CBF-QP safety filter for real-time stability margin computation
- Apply the Lie group (SE₂(3)) formulation consistently across both filter stages and the RL observation processing

## Limitations & Open Questions
- The generalized momentum observer requires accurate knowledge of the mass matrix M(q) and Coriolis terms, which are sensitive to modeling errors
- Impact modeling assumes instantaneous rigid impacts; Cassie's leaf-spring mechanisms produce compliant impacts with finite duration
- The cascaded architecture introduces a one-timestep delay between contact force estimation and state update, which may affect very high-frequency controllers
- Extension to multi-contact scenarios (e.g., hand contacts for humanoids) significantly increases the GM observer complexity
