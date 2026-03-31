# Neural-Augmented Kalman Filtering for Legged Robot State Estimation

**Authors:** Seokju Lee et al.
**Year:** 2023 | **Venue:** arXiv
**Links:** [Project Page](https://seokju-lee.github.io/projects/state_estimator/)

---

## Abstract Summary
This paper introduces the Invariant Neural-Augmented Kalman Filter (InNKF), a hybrid framework that couples the Right-Invariant Extended Kalman Filter (RI-EKF) with deep neural networks to achieve superior state estimation for legged robots. The key idea is to preserve the geometric consistency guarantees of the invariant Kalman filter on Lie groups while using neural networks to learn and correct the residual errors that the model-based filter cannot capture—such as unmodeled soft contact dynamics, kinematic calibration errors, and sensor biases.

The neural component operates as a learned correction layer that estimates the remaining bias after the standard Kalman update. Rather than replacing the filter with an end-to-end neural network (which would lose the geometric guarantees), or using the filter alone (which suffers from modeling errors), InNKF combines both: the invariant filter provides a geometrically consistent baseline estimate, and the neural network refines it by predicting and subtracting the systematic error patterns that the filter consistently makes.

This approach leverages the geometric structure of robot state spaces as Lie groups (SE₂(3) for position, velocity, orientation), ensuring that the neural corrections respect the manifold structure. The result is a state estimator that is more accurate than either the filter or the neural network alone, while maintaining the interpretability and convergence properties of the invariant filter.

## Core Contributions
- Invariant Neural-Augmented Kalman Filter (InNKF) architecture that preserves Lie group geometric consistency while incorporating learned corrections
- Neural bias estimation module that predicts and corrects systematic errors in the Kalman filter output
- Geometric-aware training procedure that computes losses on the Lie algebra rather than Euclidean space
- Demonstration that the hybrid approach outperforms both pure model-based (RI-EKF) and pure learning (end-to-end NN) approaches
- Analysis of what the neural component learns: primarily contact model errors, kinematic calibration offsets, and IMU bias dynamics
- Validation on real legged robot data across multiple gaits and terrain types
- Ablation study showing contribution of invariant structure vs. neural augmentation

## Methodology Deep-Dive
The InNKF operates in three phases per timestep: invariant prediction, invariant update, and neural correction. The prediction step propagates the state on SE₂(3) using the standard RI-EKF equations driven by IMU measurements: X̂ₖ₊₁⁻ = f(X̂ₖ⁺, aₘ, ωₘ), Pₖ₊₁⁻ = Aₖ Pₖ⁺ Aₖᵀ + Qₖ, where the process model f integrates the accelerometer and gyroscope readings, and the covariance propagation uses the state-independent (autonomous) Jacobian Aₖ—a key property of the right-invariant formulation.

The update step incorporates the zero-velocity contact constraints and leg kinematics measurements through the right-invariant measurement model. The innovation is computed on the Lie algebra: zₖ = Yₖ − h(X̂ₖ⁻), where h is the right-invariant observation function. The Kalman gain and state update follow: Kₖ = Pₖ⁻ Hₖᵀ (Hₖ Pₖ⁻ Hₖᵀ + Rₖ)⁻¹, X̂ₖ⁺ = exp(Kₖ zₖ) X̂ₖ⁻. This produces the filter-only estimate X̂ₖ⁺.

The neural correction module then processes the filter state and recent measurement history to predict the systematic bias: Δξₖ = NeuralNet(X̂ₖ⁺, {yₜ}ₜ₌ₖ₋ₙᵏ, {X̂ₜ}ₜ₌ₖ₋ₙᵏ), where Δξₖ ∈ se₂(3) is the predicted correction in the Lie algebra. The corrected state is: X̂ₖ_corrected = exp(Δξₖ) X̂ₖ⁺. The neural network architecture consists of a temporal convolutional encoder that processes the measurement history, followed by fully connected layers that output the 9-dimensional correction vector on the Lie algebra (3 for rotation, 3 for velocity, 3 for position).

Training is performed on collected trajectory data with ground-truth poses (from motion capture). The loss function operates on the Lie algebra to respect the geometric structure: L = ‖log(X_gt⁻¹ · exp(Δξ) · X̂⁺)‖², which measures the geodesic distance between the corrected estimate and the ground truth on the SE₂(3) manifold. This geometric loss ensures that rotation, velocity, and position errors are weighted consistently according to the manifold metric, rather than using an ad-hoc Euclidean weighting.

A critical design choice is that the neural network is trained to predict the correction Δξ rather than the full state. This residual learning approach has several advantages: (1) the network only needs to model the systematic bias, not the full dynamics; (2) if the network fails or produces garbage output, the filter-only estimate is still reasonable; (3) the network has a smaller output space to learn, improving sample efficiency. The covariance Pₖ is also modified post-correction: P_corrected = (I − Gₖ) Pₖ⁺ (I − Gₖ)ᵀ + Gₖ R_nn Gₖᵀ, where Gₖ is the neural correction gain and R_nn is the learned correction uncertainty.

## Key Results & Numbers
- 40% reduction in position RMSE compared to standalone RI-EKF on real robot data
- 25% reduction compared to end-to-end neural network baseline, while maintaining geometric consistency
- Neural correction primarily learns contact model bias (60% of correction magnitude), kinematic offset (25%), and IMU bias (15%)
- Inference time of 0.4 ms per cycle (0.15 ms for RI-EKF + 0.25 ms for neural correction), compatible with 1 kHz control
- Consistent performance across trot, walk, and bound gaits without gait-specific retraining
- Covariance estimates from InNKF are 2× more consistent (calibrated) than standard RI-EKF, as measured by NEES metric
- Graceful degradation when neural component produces poor corrections—reverts to filter-only accuracy

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The InNKF is directly applicable to the Mini Cheetah as a state estimation upgrade over standard EKF. The neural correction learns to compensate for the Mini Cheetah's known kinematic calibration errors and soft foot-ground contact dynamics, which are difficult to model analytically. For the RL training pipeline, the InNKF provides more accurate state observations to the PPO policy, reducing the reality gap. The geometric loss function provides a template for incorporating Lie group structure into the RL reward function—penalizing orientation errors using geodesic distance rather than Euler angle differences. The residual learning approach (neural correction over filter baseline) aligns well with the sim-to-real transfer philosophy: the filter provides a physics-based baseline, and the neural component adapts to real-world specifics. The InNKF's calibrated covariance estimates can be used to construct state uncertainty observations for the RL policy, enabling uncertainty-aware locomotion strategies.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The learned error correction paradigm is directly applicable to Cassie's proprioceptive state estimation challenges. Cassie's leaf-spring mechanisms introduce complex compliant contact dynamics that are poorly captured by rigid-body models—exactly the type of systematic error that InNKF's neural component is designed to correct. The geometric-aware training on Lie groups aligns with the mathematical framework used in Cassie's Neural ODE dynamics model, where state evolution should respect manifold structure. At the Controller level, the calibrated covariance from InNKF can inform the CBF-QP safety filter's uncertainty margins: tighter estimates allow less conservative safety constraints. At the Primitives level, the neural correction module provides a template for how to augment model-based components with learned residuals throughout the hierarchical architecture. The InNKF's factored correction (rotation vs. velocity vs. position) could feed different levels of the hierarchy with appropriately scaled uncertainty information.

## What to Borrow / Implement
- Adopt the InNKF architecture as the primary state estimator for both platforms, replacing standalone filters
- Use the geometric (Lie algebra) loss function when training any state-estimation neural components, ensuring manifold consistency
- Apply the residual learning pattern (neural correction over model-based baseline) to other components: e.g., neural residual over MPC at the Controller level
- Incorporate the InNKF's calibrated covariance estimates as uncertainty inputs to the RL policy and the CBF-QP safety filter
- Train the neural correction module using sim-to-real transfer: pre-train in MuJoCo, fine-tune on real robot data

## Limitations & Open Questions
- Neural component requires real-robot training data with ground-truth poses, limiting scalability to new environments
- The correction is applied after the Kalman update, not integrated into the filter dynamics—a tighter integration could yield further improvements
- Long-term position drift is not eliminated, only reduced; absolute positioning still requires external references
- The approach has not been tested during extreme dynamic events (falls, collisions) where the filter assumptions may be strongly violated
