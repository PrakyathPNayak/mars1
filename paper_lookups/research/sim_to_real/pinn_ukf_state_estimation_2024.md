# Enhanced Robot State Estimation using Physics-Informed Neural Networks and Unscented Kalman Filter

**Authors:** (2024)
**Year:** 2024 | **Venue:** SPIE
**Links:** [DOI](https://doi.org/10.1117/12.3022666)

---

## Abstract Summary
This paper presents a hybrid state estimation approach that integrates Physics-Informed Neural Networks (PINNs) with the Unscented Kalman Filter (UKF) for legged robot state estimation. The core innovation is embedding the robot's physical dynamic constraints directly into the neural network's loss function, ensuring that learned state predictions are consistent with the governing equations of motion. This physics-informed training regularization reduces drift and improves long-term estimation accuracy compared to purely data-driven or purely model-based methods.

The PINN component learns a mapping from proprioceptive sensor measurements (IMU, joint encoders) to state predictions, but unlike standard neural networks, its training loss includes physics residual terms that penalize violations of the robot's equations of motion: Mq̈ + C(q,q̇)q̇ + g(q) = τ + Jᵀλ. This ensures the network's predictions satisfy the manipulator equation, even in regions of the state space not well-covered by training data. The UKF then fuses these physics-informed neural predictions with the standard process model, using the neural output as a measurement update that provides drift-corrected state estimates.

Validation on proprioceptive data from legged robots demonstrates that the PINN-UKF combination significantly reduces position and velocity drift compared to standalone UKF or standalone neural network approaches, particularly over extended trajectories where accumulated integration errors dominate.

## Core Contributions
- Integration of physics-informed neural networks with the Unscented Kalman Filter for robot state estimation
- Physics residual loss function that embeds the manipulator equation (rigid-body dynamics) as a training regularizer
- Reduced long-term drift by ensuring neural predictions satisfy dynamic consistency constraints
- UKF fusion framework that uses PINN output as a learned measurement model, complementing the process model
- Comparative analysis showing advantages over standalone UKF, standalone NN, and standard NN-UKF combinations
- Validation on legged robot proprioceptive data demonstrating improved multi-minute trajectory estimation
- Analysis of how physics-informed training improves generalization to unseen gaits and terrains

## Methodology Deep-Dive
The PINN architecture processes a window of proprioceptive measurements and outputs a state prediction. The input consists of a temporal buffer of T timesteps containing IMU data (accelerometer aₜ ∈ ℝ³, gyroscope ωₜ ∈ ℝ³), joint positions qₜ ∈ ℝⁿ, joint velocities q̇ₜ ∈ ℝⁿ, and joint torques τₜ ∈ ℝⁿ. The network uses 1D convolutional layers for temporal feature extraction followed by fully connected layers for state prediction: [p̂, v̂, R̂, q̈̂] = PINN(a₁:T, ω₁:T, q₁:T, q̇₁:T, τ₁:T).

The loss function has three components. The data loss penalizes deviation from ground-truth states: L_data = ‖x̂ − x_gt‖². The physics residual loss penalizes violations of the equations of motion: L_physics = ‖M(q)q̈̂ + C(q,q̇)q̇ + g(q) − τ − Ĵᵀλ̂‖², where M is the mass matrix, C is the Coriolis matrix, g is the gravity vector, and λ̂ are estimated contact forces. The boundary loss enforces kinematic constraints: L_boundary = ‖FK(q) − p_foot_expected‖² for feet in contact. The total loss is: L = L_data + α L_physics + β L_boundary, where α and β are weighting hyperparameters.

Computing the physics residual requires evaluating the mass matrix M(q) and Coriolis terms C(q,q̇), which are obtained from the robot's URDF model via automatic differentiation through the Recursive Newton-Euler Algorithm (RNEA). This is implemented using differentiable rigid-body dynamics libraries (e.g., Pinocchio with CasADi backend), enabling gradients of L_physics to flow back through the dynamics computations into the PINN parameters.

The UKF integration uses the standard sigma-point formulation. The process model propagates the state using the IMU-driven rigid-body dynamics. The PINN provides a learned measurement model: the network's output serves as a "virtual sensor" that observes the full state. The measurement noise covariance R_PINN is learned during training as a diagonal matrix representing the network's epistemic uncertainty in each state dimension. The UKF update step fuses the process model prediction with the PINN measurement: x̂⁺ = x̂⁻ + K(y_PINN − h(x̂⁻)), where K is the UKF Kalman gain computed from the sigma points.

A key advantage of the UKF over the EKF in this application is that the sigma-point approach naturally handles the nonlinearities in both the process model and the PINN measurement model without requiring Jacobian computation. The PINN's neural network Jacobian would be expensive to compute for EKF, whereas the UKF evaluates the network at 2n+1 sigma points (where n is the state dimension).

The physics-informed training provides implicit regularization that improves generalization. Even when the training data covers only trotting gaits, the physics residual loss ensures that the network's predictions remain dynamically consistent for unseen gaits (e.g., bounding), because the equations of motion are gait-independent. This physics prior acts as a strong inductive bias that reduces the data requirements for training.

## Key Results & Numbers
- 45% reduction in position drift over 5-minute trajectories compared to standalone UKF
- 30% improvement over standard (non-physics-informed) NN-UKF, demonstrating the value of the physics residual
- Physics-informed training reduces required training data by 50% compared to purely data-driven NN for equivalent accuracy
- PINN inference time of 1.2 ms per step; total PINN-UKF cycle time of 2.5 ms (compatible with 400 Hz control)
- Generalization to unseen gaits: only 15% accuracy degradation when testing on bound gait after training on trot, vs. 60% degradation for standard NN
- Physics residual magnitude decreases by 85% during training, confirming the network learns dynamically consistent predictions
- Velocity estimation RMSE of 0.04 m/s compared to 0.07 m/s for standalone UKF

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The PINN-UKF provides an alternative state estimation approach for the Mini Cheetah that is more drift-resistant than standard filters. The physics-informed training is valuable because the Mini Cheetah's URDF model and dynamics are well-characterized, providing accurate physics residuals for training. For RL sim-to-real transfer, the PINN component can be pre-trained in MuJoCo using the simulated dynamics, and the physics residual ensures the learned estimates respect the robot's equations of motion even during novel behaviors generated by the RL policy. However, the 2.5 ms cycle time is slower than the RI-EKF-based approaches (0.15 ms), which may limit the control loop frequency. The approach is most valuable for the Mini Cheetah when long-term drift reduction is important (e.g., outdoor navigation), rather than for the high-frequency feedback needed within the locomotion controller.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The PINN framework is highly relevant to Cassie's architecture, particularly for integrating with the Neural ODE component at the Primitives level. The physics residual loss function provides a direct template for training Cassie's Neural ODE: instead of learning arbitrary dynamics, the Neural ODE can be regularized to satisfy Cassie's rigid-body equations of motion. This physics-informed Neural ODE would learn residual dynamics (unmodeled effects like cable routing, friction) while ensuring the dominant dynamics are physically consistent. The PINN's dynamic constraint embedding aligns with the RSSM world model's need for physically plausible predictions. At the Controller level, the PINN-UKF provides drift-resistant state estimates that improve the CBF-QP safety filter's stability margin computation over multi-second horizons. The generalization property (training on one gait, testing on others) is valuable for Cassie's multi-gait locomotion repertoire. The sigma-point approach of the UKF naturally handles the nonlinear dynamics of Cassie's spring-loaded mechanisms.

## What to Borrow / Implement
- Adopt the physics residual loss function for training Cassie's Neural ODE dynamics model, ensuring dynamic consistency
- Use the PINN measurement model concept to create a learned state observation module that augments the standard filter
- Implement the differentiable RNEA (via Pinocchio) for computing physics residuals during training for both platforms
- Apply the physics-informed generalization strategy: train on limited gaits with physics constraints to enable zero-shot transfer to new gaits
- Integrate PINN uncertainty estimates (learned R_PINN) as input to the CBF-QP safety filter's constraint tightening

## Limitations & Open Questions
- Computational cost (2.5 ms) is higher than pure filter approaches, limiting the maximum control loop frequency
- The physics residual assumes the URDF model is accurate; systematic modeling errors (mass, inertia) propagate into the training
- Contact force estimation (λ̂ in the physics residual) is challenging and introduces a chicken-and-egg problem with state estimation
- The UKF's Gaussian assumption may be inadequate for multimodal contact state uncertainties that naturally arise in legged locomotion
