# Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Detection

**Authors:** Lin, T., Zhang, M., Maximo, M. R. O. A., Ghaffari, M.
**Year:** 2022 | **Venue:** CoRL (PMLR)
**Links:** [PDF](https://proceedings.mlr.press/v164/lin22b/lin22b.pdf)

---

## Abstract Summary
This paper presents a state estimation framework for legged robots that combines the Right-Invariant Extended Kalman Filter (RI-EKF) on Lie groups with a deep learning-based contact detection network. The RI-EKF formulates the state estimation problem on the SE₂(3) Lie group, which encodes the robot's rotation, velocity, and position as a single matrix Lie group element. This geometric formulation ensures that the estimation error dynamics are independent of the current state estimate (a property called "log-linear"), leading to consistent covariance estimates and improved convergence.

The learned contact detection component replaces traditional threshold-based contact sensing (which requires tuning and hardware contact sensors) with a neural network that infers foot contact states from joint encoders and IMU data alone. The network outputs per-leg contact probabilities that modulate the measurement updates in the RI-EKF—when a foot is confidently in contact, the zero-velocity constraint is applied; otherwise, the measurement is down-weighted or ignored.

The framework is validated on the MIT Mini Cheetah robot, demonstrating superior performance over standard EKF approaches and threshold-based contact detection. The Lie group formulation provides geometric consistency guarantees that standard Euclidean EKFs lack, particularly during aggressive maneuvers with large rotations.

## Core Contributions
- Right-Invariant EKF on the SE₂(3) Lie group for legged robot state estimation, with provable log-linear error dynamics
- Deep learning contact detection that eliminates hardware contact sensors, using only proprioceptive signals
- Fusion of learned contact probabilities as measurement confidence weights in the RI-EKF update step
- Hardware validation on the MIT Mini Cheetah robot across multiple gaits and terrains
- Demonstration that geometric (Lie group) formulation outperforms Euclidean EKF, especially during large rotations
- Open-source implementation compatible with the Mini Cheetah software stack
- Comprehensive ablation study separating contributions of RI-EKF and learned contacts

## Methodology Deep-Dive
The state is represented as an element of the extended special Euclidean group SE₂(3), which is the matrix Lie group consisting of 5×5 matrices: X = [R v p; 0 0 0 1; 0 0 0 0 1] ∈ SE₂(3), where R ∈ SO(3) is the rotation matrix, v ∈ ℝ³ is the velocity, and p ∈ ℝ³ is the position, all expressed in the world frame. The Lie algebra se₂(3) provides a 9-dimensional error representation: ξ = log(X̂⁻¹X) = [δθ; δv; δp] that captures rotation, velocity, and position errors simultaneously.

The RI-EKF prediction step propagates the state using IMU measurements. Given accelerometer reading a_m and gyroscope reading ω_m, the continuous-time dynamics on SE₂(3) are: Ẋ = f_u(X) + XL(w), where f_u encodes the rigid-body dynamics driven by IMU inputs and w is the process noise. The key property is that the right-invariant error η = X̂⁻¹X satisfies η̇ = A_t η + noise, where A_t depends only on the IMU measurements, not on the current state estimate. This "autonomous error" property means the linearized error dynamics are exact up to first order, avoiding the inconsistency issues of standard EKF.

The measurement model uses the zero-velocity constraint for feet in contact. For a foot in contact with the ground at known position d (from forward kinematics), the measurement is: y = X⁻¹[d; 0; 1] = [R^T(p_foot − p); R^T v_foot; 1], which is a right-invariant observation. The innovation is computed on the Lie algebra, and the Kalman gain K updates the state via the exponential map: X̂⁺ = exp(K · innovation) · X̂⁻.

The contact detection network is a temporal convolutional network (TCN) that processes a window of 50 ms of proprioceptive data. Input features include joint positions (12D), joint velocities (12D), joint torques (12D), and IMU data (6D), totaling 42 dimensions per timestep. The TCN output is four contact probabilities (one per foot), passed through a sigmoid activation. Training uses binary cross-entropy loss against ground-truth contact labels obtained from MuJoCo simulation or force plate measurements.

The contact probabilities are integrated into the RI-EKF measurement update by modulating the measurement noise covariance. For foot i with contact probability p_i, the measurement noise is: R_i = R_base / p_i + R_large · (1 − p_i), effectively trusting the zero-velocity constraint proportionally to the contact confidence. This soft-switching avoids the discontinuous jumps that hard contact thresholds cause in the estimator.

## Key Results & Numbers
- Position RMSE of 3.2 cm over 50 m traversal on Mini Cheetah hardware, 35% improvement over standard EKF
- Contact detection accuracy of 94.7% on Mini Cheetah trotting data, matching hardware force sensors
- RI-EKF reduces orientation estimation error by 45% compared to Euclidean EKF during aggressive turning maneuvers
- End-to-end inference time of 0.15 ms per estimation cycle, well within the 1 kHz control loop budget
- Validation across trot, bound, and pronk gaits at speeds up to 2.5 m/s
- Ablation: RI-EKF alone provides 20% improvement; learned contacts alone provide 15% improvement; combined gives 35%

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is critically relevant as it is directly validated on the Mini Cheetah hardware platform used in Project A. The RI-EKF + learned contact detection framework can be adopted as-is for the Mini Cheetah's state estimation pipeline. The open-source implementation integrates with the Mini Cheetah software stack, minimizing implementation effort. For the RL locomotion policy, accurate state estimation is essential—the PPO policy receives velocity and orientation estimates as observations, and estimation errors directly degrade policy performance. The learned contact detection is particularly valuable because the Mini Cheetah's rubber feet lack reliable hardware contact sensors. The Lie group formulation ensures consistent estimates during the aggressive maneuvers that RL policies often produce, which standard EKFs struggle with. During sim-to-real transfer, the domain randomization should include variations in the contact detection network's accuracy to ensure the RL policy is robust to estimation uncertainty.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The RI-EKF framework is highly applicable to Cassie's state estimation needs. The SE₂(3) Lie group formulation naturally extends to Cassie's floating-base dynamics. The learned contact detection is valuable because Cassie's leaf-spring mechanisms make ground truth contact detection challenging from force measurements alone. At the Controller level, the RI-EKF provides the base pose and velocity estimates needed for whole-body control. The contact probabilities feed into the Planner level's gait phase estimation, informing footstep timing decisions. The geometric consistency of the RI-EKF is especially important for Cassie, where large pitch oscillations during walking would cause Euclidean EKF inconsistency. The soft-switching contact integration (modulating noise covariance by contact probability) provides a template for Cassie's CBF-QP safety filter, which needs continuous contact state information rather than binary contact signals.

## What to Borrow / Implement
- Adopt the RI-EKF on SE₂(3) as the core state estimator for both platforms, replacing Euclidean EKF implementations
- Train the contact detection TCN in MuJoCo simulation with domain randomization on ground friction, foot compliance, and sensor noise
- Use the soft-switching contact integration (noise covariance modulation) as a template for fusing learned contact probabilities into any filter-based estimator
- Integrate the contact probabilities as additional observations in the RL policy's input space for both platforms
- Leverage the open-source Mini Cheetah implementation as a starting point and adapt for Cassie's kinematic chain

## Limitations & Open Questions
- The zero-velocity contact constraint assumes rigid flat ground; compliant or uneven terrain introduces systematic measurement bias
- Contact detection accuracy degrades for gaits not seen during training (e.g., training on trot, testing on gallop)
- The SE₂(3) formulation does not naturally accommodate additional state variables (e.g., foot positions as landmarks) without extending the group
- Long-term position drift remains an issue since the estimator relies on dead-reckoning with contact constraints; no loop closure or absolute position updates are incorporated
