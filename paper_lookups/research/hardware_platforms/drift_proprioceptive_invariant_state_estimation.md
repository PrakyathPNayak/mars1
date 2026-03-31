# DRIFT: Proprioceptive Invariant Robot State Estimation

**Authors:** Tzu-Yuan Lin, Ray Zhang, Maani Ghaffari
**Year:** 2023 | **Venue:** arXiv 2023
**Links:** https://arxiv.org/abs/2311.04320

---

## Abstract Summary
DRIFT is a real-time invariant proprioceptive state estimator for mobile robots using only onboard IMUs and joint kinematics. Built on symmetry-preserving (invariant) Kalman filtering with optional contact and gyro filter modules, it achieves robust state tracking without exteroceptive sensors. Validated on multiple robot platforms including legged robots.

## Core Contributions
- Introduces an invariant extended Kalman filter (InEKF) framework for proprioceptive-only state estimation
- Modular design with pluggable contact detection and gyro bias correction modules
- Achieves real-time performance suitable for control-loop integration on legged robots
- Eliminates dependency on exteroceptive sensors (cameras, LiDAR) for base state estimation
- Validated across multiple robot platforms demonstrating generalizability
- Open-source implementation enabling community adoption and benchmarking
- Leverages Lie group symmetry structure for mathematically principled uncertainty propagation

## Methodology Deep-Dive
DRIFT builds on the invariant extended Kalman filter (InEKF) framework, which exploits the Lie group structure of the robot's state space (SE₂(3) for pose and velocity). Unlike standard EKF formulations, the InEKF guarantees that the linearization error is independent of the state estimate, leading to more consistent and reliable convergence properties. This is particularly important for legged robots where rapid state changes during dynamic locomotion can cause standard EKFs to diverge.

The core estimator fuses IMU measurements (accelerometer and gyroscope) with forward kinematics computed from joint encoders. When a foot is detected to be in contact with the ground, the forward kinematics chain from the base to that foot provides a pseudo-measurement of the base position relative to the contact point. The invariant formulation ensures these measurements are incorporated in a geometrically consistent manner.

The contact detection module is critical: it determines which feet are in stable ground contact and thus provide reliable kinematic measurements. DRIFT supports both threshold-based contact detection (using foot force sensors or motor current) and learned contact detectors. The modularity allows swapping in more sophisticated contact estimators without changing the core filter.

A gyroscope bias correction module handles the well-known drift problem in MEMS gyroscopes. By modeling the bias as a slowly varying state and jointly estimating it with the robot's pose and velocity, DRIFT maintains accurate orientation estimates over extended operation periods. This is essential for legged robots where cumulative orientation drift directly impacts foot placement accuracy.

The entire pipeline runs in real-time on embedded hardware, making it suitable for integration into the 500 Hz control loops typical of legged robot platforms. The computational efficiency comes from the sparse structure of the invariant filter equations and the minimal sensor requirements.

## Key Results & Numbers
- Real-time performance at control-loop frequencies (suitable for 500 Hz+ loops)
- Operates without any exteroceptive sensors (no cameras, LiDAR, or GPS)
- Validated on multiple legged robot platforms with consistent performance
- Open-source implementation available for community use
- Mathematically principled uncertainty estimates via invariant filtering
- Handles dynamic gaits including trotting and bounding

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Proprioceptive state estimation is critical for Mini Cheetah deployment, especially in scenarios where visual sensors are unavailable or unreliable. DRIFT's InEKF framework can provide robust base pose and velocity estimates using only the IMU and joint encoders already available on Mini Cheetah. The real-time performance at control frequency aligns with the 500 Hz PD control loop. This estimator could serve as the state feedback pipeline for the PPO-trained policy, ensuring accurate observations even during dynamic maneuvers in MuJoCo sim and real deployment.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Cassie's hierarchical controller requires accurate base state estimation at every level of the hierarchy. DRIFT's invariant filtering provides mathematically principled state estimates that feed into the Planner (global state), Controller (local tracking), and Safety layer (CBF constraint evaluation). The contact detection module is particularly relevant since Cassie's bipedal gait has distinct stance/swing phases where contact information directly impacts the Neural ODE Gait Phase estimator. The proprioceptive-only operation provides a fallback when the CPTE terrain encoder's exteroceptive inputs are degraded.

## What to Borrow / Implement
- Integrate InEKF state estimator into Mini Cheetah's observation pipeline for robust proprioceptive state feedback
- Use DRIFT's contact detection module to improve gait phase estimation in Cassie's Neural ODE module
- Adopt the invariant filtering framework as the base state estimator for both projects, with learned modules added on top
- Leverage the open-source implementation as a starting point for custom estimator development
- Use DRIFT's uncertainty estimates to weight observations in the RL policy input

## Limitations & Open Questions
- Proprioceptive-only estimation accumulates position drift over time without absolute reference corrections
- Contact detection accuracy degrades on compliant or slippery terrain where contact is ambiguous
- Does not handle terrain elevation changes without additional sensing modalities
- Performance during aerial phases (jumping, bounding with flight phase) relies entirely on IMU integration
- Gyro bias correction assumes slowly varying bias, which may not hold during high-frequency vibrations
