# Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events

**Authors:** Various
**Year:** 2023 | **Venue:** CoRL Workshop 2023
**Links:** https://openreview.net/forum?id=yt3tDB67lc5

---

## Abstract Summary
This paper combines invariant extended Kalman filtering with deep learning-based contact detection for legged robot state estimation. The learned contact detector replaces unreliable physical contact sensors, using only proprioceptive signals (joint positions, velocities, torques) to estimate ground contact. This hybrid approach achieves robust state estimation even when hardware contact sensors fail.

## Core Contributions
- Combines invariant EKF (InEKF) with a learned contact detection module for hybrid classical-learned estimation
- Replaces fragile hardware contact sensors with a proprioceptive-only learned contact detector
- Uses joint positions, velocities, and torques as inputs to a neural network contact classifier
- Demonstrates improved state estimation accuracy over both pure InEKF and pure learned approaches
- Handles sensor failure gracefully by eliminating dependency on physical contact sensors
- Validates on real legged robot data with diverse terrains and gaits
- Shows that contact detection quality is the primary bottleneck in kinematic-inertial state estimation

## Methodology Deep-Dive
The paper identifies contact detection as the critical bottleneck in kinematic-inertial state estimation for legged robots. Traditional approaches rely on foot force sensors, ground reaction force thresholds, or motor current measurements to determine which feet are in contact with the ground. These methods are prone to failure: force sensors break, thresholds require terrain-specific tuning, and motor current measurements are noisy during dynamic impacts.

The proposed solution trains a neural network to classify contact state from purely proprioceptive signals. The network takes as input a temporal window of joint positions, velocities, and commanded torques for each leg, and outputs a binary contact probability for each foot. The training data comes from runs where reliable contact measurements are available (e.g., lab conditions with force plates), and the trained model generalizes to conditions where hardware sensors are unavailable or unreliable.

The learned contact detector is integrated into an invariant extended Kalman filter (InEKF) framework. When the contact detector indicates a foot is in stable ground contact, the InEKF incorporates a kinematic measurement update using the forward kinematics chain from the base to that foot. The invariant formulation ensures consistent uncertainty propagation regardless of the robot's state, providing more reliable covariance estimates than standard EKF approaches.

The hybrid approach leverages the strengths of both paradigms: the neural network handles the perceptual challenge of contact detection where hand-designed rules struggle, while the InEKF provides a principled probabilistic framework for state fusion with guarantees on consistency and convergence. The learned contact probabilities are used as soft weights in the measurement update, allowing gradual transitions between contact and non-contact states rather than abrupt switches.

Extensive validation demonstrates that the quality of contact detection has a larger impact on state estimation accuracy than the choice of filter formulation. Even a simple EKF with perfect contact information outperforms a sophisticated InEKF with noisy contact sensors, motivating the investment in learned contact detection.

## Key Results & Numbers
- Robust contact detection from proprioception alone, eliminating dependency on hardware contact sensors
- Improved state estimation accuracy over pure InEKF with threshold-based contact detection
- Handles sensor failure scenarios where physical contact sensors are damaged or miscalibrated
- Validates across diverse gaits and terrain types on real robot hardware
- Demonstrates that contact detection quality is the dominant factor in estimation performance
- Soft contact probabilities improve estimation smoothness during gait transitions

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Contact estimation is critical for Mini Cheetah terrain adaptation and state estimation. Physical force sensors are often unreliable on small quadrupeds due to mechanical shock and calibration drift. A learned contact detector using only joint data (positions, velocities, torques from the 12 DoF system) can provide robust contact information for both state estimation and gait adaptation. This directly improves the observation quality fed to the PPO-trained policy and can be integrated into the 500 Hz control loop.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Learned contact events are fundamental to multiple components of Cassie's hierarchy. The Neural ODE Gait Phase estimator requires accurate contact timing to track the gait cycle. The LCBF safety layer needs to know which feet are in contact to evaluate the capture point and apply appropriate CBF constraints. The Controller level uses contact state to select appropriate impedance parameters. Replacing Cassie's physical contact sensors with a learned detector improves robustness across all hierarchy levels.

## What to Borrow / Implement
- Train a learned contact detector for Mini Cheetah using proprioceptive data (joint pos/vel/torque from all 12 joints)
- Integrate learned contact probabilities into the InEKF state estimator for both projects
- Use soft contact probabilities (rather than binary) for smoother gait phase transitions in Cassie's Neural ODE
- Feed learned contact events into the LCBF safety layer for accurate support polygon estimation
- Collect training data in MuJoCo simulation where ground-truth contact is available, then fine-tune on hardware data

## Limitations & Open Questions
- Learned contact detector requires training data with reliable ground-truth contact labels
- May struggle with novel contact scenarios (e.g., edge contacts, partial foot contacts) not seen in training
- Proprioceptive-only contact detection may be ambiguous on very soft or deformable terrain
- Latency in neural network inference adds delay to the contact detection loop
- Generalization across different robot platforms requires retraining with platform-specific data
