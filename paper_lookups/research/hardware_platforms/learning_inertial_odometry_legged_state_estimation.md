# Learning Inertial Odometry for Dynamic Legged Robot State Estimation

**Authors:** Russell Buchanan, Hartmut Geyer, Marco Hutter
**Year:** 2022 | **Venue:** PMLR / CoRL 2022
**Links:** https://proceedings.mlr.press/v164/buchanan22a.html

---

## Abstract Summary
This paper learns inertial odometry directly from IMU and joint encoder data using deep networks, enabling accurate state estimation during dynamic locomotion without visual sensors. It demonstrates that learned proprioceptive estimators can match or exceed hand-designed estimators for legged robots, achieving significant error reductions over traditional methods.

## Core Contributions
- Learns end-to-end inertial odometry from raw IMU and joint encoder data for legged robots
- Demonstrates 37-48% error reduction over traditional state estimation methods
- Operates without any visual sensors, relying purely on proprioceptive signals
- Handles highly dynamic gaits where traditional estimators often fail
- Shows that data-driven approaches can surpass hand-engineered estimator pipelines
- Validates on real legged robot data with diverse locomotion behaviors
- Provides insights into what proprioceptive features the network learns to extract

## Methodology Deep-Dive
The core idea is to replace the hand-engineered components of traditional legged robot state estimators (contact detection, kinematic model selection, noise parameter tuning) with a single learned model that maps raw sensor readings to state estimates. The network takes as input sequences of IMU measurements (3-axis accelerometer and gyroscope at high frequency) and joint encoder readings (position and velocity for all joints).

The network architecture uses temporal convolutional layers followed by recurrent units (LSTM or GRU) to capture both short-term dynamics (individual footsteps, impacts) and long-term patterns (gait cycles, drift accumulation). The temporal convolutions extract local features from the high-frequency IMU data, while the recurrent layers maintain a hidden state that tracks the robot's trajectory over longer horizons. This hybrid architecture balances computational efficiency with temporal modeling capacity.

Training uses a supervised learning approach with ground-truth trajectories obtained from motion capture systems. The loss function combines position and orientation errors with velocity terms that encourage smooth predictions. Importantly, the training data spans diverse locomotion behaviors—walking, trotting, bounding, and turning—to ensure the learned estimator generalizes across the robot's operating envelope. Data augmentation through simulated noise injection further improves robustness.

A key insight from the work is that the learned model implicitly discovers contact detection and gait phase estimation from the raw sensor signals. Analysis of the network's intermediate representations reveals neurons that activate in synchrony with foot contact events, despite never being explicitly trained on contact labels. This suggests that contact-aware state estimation naturally emerges when optimizing for trajectory accuracy.

The inference pipeline is lightweight enough for real-time deployment, requiring only a single forward pass through the network per estimation step. This makes it suitable for integration into control loops running at hundreds of hertz, as required by dynamic legged locomotion.

## Key Results & Numbers
- 37-48% reduction in state estimation error compared to traditional EKF-based methods
- Works during highly dynamic gaits (bounding, fast trotting) where traditional estimators degrade
- No visual sensors required—purely proprioceptive operation
- Real-time inference suitable for control-loop integration
- Validated on real robot data across diverse locomotion behaviors
- Learned features implicitly capture contact events and gait phases

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Learned inertial odometry is directly applicable to Mini Cheetah for blind locomotion scenarios. The 37-48% improvement over traditional estimators would directly translate to better state feedback for the PPO-trained policy, especially during dynamic gaits where Mini Cheetah's standard EKF may degrade. The proprioceptive-only requirement matches Mini Cheetah's sensor suite (IMU + joint encoders). The approach can be trained in MuJoCo simulation with ground-truth states and then deployed on hardware.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Improved state estimation directly benefits all levels of Cassie's hierarchy by providing cleaner observation inputs. The implicit contact detection learned by the network aligns with the needs of the Neural ODE Gait Phase estimator, potentially providing learned contact signals as input. The purely proprioceptive nature provides robustness when Cassie's exteroceptive sensors fail. Better state estimates feed into the LCBF safety layer, where accurate state knowledge is critical for CBF constraint evaluation.

## What to Borrow / Implement
- Train a learned inertial odometry network in MuJoCo simulation for Mini Cheetah using ground-truth state labels
- Use the implicit contact detection features as input to gait phase estimation in Cassie's Neural ODE module
- Replace or augment the standard EKF in both projects with the learned estimator for dynamic gaits
- Apply the temporal CNN + RNN architecture for proprioceptive feature extraction in observation encoders
- Leverage the training methodology (diverse gaits + noise augmentation) for robust estimator training

## Limitations & Open Questions
- Requires ground-truth trajectory data for supervised training, typically from motion capture
- Position estimate drifts over time without absolute position corrections (loop closure or GPS)
- Generalization to locomotion modes and terrain types not represented in training data is uncertain
- The supervised training paradigm means the estimator may not adapt online to changing dynamics (e.g., payload changes)
- Integration with RL policies trained on different state representations requires careful observation space alignment
