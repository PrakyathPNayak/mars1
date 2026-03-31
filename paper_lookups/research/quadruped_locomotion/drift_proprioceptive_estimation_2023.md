# DRIFT: Deep Residual Inertial Feature Transform for Proprioceptive Robot State Estimation

**Authors:** (2023)
**Year:** 2023 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2311.04320)

---

## Abstract Summary
DRIFT introduces a modular deep learning framework for real-time proprioceptive state estimation on legged robots, using only inertial measurement unit (IMU) data and joint kinematics. The system processes IMU readings (accelerometer and gyroscope) along with joint encoder positions and velocities to produce accurate odometry estimates without requiring any external perception sensors such as cameras or LiDAR. This makes it especially robust in perceptually degraded environments—darkness, fog, dust, or featureless terrain—where visual and LiDAR-based methods fail.

The architecture features a modular design with optional plug-in components for contact estimation and gyroscope bias filtering. The core module uses a deep residual network to learn a feature transform of the inertial data that captures the essential motion information. A recurrent component (LSTM or GRU) then integrates these features over time to produce velocity and pose estimates. The optional contact module learns to detect foot-ground contact from proprioceptive signals, improving the kinematic leg odometry. The optional gyroscope filter learns to correct bias drift in the gyroscope readings.

DRIFT achieves competitive or superior performance compared to state-of-the-art proprioceptive estimators across multiple legged robot platforms, with inference times under 1 ms per step, making it suitable for real-time deployment in control loops.

## Core Contributions
- Modular proprioceptive state estimation architecture with plug-and-play contact detection and gyro filtering components
- Deep residual feature transform that extracts motion-relevant representations from raw IMU data
- Real-time inference (<1 ms) enabling integration in high-frequency control loops (1 kHz)
- Robustness validation in perceptually degraded conditions where vision-based methods fail
- Cross-platform evaluation on multiple legged robot datasets
- Open-source implementation for community adoption and benchmarking
- Comparison showing advantages over classical filter-based and end-to-end learning approaches

## Methodology Deep-Dive
The DRIFT architecture consists of three stages: feature extraction, temporal integration, and state output. The feature extraction stage processes a window of N recent IMU measurements (typically N=200, corresponding to 200 ms at 1 kHz) through a deep residual network. Each residual block consists of 1D convolutions, batch normalization, and ReLU activations, with skip connections preserving gradient flow. The network transforms the 6-dimensional raw IMU input (3-axis accelerometer + 3-axis gyroscope) into a D-dimensional feature vector (typically D=128) that captures motion patterns invariant to sensor noise and bias.

The temporal integration stage uses a two-layer LSTM with hidden dimension 256 to accumulate motion information over longer time horizons. The LSTM takes the feature vectors as input and maintains an internal state that implicitly tracks position and orientation. The output at each timestep is decoded through a fully connected layer to produce the estimated velocity in the body frame: v̂ = MLP(LSTM(ResNet(IMU_window))). Position is obtained by integrating the velocity estimate, and orientation is obtained from a separate branch that processes the gyroscope data.

The optional contact detection module operates in parallel with the main estimator. It takes joint positions, velocities, and torques as input and outputs per-leg contact probabilities: P(contact_i) = σ(MLP(qᵢ, q̇ᵢ, τᵢ)), where σ is the sigmoid function and i indexes the legs. These contact probabilities modulate the leg kinematics contribution to the odometry estimate. When a leg is in contact (P > 0.5), its forward kinematics provides a velocity constraint: v_foot = 0 in the world frame, which translates to a body velocity estimate through the Jacobian: v_body = −J(q)⁻¹ · q̇.

The optional gyroscope filter is a small neural network that estimates and subtracts the time-varying gyroscope bias: ω_corrected = ω_raw − MLP(ω_raw, history). This learned bias correction adapts to temperature drift and mechanical vibrations that cause systematic gyroscope errors.

Training uses a combination of velocity and position losses. The velocity loss is the L2 norm between estimated and ground-truth body velocities: L_vel = ‖v̂ − v_gt‖². The position loss penalizes drift over trajectory segments: L_pos = ‖∫v̂dt − Δp_gt‖². The total loss is L = L_vel + α·L_pos, where α is an annealing weight that increases during training to emphasize long-term drift reduction.

## Key Results & Numbers
- Position drift of 1.2–2.5% of distance traveled, competitive with visual-inertial odometry in well-lit conditions
- Inference time of 0.3–0.8 ms per step on an NVIDIA Jetson Orin, enabling 1 kHz real-time deployment
- Contact detection accuracy of 92–96% across different gaits and terrains
- Gyroscope bias correction reduces orientation drift by 40–60% compared to raw integration
- Outperforms classical complementary filters and EKF-based proprioceptive estimators by 20–35% in position accuracy
- Robust performance maintained in darkness, fog, and dust conditions where visual methods degrade by 5–10×

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
DRIFT is directly applicable to the Mini Cheetah platform as a proprioceptive state estimation module that can run alongside the RL locomotion policy. The 12-DoF Mini Cheetah provides exactly the joint encoder and IMU inputs that DRIFT requires. The learned contact detection eliminates the need for hardware contact sensors, which are often unreliable on the Mini Cheetah's rubber feet. For sim-to-real transfer, DRIFT can be trained in MuJoCo simulation with domain randomization on sensor noise parameters and then fine-tuned on real hardware data. The modular architecture allows progressive deployment: start with the core IMU module, then add contact detection and gyro filtering as needed. The real-time inference speed ensures no computational bottleneck in the 1 kHz control loop.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
DRIFT provides a foundation for Cassie's proprioceptive state estimation within the hierarchical architecture. The contact detection module is crucial for Cassie, where accurate stance/swing detection drives the Planner level's footstep timing. The body velocity estimate feeds into the Capture Point computation in the safety layer: CP = x + v/ω₀, requiring accurate velocity. The modular design aligns with Cassie's hierarchical architecture—the core velocity estimator can run at the Controller level (1 kHz), while the contact detection feeds the Planner level at a lower rate. The LSTM-based temporal integration implicitly captures Cassie's walking dynamics, potentially complementing the RSSM world model at the Primitives level. The learned gyroscope filter addresses a known challenge on Cassie hardware where IMU vibrations during walking corrupt orientation estimates.

## What to Borrow / Implement
- Deploy DRIFT's contact detection module as the proprioceptive contact estimator for both platforms, replacing unreliable hardware sensors
- Integrate the body velocity estimator into the RL observation space, providing a more accurate velocity signal than direct sensor readings
- Train DRIFT in MuJoCo with domain randomization on IMU noise, bias drift, and latency to prepare for sim-to-real transfer
- Use DRIFT's modular architecture as a template for the state estimation pipeline, enabling incremental deployment and testing
- Adopt the residual feature extraction approach for processing raw IMU data in the RL policy's observation encoder

## Limitations & Open Questions
- Position estimate drifts over long trajectories (>100 m) due to velocity integration without absolute position correction
- Training requires ground-truth pose data (e.g., from motion capture), which limits the diversity of training environments
- The LSTM temporal integration introduces a latency of ~10 ms for the recurrent state to converge after sudden motion changes
- Cross-robot generalization is limited; the model must be retrained or fine-tuned for each new platform
