---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/learning_bipedal_walking_humanoids_current_feedback.md

**Title:** Learning Bipedal Walking for Humanoids with Current Feedback
**Authors:** Rohan P. Singh, Mehdi Benallegue, Mitsuharu Morisawa, Rafael Cisneros, Fumio Kanehiro
**Year:** 2023
**Venue:** arXiv / IEEE Access
**arXiv / DOI:** arXiv:2303.03724

**Abstract Summary (2–3 sentences):**
This paper demonstrates zero-shot sim-to-real reinforcement learning for bipedal walking on the full-sized HRP-5P humanoid robot by incorporating actuator current feedback as an observation modality. The approach trains a PPO policy in MuJoCo using a purposely degraded actuator simulation model, then leverages real-time motor current measurements during deployment to bridge the sim-to-real gap without requiring complex memory architectures or recurrent networks. The resulting simple feedforward policy outperforms classical model-based controllers on uneven terrain, demonstrating that direct actuator sensing can be a powerful tool for sim-to-real transfer.

**Core Contributions (bullet list, 4–7 items):**
- First demonstration of zero-shot RL-based walking on a full-sized humanoid (HRP-5P, 1.7m, 70kg)
- Actuator current feedback as a direct observation for bridging the sim-to-real gap in locomotion
- Purposely degraded actuator model strategy that forces the policy to rely on current feedback for accurate torque estimation
- Simple feedforward MLP policy achieving robust walking without recurrence or memory
- Outperformance of classical model-based controllers on uneven terrain with the learned policy
- Analysis showing that current feedback provides implicit ground truth about actual torque output, contact forces, and load conditions

**Methodology Deep-Dive (3–5 paragraphs):**
The central insight of this work is that motor current measurements provide a direct, low-latency signal about the actual torques being produced by the robot's actuators, which can dramatically simplify the sim-to-real transfer problem. In most RL-based locomotion approaches, the sim-to-real gap is dominated by actuator modeling errors—the simulated motors produce torques that differ from the real actuators due to friction, backlash, thermal effects, and nonlinear dynamics. Rather than attempting to build a high-fidelity actuator model or randomize over actuator parameters, the authors propose feeding the measured motor currents directly into the policy's observation space, allowing the network to learn to interpret these signals as ground truth about actual torque output.

The training procedure uses a deliberately simplified and degraded actuator model in MuJoCo. Instead of modeling the HRP-5P's complex harmonic drive actuators with full fidelity, the simulator uses a basic torque-controlled model with intentionally inaccurate dynamics. During training, the simulated "current" observations are computed from the simplified model and corrupted with additional noise. This degradation forces the policy to learn a mapping from current observations to appropriate actions that is robust to modeling errors, because the policy cannot overfit to a high-fidelity simulator—the simulator is intentionally wrong. During real-world deployment, the actual motor current measurements replace the simulated ones, and because the policy has learned to be robust to noisy and inaccurate current signals, it adapts seamlessly.

The policy is a standard feedforward MLP (3 hidden layers of 256 units each) trained with PPO. The observation space includes joint positions, joint velocities, pelvis IMU data (orientation quaternion and angular velocity), commanded walking velocity, and critically, the motor current measurements for all actuated joints. The action space is target joint positions tracked by the robot's existing low-level PD controllers. Training runs for approximately 50 million environment steps in MuJoCo with moderate domain randomization applied to ground friction (0.5–1.5), body mass (±10%), and initial state perturbations. The reward function encourages forward velocity tracking, upright posture, smooth joint motions, and penalizes excessive energy consumption and ground impact forces.

The real-world deployment on HRP-5P uses the robot's existing motor driver boards which provide current measurements at 1 kHz. The RL policy runs at 100 Hz on the robot's control computer, receiving downsampled current readings along with the standard proprioceptive observations. The authors conduct experiments on flat indoor floors, outdoor concrete with small cracks and bumps, and surfaces with deliberately placed uneven tiles. The learned policy maintains stable walking across all terrains, while the classical model-based controller (a ZMP-based pattern generator with whole-body control) fails on the uneven outdoor surfaces due to its reliance on accurate contact state estimation.

Ablation studies demonstrate the critical role of current feedback: removing current observations from the policy input and retraining results in significant performance degradation on hardware, with the robot failing to walk on uneven terrain and exhibiting poor torque tracking on flat ground. The authors also compare against a recurrent (LSTM) policy without current feedback and show that their feedforward approach with current feedback achieves comparable or better transfer performance, suggesting that current measurements provide a more direct and reliable signal than temporal memory for bridging the actuator gap.

**Key Results & Numbers:**
- Zero-shot sim-to-real transfer on HRP-5P (1.7m tall, 70 kg life-sized humanoid)
- Feedforward MLP outperforms classical ZMP-based controller on uneven terrain
- Simple 3-layer MLP sufficient—no RNN, transformer, or memory needed
- Walking speeds up to 0.4 m/s on flat ground, 0.2 m/s on uneven terrain
- Policy runs at 100 Hz on onboard computer
- Current feedback ablation shows 60%+ failure rate without current observations on uneven terrain
- Energy consumption comparable to classical controller on flat ground, superior on rough terrain

**Relevance to Project A (Mini Cheetah):** MEDIUM — The current feedback concept is applicable to motor modeling on the Mini Cheetah, which also uses electric actuators with measurable currents. Incorporating current observations could improve torque estimation accuracy and sim-to-real transfer for the quadruped.

**Relevance to Project B (Cassie HRL):** HIGH — Cassie uses series elastic actuators (SEAs) where actuator dynamics are a primary source of sim-to-real gap. The current feedback approach could be incorporated into the Controller level of the hierarchy to improve low-level torque tracking accuracy. This provides an alternative or complement to the Neural ODE actuator modeling approach.

**What to Borrow / Implement:**
- Actuator current feedback as an additional observation channel for the Controller level
- Purposely degraded actuator model training strategy as a principled approach to sim-to-real robustness
- Validation that feedforward policies with good observation design can match recurrent policies for sim-to-real
- The insight that direct sensing of actuator state can substitute for complex memory architectures
- Experimental protocol for evaluating sim-to-real transfer quality through terrain variation tests

**Limitations & Open Questions:**
- Limited to walking gaits only; no running, jumping, or dynamic maneuvers demonstrated
- HRP-5P specific actuator modeling and current characteristics may not transfer directly to Cassie's SEAs
- Maximum walking speed of 0.4 m/s is conservative for a humanoid of HRP-5P's size
- Uneven terrain evaluation limited to small perturbations (few cm); no stairs or large obstacles
- Current measurements may be noisy or delayed on different robot platforms
- Does not address how current feedback interacts with more complex policy architectures (transformers, hierarchical)
---
