# Learning Stable Bipedal Locomotion Skills for Quadrupedal Robots (TumblerNet)

**Authors:** Nature Robotics 2025 Authors
**Year:** 2025 | **Venue:** Nature Robotics
**Links:** [Nature](https://www.nature.com/articles/s44182-025-00043-2)

---

## Abstract Summary
This paper introduces TumblerNet, a deep reinforcement learning controller that enables stable bipedal walking on quadruped robot platforms. The system estimates the robot's center-of-mass (CoM) and center-of-pressure (CoP) in real-time using a learned estimator, feeding these dynamic balance metrics into a locomotion policy that maintains upright bipedal posture on two legs while the other two are lifted. The name "TumblerNet" reflects the tumbler toy analogy—a system that returns to equilibrium after perturbation through passive dynamics augmented by active control.

The controller is trained entirely in simulation using deep RL and transfers to real quadruped hardware for deployment on unstructured outdoor terrains. A key technical contribution is the automatic fall recovery mechanism: when the robot detects impending loss of balance (via CoM-CoP divergence exceeding a safety threshold), it transitions smoothly from bipedal to quadrupedal stance, recovers balance, and optionally returns to bipedal walking. This makes the system robust enough for real-world deployment where perturbations and terrain irregularities are unavoidable.

Experimental validation demonstrates stable bipedal walking on flat ground, uneven terrain, soft surfaces (foam, sand), and inclined planes. The robot maintains bipedal locomotion for extended periods (>60 seconds) on flat ground and successfully negotiates moderate terrain challenges. The fall recovery mechanism engages reliably when balance limits are exceeded, preventing hardware damage and enabling continued operation.

## Core Contributions
- TumblerNet deep RL controller enabling stable bipedal walking on quadruped robot platforms
- Real-time CoM and CoP estimation via learned neural network estimator, without external motion capture
- Automatic fall recovery mechanism triggered by CoM-CoP divergence threshold
- Smooth transition between bipedal and quadrupedal modes for graceful degradation
- Validated on real hardware across flat, uneven, soft, and inclined terrains
- Analysis of dynamic balance margins and stability regions for bipedal quadruped configurations
- Demonstration that quadruped hardware can achieve functional bipedal locomotion without hardware modification

## Methodology Deep-Dive
TumblerNet's architecture consists of three interconnected modules: a CoM/CoP estimator, a balance controller, and a gait generator. The CoM/CoP estimator is a recurrent neural network (GRU, 2 layers, 128 hidden units) that processes a history of proprioceptive observations (joint positions, velocities, IMU data) over a 0.5-second window (25 timesteps at 50 Hz) to estimate the robot's center-of-mass position and velocity in the world frame, and the center-of-pressure location on the ground contact polygon. Ground truth for training is computed from the full simulation state.

The balance controller receives the CoM/CoP estimates along with their temporal derivatives and produces corrective body pose adjustments (trunk roll, pitch, yaw rate, height). The control law is inspired by the linear inverted pendulum model (LIPM): the controller drives the CoM toward a reference trajectory that keeps it within the support polygon defined by the two active ground-contact feet. The corrective signal is parameterized as a learned residual on top of a hand-designed LIPM baseline, enabling the neural network to handle nonlinear dynamics and terrain irregularities that the LIPM cannot capture.

The gait generator produces joint-level targets for the rear (or front, depending on configuration) two legs performing the bipedal walking motion. It operates on a phase-based oscillator: each leg follows a parameterized swing/stance trajectory with learned amplitude, frequency, and offset corrections conditioned on the balance controller's output. The swing leg trajectory is generated as a Bézier curve in Cartesian space, converted to joint targets via analytical inverse kinematics. The stance leg provides support and actively modulates ground reaction forces for balance.

Training uses PPO with an asymmetric actor-critic architecture: the critic receives privileged information (true CoM, CoP, terrain heightmap, contact forces) while the actor receives only onboard-observable quantities. The reward function includes bipedal stability (CoM within support polygon), forward velocity tracking, energy efficiency, foot clearance during swing, and a significant penalty for falling (triggering episode termination). Curriculum learning progressively increases terrain difficulty and perturbation magnitude.

The fall recovery mechanism monitors the distance between the CoM projection and the CoP. When d(CoM_proj, CoP) exceeds a learned threshold (approximately 60–80% of the support polygon radius), the controller initiates a transition to quadrupedal stance by lowering the lifted legs to the ground with a smooth 0.3-second transition trajectory. After recovering quadrupedal balance (d < 20% threshold for 1 second), the robot optionally returns to bipedal mode.

## Key Results & Numbers
- Stable bipedal walking for >60 seconds on flat ground without falling
- Walking speed: 0.15–0.3 m/s in bipedal mode (vs. 0.8–1.5 m/s in quadrupedal mode)
- CoM estimation error: <1.5 cm RMS, CoP estimation error: <2.0 cm RMS
- Fall recovery success rate: >95% across all tested terrain types
- Bipedal-to-quadrupedal transition time: 0.3 seconds
- Handles terrain inclinations up to 8° in bipedal mode, 15° after transition to quadrupedal
- Survives external push perturbations up to 15 N·s impulse in bipedal mode
- Energy cost in bipedal mode: approximately 3× quadrupedal mode CoT (expected due to reduced contact)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
TumblerNet's CoM/CoP estimation module is directly applicable to improving the Mini Cheetah's balance control, even in standard quadrupedal mode. The learned estimator (GRU-based, trained with privileged information) provides a robust alternative to model-based state estimation, which can be unreliable during dynamic maneuvers. The fall recovery mechanism is valuable for the Mini Cheetah's real-world deployment: detecting impending falls and transitioning to a safe recovery posture prevents hardware damage.

Additionally, exploring bipedal modes on the Mini Cheetah platform could enable novel capabilities (rearing up to inspect elevated objects, transitioning over narrow obstacles). The residual learning approach (neural network corrections on top of LIPM) provides a principled way to combine physics-based priors with learned adaptations.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This paper is among the most directly relevant to the Cassie project. The CoM/CoP estimation methodology can be directly integrated into Cassie's balance control architecture, where accurate CoM tracking is essential for the inverted pendulum dynamics of bipedal walking. The GRU-based estimator architecture, trained with privileged information and deployed with onboard sensors only, aligns with Cassie's asymmetric actor-critic training approach.

The fall recovery mechanism is critically important for Cassie, where falls are both dangerous (hardware damage) and expensive (resetting the robot). The CoM-CoP divergence threshold provides a concrete implementation for Cassie's CBF-QP safety layer: the control barrier function can enforce the constraint d(CoM_proj, CoP) < d_max, ensuring the robot remains within its balance envelope. The smooth bipedal recovery trajectories provide templates for Cassie's fall recovery primitives.

## What to Borrow / Implement
- Implement the GRU-based CoM/CoP estimator for both Mini Cheetah and Cassie using privileged-information training
- Adopt the CoM-CoP divergence metric as a balance safety constraint in Cassie's CBF-QP layer
- Use the residual learning approach (learned corrections on LIPM) for Cassie's balance controller
- Integrate the fall recovery transition mechanism (smooth 0.3s transition) into both platforms' safety systems
- Apply the phase-based gait generator with Bézier swing trajectories for Cassie's low-level Controller

## Limitations & Open Questions
- Bipedal walking speed (0.15–0.3 m/s) is significantly slower than quadrupedal, limiting practical utility
- CoM/CoP estimator accuracy degrades on highly uneven terrain where the ground plane assumption breaks down
- Fall recovery assumes sufficient space for quadrupedal transition; confined spaces may not permit this
- Energy cost 3× higher in bipedal mode raises questions about practical deployment duration on battery power
