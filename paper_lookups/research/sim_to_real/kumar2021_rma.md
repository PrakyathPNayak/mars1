# RMA: Rapid Motor Adaptation for Legged Robots

**Authors:** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
**Year:** 2021 | **Venue:** RSS (Robotics: Science and Systems)
**Links:** https://arxiv.org/abs/2107.04034

---

## Abstract Summary
RMA introduces a two-phase learning framework that enables legged robots to adapt rapidly to new environments without any fine-tuning at deployment time. The first phase trains a base policy in simulation using model-free reinforcement learning with access to privileged environmental information (terrain friction, payload mass, motor strength, etc.) that would not be available on the real robot. The base policy learns to condition its behavior on an environment factor vector, achieving near-optimal performance across a wide range of simulated conditions.

The second phase trains a lightweight adaptation module that takes a short history of the robot's proprioceptive observations and infers the environment factor vector in real-time. This module is trained via supervised learning to match the privileged environment factors from simulation data. At deployment, the adaptation module replaces the privileged information, allowing the robot to implicitly estimate terrain and body properties from its own sensory experience and adjust its gait accordingly.

The complete system was trained entirely in simulation and deployed zero-shot on a Unitree A1 quadruped robot across diverse real-world terrains including sand, mud, tall grass, pebbles, concrete, and stairs. The robot adapts its gait within fractions of a second to new surfaces and disturbances, including sudden payload changes and motor degradation, without any real-world training data or manual calibration.

## Core Contributions
- **Two-phase training paradigm:** base policy with privileged info followed by adaptation module that infers environment from observation history
- **Environment factor vector:** compact representation of terrain and body properties that conditions the base policy, enabling systematic variation during training
- **Asynchronous adaptation:** the adaptation module runs at a lower frequency than the base policy, reducing computational requirements while maintaining responsiveness
- **Zero-shot sim-to-real transfer** on a commercial quadruped (Unitree A1) across drastically different terrains
- **Robustness to unmodeled perturbations:** the system handles disturbances never seen during training (e.g., someone pushing the robot) through the adaptation module's implicit inference
- **Sub-second adaptation speed** to new terrain properties, enabling reactive behavior during locomotion

## Methodology Deep-Dive
The base policy is trained using PPO in Isaac Gym with domain randomization over a comprehensive set of environment parameters. These parameters include terrain friction coefficients (0.05 to 2.0), terrain restitution, ground slope, payload mass (0 to 5 kg), center of mass offset, motor strength scaling (0.7 to 1.3), joint damping, and observation noise levels. The environment factor vector $e_t$ concatenates all these randomized parameters into a single vector. The base policy $\pi(a_t | o_t, e_t)$ takes the current proprioceptive observation $o_t$ (joint positions, velocities, body orientation, angular velocity) and the privileged factor vector $e_t$, producing target joint positions that are tracked by a PD controller.

The adaptation module is a temporal convolution network (TCN) that processes the last 50 timesteps of proprioceptive observations to produce an estimate $\hat{e}_t$ of the current environment factors. The TCN architecture was chosen over recurrent networks for its parallelizability during training and stable gradient flow. The adaptation module is trained purely via supervised regression: given trajectories collected by the base policy in simulation (where $e_t$ is known), the TCN learns to minimize $\|e_t - \hat{e}_t\|^2$.

A critical design choice is that the adaptation module runs asynchronously at 10 Hz, while the base policy runs at 50 Hz. This means the environment estimate is updated every 5 control steps, which is sufficient for terrain properties that change at the timescale of footsteps. This asynchronous design reduces the computational load on the onboard computer and ensures the adaptation does not interfere with the real-time control loop.

The reward function for the base policy is a weighted combination of: forward velocity tracking ($w=1.0$), lateral velocity penalty ($w=-0.5$), angular velocity tracking ($w=0.5$), torque penalty ($w=-0.0001$), joint acceleration penalty ($w=-2.5 \times 10^{-7}$), action rate penalty ($w=-0.01$), and survival bonus ($w=0.5$). The policy is trained for approximately 5000 iterations with 4096 parallel environments in Isaac Gym, requiring about 2 hours on a single GPU.

At deployment, the system uses only an onboard IMU and joint encoders (no cameras, no force sensors). The observation vector includes 12 joint positions, 12 joint velocities, 3-axis gravity vector (from IMU), and the previous action. The adaptation module infers terrain properties purely from how the robot's joints respond to contact, essentially performing system identification in real-time.

## Key Results & Numbers
- **Zero-shot transfer:** Successful deployment on A1 quadruped across 8+ terrain types without any real-world training
- **Adaptation speed:** Environment factor estimates converge within 0.2-0.5 seconds of encountering new terrain
- **Terrain diversity:** Sand, mud, tall grass, pebbles, concrete, wooden planks, foam, and stairs all traversed successfully
- **Payload robustness:** Handles 0-5 kg payload changes (up to 40% of robot body mass) with automatic gait adaptation
- **Motor degradation:** Maintains stable locomotion when motor strength is reduced by up to 30%
- **Comparison to baselines:** Outperforms domain randomization alone (no adaptation) by 45% in success rate on challenging terrains
- **Training time:** Full pipeline (base policy + adaptation module) trains in approximately 3 hours on a single NVIDIA V100 GPU
- **Inference speed:** Base policy runs at 50 Hz, adaptation module at 10 Hz on onboard Jetson TX2

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High (Critical)**
RMA is foundational for the Mini Cheetah sim-to-real pipeline. The two-phase training approach (privileged base policy + adaptation module) is directly implementable for Mini Cheetah. The architecture maps cleanly: train a PPO base policy in MuJoCo with access to ground-truth terrain parameters, then train a TCN adaptation module to infer those parameters from Mini Cheetah's 12 joint encoders and IMU data. The environment factor vector should include Mini Cheetah-specific parameters: leg mass variations, actuator bandwidth limits, ground friction, and terrain slope.

The asynchronous adaptation design is particularly practical for Mini Cheetah, which has limited onboard compute. Running the adaptation module at 10 Hz while the low-level controller runs at 1 kHz (or 500 Hz for the RL policy) ensures real-time performance. The domain randomization ranges from RMA can be directly used as starting points for Mini Cheetah training, adjusted for its specific hardware characteristics (e.g., Mini Cheetah's proprietary actuators have different torque-speed curves than the A1's). RMA's success with zero-shot transfer on A1 provides strong evidence that the same approach will work for Mini Cheetah.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
RMA's adaptation module concept is directly applicable to Cassie's Contextual Phase-Terrain Encoder (CPTE). The CPTE in Project B needs to infer terrain properties and gait phase from proprioceptive history, which is exactly what RMA's adaptation module does for terrain properties. The privileged-to-deployed information asymmetry in RMA closely mirrors Project B's asymmetric context design in the Dual Asymmetric-Context Transformer, where the training-time teacher has access to privileged simulation state while the student must infer from observation history.

Specific adaptations for Cassie include: (1) the environment factor vector must include Cassie-specific parameters like spring stiffness (Cassie has passive leaf springs that vary with temperature and wear), ground reaction force distribution, and foot-ground contact geometry; (2) the temporal convolution window may need to be longer for bipedal locomotion since each gait cycle takes approximately 0.6-0.8 seconds (vs. 0.3-0.4 for quadrupeds); and (3) the adaptation module should additionally infer gait phase, which is critical for Cassie's phase-dependent control but not needed for the A1's simpler trotting gait.

## What to Borrow / Implement
- **Two-phase training pipeline:** directly adopt the privileged base policy + adaptation module architecture for both Mini Cheetah and Cassie
- **Temporal convolution adaptation module:** implement the TCN architecture for real-time environment estimation from 50-step proprioceptive history
- **Asynchronous execution:** run adaptation at 10 Hz and policy at 50+ Hz to manage computational budget on embedded hardware
- **Domain randomization ranges:** use RMA's published parameter ranges as starting points, calibrate to Mini Cheetah and Cassie hardware
- **Supervision from privileged info:** train CPTE using privileged terrain labels available in simulation, matching RMA's supervised adaptation training

## Limitations & Open Questions
- **No vision:** RMA uses only proprioception; it cannot anticipate terrain changes before contact, limiting proactive adaptation (e.g., slowing down before a visible gap)
- **Fixed environment factor space:** the set of environment parameters is defined at training time; truly novel disturbances outside the training distribution may not be well-estimated by the adaptation module
- **Linear extrapolation assumption:** the TCN's mapping from observation history to environment factors may not generalize to extreme parameter combinations never seen during training
- **Quadruped-specific:** all results are on A1 quadruped; adaptation to bipedal dynamics (Cassie) is untested and may require different temporal scales and factor representations
