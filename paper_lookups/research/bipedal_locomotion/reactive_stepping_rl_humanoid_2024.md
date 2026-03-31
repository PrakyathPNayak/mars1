# Reactive Stepping for Humanoid Robots using Reinforcement Learning: Application to Standing Push Recovery on the Exoskeleton Atalante

**Authors:** (2024)
**Year:** 2024 | **Venue:** IEEE
**Links:** IEEE (2024)

---

## Abstract Summary
This paper presents an RL-based framework for reactive stepping on humanoid robots, with real-world deployment on the Atalante lower-limb exoskeleton. The system learns when and where to step autonomously in response to external perturbations, eliminating the need for explicit balance criteria or hand-designed stepping heuristics. The RL agent observes the robot's full state (body pose, joint positions/velocities, estimated external forces) and outputs a continuous stepping decision: whether to step, where to place the foot, and with what timing. This unified policy handles both standing balance recovery and walking gait perturbation rejection.

A key contribution is the successful deployment on physical hardware — the Atalante exoskeleton — demonstrating that RL-based reactive stepping transfers from simulation to a real bipedal system with significant modeling uncertainties. The Atalante presents unique challenges: it has no ankle actuation (fully actuated at hip and knee only), limited joint speed, and carries a human patient, making conservative and reliable recovery essential. The sim-to-real transfer is achieved through extensive domain randomization and a simulation fidelity pipeline that calibrates key dynamics parameters against real-world data.

The framework demonstrates robust push recovery on the physical exoskeleton, handling perturbations that would topple the robot without reactive stepping. The learned policy outperforms a hand-tuned capture point controller on the same hardware, particularly for lateral pushes and pushes applied during the swing phase.

## Core Contributions
- **Unified stepping decision policy** that simultaneously decides whether to step, where to place the foot, and when to initiate stepping, all from a single RL policy
- **Successful sim-to-real transfer** of reactive stepping on the Atalante exoskeleton, one of the first demonstrations of RL-based push recovery on bipedal hardware
- **Ankle-free bipedal balance** — demonstrates reactive stepping on a robot without ankle actuation, relying entirely on hip and knee control for foot placement
- **Simulation calibration pipeline** that tunes dynamics parameters (friction, damping, actuator models) against real-world experimental data for improved sim-to-real transfer
- **Comprehensive domain randomization** covering body mass (±20%), joint friction (±50%), ground friction (±30%), actuator delay (0-20ms), and observation noise
- **Comparison with capture point controller** demonstrating RL policy superiority, especially for lateral disturbances and swing-phase perturbations

## Methodology Deep-Dive
The RL framework uses Proximal Policy Optimization (PPO) with a custom observation space and action space designed for reactive stepping. The observation includes: (1) body CoM position and velocity (6D), (2) body orientation as quaternion (4D), (3) angular velocity (3D), (4) all joint positions and velocities (24D for Atalante's 12 DoF per leg), (5) foot contact forces (6D, 3D per foot), (6) estimated external disturbance (6D from residual-based observer), and (7) gait clock signal (2D, sin/cos of gait phase). The total observation dimension is 51.

The action space is designed hierarchically within a single output vector: (1) step trigger probability (1D, sigmoid-activated), (2) desired foot placement offset from nominal (2D, forward and lateral), (3) step height (1D, for obstacle clearance), and (4) joint position offsets (12D, modifying the nominal trajectory during recovery). The step trigger is thresholded during deployment: values above 0.5 initiate a stepping response, while values below maintain the current stance. During training, the trigger is treated as a continuous variable to maintain gradient flow.

The reward function emphasizes safety and stability: R = 10·r_alive + 5·r_upright + 2·r_velocity_track + 1·r_energy - 5·r_fall. The heavy alive bonus (10 per timestep) drives the policy toward survival above all else. The upright reward uses the cosine of body tilt angle, providing smooth gradients for balance improvement. The velocity tracking reward is only active during walking (not during standing recovery). The energy penalty encourages efficient recovery motions. The fall penalty is a large negative terminal reward.

The simulation calibration pipeline is a critical component for sim-to-real success. Real-world experiments measure the Atalante's step response (applying known torques and measuring joint trajectories), ground reaction forces during standing, and passive dynamics (letting the robot fall from small tilts). These measurements are used to optimize simulation parameters through Bayesian optimization: joint friction coefficients, motor torque constants, ground contact stiffness/damping, and body inertia parameters. The calibrated simulation reduces the sim-to-real gap from ~15% trajectory tracking error to ~5%.

Domain randomization is applied on top of the calibrated simulation to build robustness margins. Each training episode randomizes: body mass (±20%), segment lengths (±5%), joint damping (±50%), ground friction (0.4-1.0), actuator response delay (0-20ms), and observation noise (Gaussian, σ calibrated from real sensor data). This randomization ensures the policy works across the range of modeling uncertainties, not just the calibrated nominal model.

Training uses PPO with 4096 parallel environments in Isaac Gym. The curriculum starts with standing recovery under small pushes (20N), progresses to larger pushes (up to 150N), then adds walking recovery with pushes during various gait phases. Each curriculum stage requires maintaining >90% survival rate before advancing. Total training takes approximately 10 hours on a single A100 GPU.

## Key Results & Numbers
- **Real-world push recovery**: Atalante exoskeleton recovers from pushes up to **80N for 0.3s** (24 N·s impulse) in standing, and **50N for 0.3s** in walking
- **Sim-to-real gap**: <5% trajectory tracking error after simulation calibration, enabling reliable policy transfer
- **RL vs. CP controller**: RL policy recovers from **30% larger lateral pushes** and **20% larger frontal pushes** than the hand-tuned capture point controller on the same hardware
- **Swing phase recovery**: RL policy successfully recovers from pushes during **85%** of swing phase instances, compared to 50% for CP controller (which struggles with in-swing perturbations)
- **Step initiation latency**: RL policy triggers stepping within **80ms** of disturbance onset (estimated from policy observation update), compared to 150ms for the CP controller's explicit detection logic
- **Training efficiency**: converges in **~10 hours** on a single A100 with 4096 parallel environments
- **Policy inference**: **<0.5ms** on onboard ARM Cortex-A72, well within the 5ms control loop budget
- **Domain randomization ablation**: removing randomization increases sim-to-real failure rate from 5% to 35%

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
This paper focuses on bipedal/exoskeleton push recovery, which is architecturally different from quadruped locomotion. However, the sim-to-real transfer methodology — simulation calibration pipeline combined with domain randomization — is broadly applicable to the Mini Cheetah project. The calibration approach of using real-world measurements to tune simulation parameters could improve Mini Cheetah's sim-to-real transfer. The PPO training with curriculum over disturbance magnitude aligns with the project's methodology.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critical to Project B as it demonstrates real-world RL-based reactive stepping on bipedal hardware, directly validating the approach planned for Cassie's push recovery. The unified stepping decision policy (when/where/how to step) provides a template for Cassie's Safety level reactive stepping primitive. The ankle-free balance demonstrated on Atalante is relevant to Cassie, which also has limited ankle actuation compared to humanoids.

The simulation calibration pipeline is directly applicable to Cassie's sim-to-real transfer. Cassie's dynamics model can be calibrated using the same Bayesian optimization approach against real-world joint response measurements. The domain randomization scheme (mass, friction, delay, noise) provides validated ranges for Cassie's training. The demonstrated 30% improvement over hand-tuned CP control motivates the RL-based approach for Cassie's safety controller over traditional model-based methods. The 80ms step initiation latency provides a benchmark for Cassie's reactive stepping response time.

## What to Borrow / Implement
- **Unified stepping decision policy** — implement a single policy that outputs step trigger, placement, and timing for Cassie's reactive stepping primitive in the Safety level
- **Simulation calibration pipeline** — calibrate Cassie's MuJoCo model against real-world measurements using Bayesian optimization of dynamics parameters
- **Step trigger as continuous variable** — treat the stepping decision as a continuous sigmoid output during training for gradient flow, with thresholding during deployment
- **Gait-phase-aware push recovery** — train push recovery that is conditioned on gait phase, enabling different recovery strategies during stance vs. swing
- **Domain randomization ranges** — use validated randomization ranges (mass ±20%, friction ±30%, delay 0-20ms) as starting points for Cassie's training

## Limitations & Open Questions
- **Exoskeleton-specific constraints** — Atalante's dynamics (carrying human patient, specific joint limits) differ significantly from Cassie; the policy architecture transfers but the specific parameters need retraining
- **Standing and walking only** — no running or dynamic locomotion recovery demonstrated; Cassie's higher-speed gaits introduce additional recovery challenges
- **Limited terrain variation** — all experiments on flat ground; push recovery on uneven terrain (stairs, slopes) is not addressed, which is essential for Cassie's deployment
- **No formal safety guarantees** — the RL policy provides statistical reliability but no hard safety certificates; integration with CBF-QP could provide both learning-based performance and formal safety bounds
