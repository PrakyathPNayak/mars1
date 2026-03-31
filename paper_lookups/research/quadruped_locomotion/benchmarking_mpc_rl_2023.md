# Benchmarking Model Predictive Control and Reinforcement Learning-Based Controllers for Legged Robot Locomotion

**Authors:** Akki et al.
**Year:** 2023 | **Venue:** IEEE / Michigan Tech
**Links:** https://digitalcommons.mtu.edu/etdr/1677/

---

## Abstract Summary
This paper provides a comprehensive benchmarking comparison between Model Predictive Control (MPC) and Reinforcement Learning (RL) based controllers for legged robot locomotion, evaluated on the Unitree Go1 quadruped. The study systematically compares both control paradigms across multiple performance dimensions: disturbance rejection, energy efficiency, torque distribution, terrain generalization, training and deployment computational costs, and gait quality. The results reveal nuanced trade-offs rather than a clear winner, with each approach excelling in different aspects.

The RL controller is trained using PPO in simulation with domain randomization and deployed zero-shot on the Go1 hardware. The MPC controller uses a convex optimization approach that plans center-of-mass trajectories and foot placements over a receding horizon, solving a quadratic program at each control step. Both controllers are evaluated on identical hardware under identical test conditions, providing a fair comparison that controls for platform-specific effects.

The key finding is that RL excels at force rejection (recovering from pushes and external disturbances) but concentrates actuation effort in fewer joints, leading to potential wear and overheating. MPC provides more evenly distributed torques across all joints and better prompt recovery after moderate disturbances, but struggles with large impulse forces. RL generalization suffers significantly on slippery and uneven terrains not encountered during training, while MPC's model-based planning provides more graceful degradation on novel terrains. These results inform the design of hybrid controllers that combine the strengths of both paradigms.

## Core Contributions
- **Fair head-to-head comparison** of MPC and RL on identical hardware (Unitree Go1) under controlled test conditions
- **Multi-dimensional benchmarking** across disturbance rejection, torque distribution, energy efficiency, terrain generalization, and computational cost
- **Quantitative analysis of RL torque concentration:** identifying that RL policies overuse hip joints while underutilizing knee joints
- **Terrain generalization analysis:** systematic evaluation on slippery, uneven, and inclined surfaces showing RL's out-of-distribution degradation
- **Computational cost comparison:** training-time vs. deployment-time trade-offs between offline RL training and online MPC optimization
- **Insights for hybrid design:** specific recommendations for combining MPC and RL in hierarchical architectures

## Methodology Deep-Dive
The RL controller is trained using Proximal Policy Optimization (PPO) in NVIDIA Isaac Gym with 2048 parallel environments. The policy network is a 3-layer MLP (256-256-128) that maps observations (base velocity, orientation, joint states, previous actions, velocity commands) to 12 joint position targets. Training uses the standard Legged Gym reward function with velocity tracking, energy penalty, smoothness reward, and terrain curriculum. Domain randomization covers friction (0.5-1.5), mass (plus or minus 10%), motor strength (0.8-1.2), and observation noise. Training completes in approximately 45 minutes on a single RTX 3090, after which the policy is deployed directly on the Go1's onboard Jetson Nano.

The MPC controller implements a convex MPC formulation based on the MIT Cheetah 3 controller architecture. The robot is modeled as a single rigid body with point-foot contacts. The optimization problem minimizes a quadratic cost: J = sum over horizon of (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k, where x_k is the state (CoM position, orientation, velocities), x_ref is the reference trajectory, u_k are the ground reaction forces, Q and R are weight matrices. Constraints include friction cone constraints (f_x^2 + f_y^2 <= mu^2 * f_z^2), force limits (0 <= f_z <= f_max), and contact schedule constraints (no force during swing phase). The QP is solved using OSQP at 30 Hz, with a prediction horizon of 10 steps (0.33 seconds). The resulting ground reaction forces are converted to joint torques via the Jacobian transpose and added to a PD tracking controller.

The benchmarking protocol consists of six test scenarios, each repeated 20 times for statistical significance:

1. **Forward velocity tracking:** Command 0.5 m/s forward, measure steady-state tracking error, settling time, and velocity variance. Both controllers are tuned to the same target speed.

2. **Lateral push rejection:** Apply a 40N lateral impulse for 0.1 seconds at the robot's center of mass. Measure maximum lateral deviation, recovery time (time to return within 5cm of original trajectory), and whether the robot falls.

3. **Sustained force rejection:** Apply a continuous 15N lateral force. Measure steady-state lateral deviation and torque distribution across joints.

4. **Slippery terrain:** Reduce ground friction to 0.2 (simulating ice). Measure walking stability, slip frequency, and fall rate over 30-second trials.

5. **Uneven terrain:** Deploy on terrain with 3cm random height variations. Measure velocity tracking, stumble frequency, and torque spikes.

6. **Inclined surface:** Walking up a 15-degree slope. Measure forward velocity maintenance, energy consumption, and joint torque distribution.

All tests are performed both in simulation (for controlled comparison) and on real Go1 hardware (for deployment validation). Each test records: joint positions and velocities at 500 Hz, joint torques (from motor current), base IMU data at 400 Hz, foot contact forces (from strain gauges), and external camera tracking for ground truth position.

The torque distribution analysis examines the RMS torque of each joint during steady-state walking. For each joint, the mean and standard deviation of torque across the 12 joints is computed, along with a Gini coefficient measuring torque inequality (0 = perfectly equal distribution, 1 = all torque in one joint).

## Key Results & Numbers
- **Velocity tracking:** RL achieves 3.2% steady-state error vs. MPC's 5.1% error at 0.5 m/s forward velocity
- **Lateral push (40N):** RL recovers in 0.8s with 12cm max deviation; MPC recovers in 1.1s with 18cm max deviation; RL fall rate 5% vs. MPC 15%
- **Sustained force (15N):** RL maintains tighter tracking (4.2cm steady-state offset vs. MPC's 6.8cm)
- **Torque Gini coefficient:** RL has Gini of 0.42 vs. MPC's 0.18, confirming RL concentrates torque in fewer joints (primarily hip flexion)
- **Peak joint torque:** RL's peak hip torque is 2.3x higher than MPC's, while knee torques are 0.6x MPC's
- **Slippery terrain (mu=0.2):** RL fall rate 45% vs. MPC fall rate 20%; MPC adapts more gracefully to reduced friction
- **Uneven terrain:** RL stumble rate 30% vs. MPC 18%; MPC's model-based planning anticipates ground contact timing better
- **Incline (15 deg):** MPC energy consumption 15% lower than RL due to more efficient force distribution
- **Computational cost:** RL policy inference takes 0.2ms (runs easily on Jetson Nano at 50 Hz); MPC QP solve takes 8-12ms (requires aggressive optimization for real-time performance at 30 Hz)
- **Training cost:** RL requires 45 min offline training on GPU + domain randomization; MPC requires manual model identification and parameter tuning (estimated 10-20 hours of engineering time)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This benchmarking study directly informs controller design choices for Mini Cheetah. The finding that RL excels at disturbance rejection but concentrates torque has immediate implications: Mini Cheetah's proprietary actuators have specific thermal limits, and an RL policy that overuses hip joints could cause overheating during extended operation. The study suggests incorporating torque distribution penalties in the PPO reward function (penalize high Gini coefficient) to encourage more balanced actuation.

The terrain generalization findings are critical for Mini Cheetah deployment. The study shows that RL policies trained with standard domain randomization still struggle on out-of-distribution terrains (slippery surfaces, extreme roughness). This motivates more aggressive domain randomization ranges and an explicit curriculum that includes low-friction terrains during training. Alternatively, a hybrid MPC/RL approach could use MPC for the footstep planning and RL for the body stabilization, combining MPC's terrain generalization with RL's disturbance rejection. The MIT Mini Cheetah already has an MPC controller, so the hybrid approach has a natural starting point.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
Understanding MPC vs. RL trade-offs is relevant to Cassie's hierarchical design where different levels may use different control paradigms. Specifically, the finding that MPC provides better torque distribution suggests that the Safety level (LCBF - Learned Control Barrier Functions) could incorporate MPC-like optimization for joint torque limits, while the Controller level uses RL for adaptive behavior. The hierarchical architecture naturally supports this: higher levels can use planning-based methods while lower levels use reactive RL.

The torque concentration finding is relevant to Cassie, which has specific actuator thermal limits on its hip and knee motors. The reward function for Cassie's Controller level should include torque distribution terms informed by this study. The terrain generalization gap for RL motivates the Adversarial Curriculum component in Project B: by adversarially generating challenging terrain conditions during training, the RL policies should be more robust to out-of-distribution deployment scenarios than the standard domain randomization approach evaluated in this benchmarking study.

## What to Borrow / Implement
- **Torque distribution reward term:** add a Gini coefficient penalty or max-joint-torque penalty to the PPO reward function to prevent unhealthy torque concentration
- **Multi-dimensional benchmarking protocol:** adopt the 6-test evaluation suite (velocity tracking, push rejection, sustained force, slippery, uneven, incline) as the standard evaluation for trained policies
- **Terrain generalization testing:** explicitly evaluate trained policies on low-friction and high-roughness terrains not in the training distribution to quantify robustness
- **Hybrid MPC/RL architecture consideration:** explore using MPC for footstep planning with RL for body stabilization, leveraging each method's strengths
- **Computational budget analysis:** use the inference time measurements (0.2ms for RL vs. 8-12ms for MPC) to inform real-time control design on Mini Cheetah and Cassie hardware

## Limitations & Open Questions
- **Single platform:** all results are on Unitree Go1; generalization to other platforms (Mini Cheetah, Cassie, ANYmal) with different actuator characteristics is uncertain
- **Standard RL training only:** the RL controller uses basic Legged Gym training without advanced techniques (RMA adaptation, privileged training, adversarial curriculum); a more sophisticated RL approach might close the terrain generalization gap
- **MPC model limitations:** the single rigid body MPC model is a simplification; full-body MPC or centroidal dynamics MPC might perform differently
- **No hybrid evaluation:** the study compares pure MPC vs. pure RL but does not evaluate hybrid approaches that combine both, which is arguably the most practical design for real-world deployment
