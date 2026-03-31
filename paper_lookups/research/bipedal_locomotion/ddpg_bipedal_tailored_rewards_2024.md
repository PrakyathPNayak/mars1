# Improving Bipedal Robot Motion via Reinforcement Learning and Tailored Reward Functions

**Authors:** (Engineering Science, 2024)
**Year:** 2024 | **Venue:** Engineering Science
**Links:** https://www.espublisher.com/uploads/article_pdf/es1287.pdf

---

## Abstract Summary
This paper presents a comprehensive study on designing tailored reward functions for reinforcement learning-based bipedal locomotion. Using DDPG (Deep Deterministic Policy Gradient) as the base algorithm, the authors systematically investigate how different reward function components affect the quality, stability, and naturalness of bipedal walking gaits. The reward function incorporates terms for body orientation maintenance, Zero Moment Point (ZMP) stability, ground reaction force distribution, step regularity, and energy efficiency, each with carefully tuned weights.

The key finding is that generic locomotion reward functions (e.g., "maximize forward velocity, minimize energy") produce functional but unnatural gaits for bipedal robots, often exploiting simulation artifacts or adopting energy-efficient but unstable strategies. By incorporating biomechanically-motivated reward terms that specifically encourage human-like gait characteristics, the resulting policies exhibit significantly improved stability, naturalness, and robustness. The tailored rewards act as implicit constraints that guide the policy toward physically plausible solutions without explicitly enforcing them through hard constraints.

The study evaluates the tailored reward approach across multiple simulated bipedal robot morphologies with varying leg lengths, masses, and joint configurations, demonstrating that the reward design principles generalize across platforms. Comparison with standard reward functions shows measurable improvements in gait quality metrics including step symmetry, CoM trajectory smoothness, and disturbance recovery time.

## Core Contributions
- **Systematic reward engineering** for bipedal locomotion with biomechanically-motivated components targeting specific gait quality aspects
- **ZMP-based stability reward** that encourages the center of pressure to remain within the support polygon, improving dynamic balance
- **Body orientation reward terms** that penalize excessive roll, pitch, and yaw deviations from upright, promoting stable torso posture
- **Step regularity rewards** that encourage consistent step length, width, and timing, producing more human-like and predictable gaits
- **Cross-morphology evaluation** demonstrating reward design principles generalize across different bipedal robot configurations
- **Comprehensive ablation study** quantifying the contribution of each reward component to overall gait quality
- **DDPG-based continuous control** for bipedal locomotion with detailed hyperparameter analysis

## Methodology Deep-Dive
The DDPG algorithm is used with an actor-critic architecture. The actor network takes the robot's state (body position, orientation as quaternion, linear and angular velocities, 6 joint angles and velocities per leg, foot contact forces) and outputs continuous torque commands for each actuated joint. The critic network estimates the Q-value of state-action pairs. Both networks use 3-layer MLPs with 256 hidden units and ReLU activations. The replay buffer stores 1M transitions, with batch size 256 and soft target update parameter tau of 0.005.

The tailored reward function R_total is decomposed as: R_total = w_vel * R_velocity + w_orient * R_orientation + w_zmp * R_ZMP + w_grf * R_GRF + w_step * R_step + w_energy * R_energy + w_alive * R_alive. Each component is designed as follows:

R_velocity rewards forward velocity tracking: R_vel = exp(-alpha * (v_x - v_target)^2) where alpha controls the sharpness. This exponential form provides smooth gradients near the target while strongly penalizing large deviations. R_orientation penalizes body tilt: R_orient = -beta * (roll^2 + pitch^2 + gamma * yaw_rate^2), where roll and pitch are measured from the upright reference and yaw_rate deviation from commanded turning. The weights beta and gamma are tuned so that the orientation penalty is significant but does not dominate the forward velocity reward.

R_ZMP is the core stability reward: R_ZMP = exp(-delta * d_ZMP^2), where d_ZMP is the Euclidean distance from the Zero Moment Point to the center of the support polygon. During single support, the support polygon is the stance foot; during double support, it is the convex hull of both feet. This reward directly encourages the policy to maintain dynamic balance by keeping the effective point of ground reaction force within the stable region. The ZMP is computed from: x_ZMP = (sum of (p_i * f_iz - tau_iy)) / (sum of f_iz), where p_i are contact point positions, f_iz are vertical forces, and tau_iy are moment contributions.

R_GRF (Ground Reaction Force) encourages smooth, biologically plausible force profiles: R_GRF = -epsilon * sum(|df_i/dt|^2), penalizing rapid changes in ground contact forces. This discourages impulsive contacts (stomping) and encourages smooth heel-strike to toe-off transitions. R_step rewards step regularity: R_step = exp(-zeta * ((L_step - L_target)^2 + (W_step - W_target)^2 + (T_step - T_target)^2)), where L, W, T are step length, width, and period respectively, with target values derived from biomechanics literature for the robot's leg length.

R_energy penalizes excessive actuator effort: R_energy = -eta * sum(|tau_i * omega_i|), summing the absolute mechanical power across all joints. R_alive provides a constant positive reward for remaining upright, encouraging the policy to prioritize balance: R_alive = +1.0 per timestep if the base height is above a threshold and the orientation is within 45 degrees of upright.

The training procedure uses curriculum learning over velocity targets. Training begins with a target velocity of 0.0 m/s (standing balance), progresses to 0.1 m/s (slow walking), and incrementally increases to the target speed (0.5-1.0 m/s). Each curriculum stage trains for 200K timesteps before advancing. The curriculum ensures the policy learns stable balance before attempting locomotion, which is critical for bipedal robots where walking requires controlled falling.

## Key Results & Numbers
- **Gait naturalness:** Tailored rewards produce gaits rated 4.2/5.0 by human evaluators for naturalness, compared to 2.1/5.0 for standard reward functions
- **ZMP stability:** Mean ZMP distance from support polygon center reduced by 42% compared to baseline (from 0.08m to 0.046m)
- **Step symmetry:** Left-right step length difference reduced from 15% (standard rewards) to 3.2% (tailored rewards)
- **Disturbance recovery:** Time to recover from 15N lateral push reduced from 2.1s to 1.4s with tailored rewards
- **Energy efficiency:** Cost of Transport improved by 18% due to smoother force profiles and reduced unnecessary joint oscillations
- **Cross-morphology:** Reward weights transfer across 3 bipedal morphologies with less than 10% performance variation, requiring only minor re-tuning
- **Training stability:** Tailored rewards reduce training variance by 35% (measured across 10 random seeds), producing more consistent learning curves
- **Ablation highlights:** Removing ZMP reward degrades stability by 28%; removing GRF smoothness reward increases energy consumption by 22%; removing step regularity reduces gait predictability but has minimal impact on forward speed

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
This paper focuses specifically on bipedal locomotion reward design, which has limited direct applicability to the quadrupedal Mini Cheetah. Quadrupeds have fundamentally different stability characteristics (four-point base of support, static stability during walking) that make bipedal-specific reward terms like ZMP and single-support phase rewards less relevant. However, the general methodology of systematic reward engineering with ablation studies provides a template for Mini Cheetah reward function design. The GRF smoothness reward and energy efficiency terms are applicable to any legged robot.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The reward engineering insights are directly applicable to Cassie's Controller-level policy training. The ZMP stability reward is particularly critical for Cassie, which has small feet and a high center of mass, making dynamic balance challenging. The specific reward formulations can be implemented in Cassie's training: R_ZMP for encouraging stable weight transfer during walking, R_GRF for smooth contact transitions (important for Cassie's compliant leg design), R_step for consistent gait patterns, and R_orient for torso stability.

The curriculum learning over velocity targets (standing to slow walking to normal speed) is directly applicable to Cassie's training schedule. Cassie's initial policy training should focus on standing balance before progressing to walking, matching this paper's approach. The body orientation reward is especially important for Cassie, which tends to develop excessive lateral sway when trained with naive reward functions. The DDPG algorithm used here could be compared with PPO in Project B's pipeline, though PPO generally provides more stable training for high-dimensional locomotion.

## What to Borrow / Implement
- **ZMP stability reward term** for Cassie's Controller level: R_ZMP = exp(-delta * d_ZMP^2) with delta calibrated for Cassie's foot geometry and mass distribution
- **GRF smoothness reward** to discourage impulsive contacts and encourage smooth gait transitions, particularly important for Cassie's leaf spring legs
- **Velocity curriculum schedule:** train standing balance first (v_target=0), then progress through 0.1, 0.2, ..., 0.8 m/s incrementally
- **Step regularity rewards** with biomechanically-derived target values for Cassie's leg length and mass
- **Systematic ablation methodology:** evaluate each reward component's contribution independently to understand the reward landscape

## Limitations & Open Questions
- **DDPG limitations:** DDPG is known to be brittle in high-dimensional continuous control; PPO or SAC may produce more reliable results for the same reward function
- **Simulation-only evaluation:** no real-world validation; the reward terms may encourage behaviors that exploit simulation inaccuracies (e.g., ZMP computation assumes perfect ground contact modeling)
- **Fixed target values:** biomechanically-derived target step lengths and widths may not be optimal for robot morphologies that differ significantly from human proportions (e.g., Cassie's digitigrade leg structure)
- **No terrain variation:** all experiments are on flat ground; the tailored rewards may need additional terms or re-weighting for uneven terrain, slopes, and stairs
