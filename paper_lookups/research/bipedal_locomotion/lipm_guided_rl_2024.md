# LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion of Bipedal Robots

**Authors:** (arXiv 2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2509.09106

---

## Abstract Summary
This paper proposes a novel reward design framework for bipedal robot locomotion that integrates principles from the Linear Inverted Pendulum Model (LIPM) into reinforcement learning training. Traditional RL reward functions for locomotion rely on hand-crafted terms that encourage velocity tracking, energy efficiency, and smoothness but lack direct grounding in stability theory. This work bridges the gap by deriving reward terms directly from LIPM stability conditions, encouraging the RL agent to maintain stable center-of-mass (CoM) trajectories and appropriate torso orientations throughout the gait cycle.

The LIPM is a simplified dynamics model widely used in bipedal robotics that represents the robot as a point mass on a massless leg, with the constraint that the CoM moves at a constant height. From this model, key stability metrics can be derived: the orbital energy, the capture point, and the divergent component of motion (DCM). The paper formulates reward terms that penalize deviations from LIPM-predicted CoM trajectories, encourage foot placement at or near the capture point, and maintain the DCM within a stable region.

The resulting RL policies demonstrate improved balance stability, more human-like gait characteristics, and better robustness to perturbations compared to standard reward designs. The policies maintain more consistent CoM heights, exhibit smoother weight transfer between feet, and recover more gracefully from pushes. The approach is validated on bipedal robot models in MuJoCo simulation, showing that physics-informed reward design leads to fundamentally better locomotion policies.

## Core Contributions
- Novel LIPM-derived reward terms that ground RL training in bipedal stability theory
- Orbital energy reward that encourages the robot to maintain energy-consistent periodic gaits
- Capture point reward that guides foot placement for balance recovery
- DCM stability reward that penalizes divergent center-of-mass motion
- Improved CoM height consistency and smoother gait compared to standard RL rewards
- More human-like gait characteristics including natural weight transfer and heel-to-toe progression
- Better push recovery performance due to stability-grounded policy training

## Methodology Deep-Dive
The LIPM describes the dynamics of a bipedal robot's CoM as: x_ddot = omega^2 * (x - p), where x is the CoM position, p is the center of pressure (CoP) location, and omega = sqrt(g/z_0) is the natural frequency determined by gravity g and the nominal CoM height z_0. This simple model captures the fundamental instability of bipedal walking—the CoM naturally diverges from the CoP, requiring active foot placement for balance.

From the LIPM, the authors derive three key reward components. First, the orbital energy reward: E_orb = 0.5 * x_dot^2 - 0.5 * omega^2 * (x - p)^2. For periodic walking, E_orb should be constant throughout the gait cycle. The reward penalizes deviations from a target orbital energy: r_orb = -alpha_orb * |E_orb - E_target|. This encourages the policy to maintain consistent gait energy, producing smoother and more periodic locomotion.

Second, the capture point reward: xi = x + x_dot / omega. The capture point is the location where the robot would need to place its foot to come to a complete stop. For stable walking, the swing foot should be placed near the capture point with a small offset in the walking direction. The reward is: r_cp = -alpha_cp * |p_foot - (xi + delta_walk)|, where p_foot is the swing foot target and delta_walk is a walking-direction offset. This directly encourages foot placement that maintains balance.

Third, the DCM stability reward: xi_dot = omega * (xi - p). The DCM should be bounded and cyclic during stable walking. The reward penalizes DCM divergence: r_dcm = -alpha_dcm * |xi_dot| when |xi_dot| exceeds a threshold. This prevents the policy from entering states where the CoM motion becomes irrecoverably divergent.

These LIPM-derived rewards are combined with standard locomotion rewards (velocity tracking, energy minimization, joint limit avoidance, contact schedule adherence) in a weighted sum. The total reward is: r_total = r_track + r_energy + r_orb + r_cp + r_dcm + r_style, where r_style includes additional terms for torso orientation and gait symmetry. The weights are tuned through ablation studies.

Training uses PPO with a standard MLP policy network. The observation space includes joint positions and velocities, body orientation (from IMU), angular velocity, and a gait phase clock signal. The policy outputs target joint positions that are tracked by a PD controller. Domain randomization includes friction (0.4-1.2), body mass (plus/minus 15%), CoM position offset (plus/minus 2cm), and motor strength (85-115%).

## Key Results & Numbers
- 35% reduction in CoM height variance compared to standard RL reward design
- 25% improvement in push recovery success rate (lateral pushes up to 80N)
- More periodic gait with 40% lower gait cycle variance
- Velocity tracking RMSE comparable to standard rewards (no sacrifice in tracking performance)
- 15% lower energy consumption due to more efficient gait dynamics
- Gait characteristics closer to human walking patterns as measured by kinematic similarity metrics
- Successful training convergence with standard PPO, no specialized algorithm required

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
The LIPM is specific to bipedal locomotion dynamics and does not directly apply to quadruped robots like the Mini Cheetah, which have a fundamentally different balance paradigm (static stability vs. dynamic balance). However, the broader principle of physics-informed reward design transfers: using simplified dynamics models to derive reward terms that encode desired physical properties. For quadrupeds, analogous reward terms could be derived from the Zero-Moment Point (ZMP) or support polygon stability criteria.

The reward engineering methodology—deriving terms from first-principles physics models rather than purely hand-crafting them—is valuable for any locomotion RL project. The ablation study approach for tuning reward weights is also applicable to Mini Cheetah reward design.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
LIPM-guided rewards are directly applicable to Cassie's bipedal locomotion training. The Cassie robot's Controller level in the 4-level hierarchy is responsible for balance and gait execution, which is exactly what LIPM-derived rewards target. The orbital energy reward would encourage Cassie to maintain consistent gait dynamics, the capture point reward would improve foot placement for balance (directly complementing the Differentiable Capture Point module in CPTE), and the DCM reward would prevent divergent CoM motion.

The capture point reward is particularly synergistic with Project B's architecture: while the CPTE module uses capture point theory for trajectory estimation, the LIPM-guided reward would ensure the RL policy at the Controller level is intrinsically motivated to respect capture point-based balance. This creates a consistent stability framework across both the reward design and the architectural components. The Neural ODE Gait Phase component could also benefit from the orbital energy reward, which encourages periodic gait patterns that the gait phase model can better predict.

## What to Borrow / Implement
- Implement the LIPM-derived reward terms (orbital energy, capture point, DCM stability) for Cassie's Controller-level RL training in Project B
- Adapt the physics-informed reward design principle for Mini Cheetah using quadruped-appropriate stability metrics (ZMP, support polygon)
- Use the reward weight tuning methodology (systematic ablation) for both projects
- Integrate the capture point reward with Project B's CPTE module for consistent stability-aware training
- Apply the gait cycle periodicity reward to complement the Neural ODE Gait Phase model in Project B

## Limitations & Open Questions
- LIPM assumes constant CoM height, which is violated during dynamic maneuvers (jumping, crouching, stair climbing)
- The capture point formulation assumes flat ground; extension to 3D terrain is more complex (3D-LIPM or variable-height models)
- Reward weight tuning still requires manual effort despite the physics-informed derivation
- Interaction between LIPM-derived rewards and other reward terms (energy, style) can create conflicting gradients that slow training
