# Learning Agile Locomotion and Adaptive Behaviors via RL-augmented MPC

**Authors:** (ICRA 2024 authors)
**Year:** 2024 | **Venue:** ICRA
**Links:** DOI: 10.1109/icra57147.2024.10610453

---

## Abstract Summary
This paper presents an RL-augmented MPC framework that integrates reinforcement learning as a plug-in module for model-predictive control to achieve agile and adaptive quadruped locomotion. Rather than replacing MPC entirely with end-to-end RL, the system uses RL to learn residual corrections and adaptive parameters that enhance the MPC's capabilities. This enables rapid velocities, tight turns, robust blind stair climbing, and heavy load carrying while maintaining the interpretability and safety guarantees of the underlying MPC framework.

A key contribution is the demonstration of zero-shot cross-platform transfer: the RL module trained on one robot platform can be applied to different quadrupeds without retraining. This is possible because the RL module operates in a platform-agnostic representation space (desired velocity, terrain estimate, gait phase) and outputs corrections to the MPC's reference trajectory rather than raw joint commands. The MPC then handles the platform-specific dynamics and constraints.

The system achieves several impressive real-world demonstrations including blind stair climbing (without any exteroceptive sensing), robustness to heavy loads (up to 50% body weight), rapid velocity tracking up to 3.5 m/s, and agile turning maneuvers. These capabilities emerge from the RL module's ability to learn terrain-adaptive gait timing, foot placement adjustments, and body posture corrections that the nominal MPC cannot achieve alone.

## Core Contributions
- RL as a generalizable plug-in module for MPC that outputs residual corrections to reference trajectories
- Zero-shot cross-platform transfer by operating in platform-agnostic representation space
- Blind stair climbing without exteroceptive sensors through learned proprioceptive adaptation
- Robustness to heavy loads (up to 50% body weight) through learned posture and gait adjustments
- Rapid velocity tracking up to 3.5 m/s with stable gaits
- Agile turning maneuvers with tight turning radii
- Maintained interpretability and constraint satisfaction from the MPC framework

## Methodology Deep-Dive
The architecture separates locomotion control into a nominal MPC layer and a learned RL augmentation layer. The nominal MPC follows a standard convex MPC formulation for quadruped locomotion: it takes a desired velocity command and gait schedule as input, optimizes ground reaction forces over a prediction horizon using a single rigid body dynamics model, and outputs foot force references. A whole-body controller then converts these force references to joint torques using the robot's full dynamics model and QP-based torque optimization.

The RL augmentation module is trained to output corrections to the MPC's inputs and parameters. Specifically, the RL module outputs: (1) residual adjustments to the desired CoM velocity and body orientation, (2) modifications to the gait timing parameters (stance/swing durations, phase offsets), (3) additive corrections to the foot placement targets, and (4) adjustments to the MPC's cost function weights. These corrections allow the system to adapt to conditions the nominal MPC cannot handle, such as unseen terrain or external loads.

The RL module is a multi-layer perceptron trained with PPO in simulation. Its observation space includes proprioceptive measurements (joint positions, velocities, body orientation, angular velocity), a short history of these measurements for temporal context, and the commanded velocity. Crucially, it does not include any exteroceptive information (cameras, lidar, terrain maps), making the entire system blind. The reward function encourages velocity tracking while penalizing energy consumption, contact forces, and body orientation deviations.

The zero-shot transfer capability arises from the observation and action space design. The RL module's observations are normalized and expressed in body-frame coordinates, making them platform-independent. The outputs are expressed as relative corrections to the MPC's reference values, which the MPC then maps to platform-specific controls. This means the same RL module can be paired with different MPC configurations for different robots, with the MPC handling the platform-specific dynamics.

Training uses extensive domain randomization including friction variations (0.3 to 1.5), payload mass (0% to 50% body weight), motor strength scaling (80% to 120%), terrain roughness (flat to 8cm height variation), and simulation parameter perturbations. A curriculum over terrain difficulty is employed, starting with flat ground and progressively introducing stairs, slopes, and rough terrain as the policy improves.

## Key Results & Numbers
- Blind stair climbing: successfully ascends and descends stairs up to 15cm height without any vision
- Zero-shot transfer: RL module trained on Robot A works on Robot B without retraining (tested on 3 platforms)
- Maximum velocity: stable tracking up to 3.5 m/s on flat terrain
- Load carrying: maintains stable locomotion with up to 50% body weight additional load
- Turning: achieves turning rates up to 2.5 rad/s with stable gait
- Real-time performance: RL inference + MPC solve within 2ms at 500Hz control rate
- Energy efficiency: 20% lower cost of transport compared to pure RL at moderate speeds

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The zero-shot cross-platform transfer and blind stair climbing capabilities are directly applicable to Mini Cheetah deployment. The RL-augmented MPC architecture provides an alternative to pure end-to-end RL that maintains interpretability while achieving agile locomotion. The blind stair climbing result is particularly relevant since Mini Cheetah may need to navigate stairs without exteroceptive sensors in early deployment stages.

The domain randomization strategy (friction, payload, motor strength, terrain) provides a concrete reference for Mini Cheetah's randomization ranges. The residual RL architecture (corrections to MPC references) is a proven approach that could serve as a comparison baseline for the pure RL approach, and the MPC cost function design informs reward function engineering for the pure RL policy.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The hybrid MPC-RL architecture provides insight for Cassie's hierarchical Planner level, where MPC could generate reference trajectories that RL-based lower levels track. The residual correction concept—where RL outputs adjustments rather than raw commands—is relevant to the interface between the Planner and Primitives levels in Project B's hierarchy. The Planner could use MPC for nominal trajectory generation while an RL module provides terrain-adaptive corrections.

However, the specific locomotion demonstrations are for quadrupeds, and the dynamics of bipedal locomotion (which requires active balance) differ significantly. The blind locomotion approach would need to be combined with Cassie's capture point-based balance control for bipedal applicability.

## What to Borrow / Implement
- Adopt the residual RL architecture concept: instead of end-to-end RL, consider RL that outputs corrections to a nominal controller for more stable training
- Use the domain randomization ranges (friction 0.3-1.5, payload 0-50%, motor strength 80-120%) as starting points for Mini Cheetah training
- Implement the platform-agnostic observation/action space design for potential cross-platform transfer between Mini Cheetah variants
- Borrow the curriculum strategy of progressive terrain difficulty with the specific terrain types tested (flat, slopes, stairs, rough)
- Consider the MPC + RL plug-in architecture for Project B's Planner level to combine formal guarantees with learned adaptation

## Limitations & Open Questions
- The linear MPC model (single rigid body) may not capture the full dynamics of agile maneuvers, limiting peak performance compared to end-to-end RL
- Zero-shot transfer works for similar-scale quadrupeds but generalization to bipeds or significantly different morphologies is unproven
- Blind stair climbing is impressive but limited to moderate stair heights; vision integration would extend capability
- The interaction between RL corrections and MPC constraints could lead to infeasible optimization problems in edge cases
