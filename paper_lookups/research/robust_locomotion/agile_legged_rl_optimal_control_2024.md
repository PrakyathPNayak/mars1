# Agile Legged Robots through Reinforcement Learning and Optimal Control

**Authors:** (UW PhD Dissertation 2024)
**Year:** 2024 | **Venue:** University of Washington Digital Library
**Links:** https://digital.lib.washington.edu/researchworks/items/8610fec3-dc43-48a0-bb1a-4e682f9b908a

---

## Abstract Summary
This comprehensive PhD dissertation from the University of Washington presents a unified framework for agile legged robot locomotion that bridges reinforcement learning and optimal control. The work addresses the fundamental challenge of achieving robust, agile locomotion across diverse terrains by combining the generalization capabilities of RL with the precision and stability guarantees of optimal control. The dissertation covers hierarchical control architectures with high-level gait selection and low-level joint control, perception-integrated locomotion policies that use exteroceptive sensing, and reward shaping techniques for holistic locomotion goals.

The dissertation makes contributions across three primary axes: (1) hierarchical policy architectures where a high-level policy selects gaits, foothold targets, and body trajectories while a low-level policy tracks these references with whole-body control, (2) integration of heightmap and depth-camera perception into the locomotion policy for terrain-aware behavior, and (3) systematic reward shaping that encodes locomotion objectives spanning energy efficiency, robustness, agility, and terrain adaptation into a coherent multi-objective optimization. Each axis is validated on simulated quadruped and bipedal platforms with sim-to-real transfer demonstrations.

The work provides extensive experimental evaluation across flat terrain, stairs, rough outdoor terrain, and dynamic obstacles, demonstrating that the proposed hierarchical RL + optimal control framework achieves state-of-the-art performance on both quadruped and bipedal platforms. The dissertation also provides a thorough literature review and comparison of existing approaches, making it a valuable reference for the field.

## Core Contributions
- A hierarchical control architecture combining RL-based high-level gait and foothold planning with model-based optimal control for low-level joint tracking
- Integration of exteroceptive perception (heightmaps from LiDAR, depth images from cameras) into the locomotion policy via a terrain encoder network
- Systematic reward shaping framework covering five objective categories: task performance, energy efficiency, stability, naturalness, and terrain adaptation
- Demonstration of sim-to-real transfer for the hierarchical architecture on both quadruped (A1-like) and bipedal (Cassie-like) platforms
- Comprehensive comparison of end-to-end RL, hierarchical RL, and hybrid RL+optimal control approaches across multiple terrains and tasks
- Analysis of the RL–optimal control spectrum: when to use pure RL, when to use pure optimal control, and when hybrid approaches are optimal
- Techniques for stable training of hierarchical policies, including asynchronous level training, frozen-level curriculum, and reward alignment across levels

## Methodology Deep-Dive
The hierarchical architecture consists of three layers: (1) a high-level policy π_high that observes the robot state, terrain information, and task command and outputs gait type, foothold targets, and desired body trajectory at 10 Hz; (2) a mid-level trajectory optimizer that uses convex model predictive control (MPC) to plan center-of-mass and foot swing trajectories satisfying dynamics constraints at 50 Hz; and (3) a low-level whole-body controller using quadratic programming (QP) that computes joint torques to track the mid-level trajectories at 500 Hz.

The high-level RL policy uses an observation space comprising: proprioceptive state (joint positions 12D, joint velocities 12D, base angular velocity 3D, projected gravity 3D), commanded velocity and heading (4D), gait phase clock (4D), and a terrain encoding from the perception module (32D latent vector). The terrain encoder processes a 20×20 heightmap (centered on the robot, 4m × 4m, 20cm resolution) through a 4-layer CNN producing the 32D latent representation. The policy output includes: gait selection (categorical over walk/trot/pace/gallop), foothold offsets (8D, Δx and Δy for each foot relative to the default foothold), and body velocity modulation (3D, scaling the commanded velocity based on perceived terrain difficulty).

The mid-level convex MPC solves: min Σ ||x_t - x_ref||²_Q + ||f_t||²_R subject to dynamics constraints (single rigid body model), friction cone constraints (|f_xy| ≤ μ·f_z), and kinematic reachability constraints (footholds within workspace). This runs at 50 Hz with a prediction horizon of 0.5 seconds (25 steps). The MPC formulation uses a simplified single-rigid-body dynamics model that captures the essential center-of-mass dynamics while being computationally efficient.

The low-level QP whole-body controller computes joint torques τ that: (1) track the desired ground reaction forces from MPC via the contact Jacobian, (2) track the desired joint trajectories for swing legs, and (3) satisfy torque limits and contact constraints. The QP runs at 500 Hz and includes regularization terms for smooth torque profiles. This is formulated as: min ||J_c^T f - τ_desired||² + w_swing · ||q̈ - q̈_ref||² + w_reg · ||τ||² subject to τ_min ≤ τ ≤ τ_max, f ∈ friction_cone.

The reward function for the high-level policy is structured into five categories: (1) Task: velocity tracking (exp kernel), heading tracking (angular difference penalty); (2) Energy: mechanical power penalty (Σ|τ·ω|), cost of transport tracking (reward for CoT below threshold); (3) Stability: angular velocity penalty, foot slip penalty, support polygon margin reward; (4) Naturalness: gait symmetry reward, stride regularity, smooth gait transitions (penalize rapid gait switching); (5) Terrain: foothold safety reward (reward placing feet on flat surfaces), body clearance reward (reward maintaining height over obstacles). Each category has 2–4 terms with learnable weights that are updated via a population-based meta-optimization every 200 training iterations.

Training follows an asynchronous curriculum: the low-level QP controller is designed first (no learning needed), the mid-level MPC parameters are tuned on flat terrain, and then the high-level RL policy is trained with the mid-level and low-level controllers frozen. After high-level training converges, a fine-tuning phase jointly updates the high-level policy and mid-level MPC parameters with a reduced learning rate. This asynchronous approach avoids the instability of end-to-end training of the full hierarchy.

## Key Results & Numbers
- Flat terrain velocity tracking: 0.03 m/s RMSE for quadruped, 0.05 m/s for bipedal
- Stair climbing success rate: 95% on 15 cm stairs (quadruped), 82% on 12 cm stairs (bipedal)
- Rough terrain survival: 92% over 50m courses with ±8 cm height variation
- Energy efficiency: 22% CoT reduction compared to end-to-end RL baselines due to optimal control precision in the low-level
- Sim-to-real transfer: 85% of simulation performance maintained on real quadruped hardware (A1)
- Perception integration: terrain encoder reduces foothold placement error by 45% compared to blind policies
- Hierarchical vs. end-to-end: hierarchical approach achieves 30% higher success rate on challenging terrains while using 40% fewer training samples
- Training time: ~24 hours total (16 hours high-level RL + 8 hours fine-tuning) on 4×A100 GPUs
- Comparison across 6 baseline methods on 4 terrain types, establishing state-of-the-art on 3/4 terrain types

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This dissertation provides a comprehensive guide for Mini Cheetah's hierarchical RL + optimal control pipeline. The three-layer architecture (high-level RL → mid-level MPC → low-level QP) is directly applicable to Mini Cheetah's 12-DoF control problem. The terrain encoder integration using heightmaps provides a blueprint for adding exteroceptive perception to Mini Cheetah. The systematic five-category reward framework gives a complete template for Mini Cheetah's reward function design.

The sim-to-real transfer results (85% of simulation performance) and the asynchronous training curriculum provide practical guidance for Mini Cheetah's deployment pipeline. The comparison between end-to-end RL and hierarchical approaches, showing 30% higher success rate for the hierarchical approach on challenging terrains, strongly supports using a hierarchical architecture for Mini Cheetah's outdoor deployment.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This dissertation is critically relevant to Cassie's 4-level hierarchical architecture. The three-layer hierarchy (high-level gait selection → mid-level trajectory optimization → low-level joint control) directly mirrors Cassie's Planner→Controller pipeline, and the dissertation's bipedal experiments on Cassie-like platforms provide directly transferable results. The perception integration architecture (terrain encoder → high-level policy) maps to Cassie's Planner level, which must incorporate environmental awareness.

The asynchronous training curriculum (training levels sequentially then fine-tuning jointly) addresses one of the key challenges in Cassie's 4-level hierarchy: how to stably train multiple interacting levels. The reward alignment analysis across hierarchy levels is directly applicable to ensuring Cassie's Planner, Primitives, Controller, and Safety levels have compatible objectives. The QP-based safety controller at the low level parallels Cassie's CBF-QP Safety level, providing implementation guidance.

## What to Borrow / Implement
- Adopt the three-layer hierarchical architecture (RL gait selection → MPC trajectory optimization → QP whole-body control) as a reference design for both Mini Cheetah and Cassie
- Implement the CNN-based terrain encoder for heightmap processing in Mini Cheetah's perception pipeline
- Use the five-category reward framework (task, energy, stability, naturalness, terrain) as a template for reward function design on both platforms
- Apply the asynchronous training curriculum (sequential level training → joint fine-tuning) to Cassie's 4-level hierarchy training
- Borrow the convex MPC formulation with single-rigid-body dynamics for Mini Cheetah's mid-level trajectory optimization

## Limitations & Open Questions
- The hierarchical architecture introduces latency between the high-level decisions (10 Hz) and low-level execution (500 Hz); fast terrain changes or unexpected disturbances may not be handled quickly enough by the high level
- The terrain encoder assumes access to a reliable heightmap, which is challenging in real-world conditions with sensor noise, occlusions, and dynamic obstacles
- The asynchronous training curriculum does not guarantee global optimality of the joint hierarchy; the frozen-level training phases may converge to locally suboptimal solutions that are not recoverable during fine-tuning
- The population-based meta-optimization for reward weights adds significant computational overhead and complexity; simpler reward tuning methods may be sufficient for less complex locomotion objectives
