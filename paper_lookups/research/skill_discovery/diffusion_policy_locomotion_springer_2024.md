# Locomotion Policy Learning via Diffusion Policy

**Authors:** (Springer 2024)
**Year:** 2024 | **Venue:** Springer Lecture Notes
**Links:** [Springer Chapter](https://link.springer.com/chapter/10.1007/978-981-97-8658-9_66)

---

## Abstract Summary
This paper investigates the application of diffusion-based policy learning to legged robot locomotion, demonstrating that diffusion policies produce more robust and natural gaits compared to traditional deep reinforcement learning approaches such as PPO and SAC. The core argument is that the iterative denoising process inherent in diffusion models enables multi-modal action sampling, which captures the richness of natural locomotion behaviors that unimodal Gaussian policies fundamentally cannot represent.

The authors show that diffusion policies handle high-dimensional action spaces more effectively than conventional RL, producing smoother and more energy-efficient joint trajectories. The multi-modal sampling capability allows the policy to explore a richer set of locomotion strategies during training and maintain behavioral diversity at deployment. This is particularly beneficial for adaptive locomotion in diverse environments, where the robot must adjust its gait pattern based on terrain and task demands.

The experimental evaluation covers multiple simulated locomotion tasks with varying complexity, demonstrating consistent improvements in gait quality metrics (smoothness, energy efficiency, ground reaction force patterns) alongside competitive or superior task performance metrics (velocity tracking, stability). The work provides a thorough empirical analysis of why diffusion policies are well-suited for the locomotion domain.

## Core Contributions
- Provides empirical evidence that diffusion policies generate more natural and robust gaits than PPO/SAC baselines across multiple locomotion benchmarks
- Demonstrates that multi-modal action sampling in diffusion models captures diverse locomotion strategies unavailable to unimodal Gaussian policies
- Shows improved performance in high-dimensional action spaces where traditional RL struggles with exploration
- Analyzes gait quality beyond task reward, examining smoothness, energy efficiency, and biomechanical naturalness metrics
- Introduces adaptive locomotion capability where the diffusion policy adjusts behavior based on environmental context without explicit mode switching
- Provides ablations on diffusion model hyperparameters (denoising steps, noise schedule, action horizon) specific to locomotion

## Methodology Deep-Dive
The diffusion policy architecture follows the DDPM framework adapted for continuous robotic control. The state encoder processes proprioceptive observations (joint angles, angular velocities, body pose from IMU, and command velocities) through an MLP to produce a conditioning vector. This vector conditions the denoising network—a 1D temporal U-Net operating over the action sequence dimension—via FiLM (Feature-wise Linear Modulation) layers at each resolution level.

The training procedure uses a behavior cloning objective on demonstration data augmented with online interaction data. Demonstrations are generated from well-trained RL policies and optionally from motion capture reference data. The diffusion training objective minimizes the variational lower bound: at each training step, a random noise level t is sampled, noise is added to the ground-truth action sequence, and the network predicts the noise to be removed. The authors explore both ε-prediction and v-prediction parameterizations, finding v-prediction slightly more stable for locomotion.

A key methodological contribution is the analysis of action horizon length and its effect on locomotion quality. Shorter horizons (4–8 steps) provide more reactive control but can produce jittery gaits, while longer horizons (16–32 steps) produce smoother trajectories but reduce adaptability. The authors propose an adaptive horizon scheme that shortens the prediction window in response to large state deviations (e.g., perturbations), balancing smoothness with reactivity.

For inference efficiency, the paper employs DDIM sampling with classifier-free guidance. The guidance scale controls the trade-off between action diversity and conditioning fidelity—higher guidance produces more deterministic, command-following behavior while lower guidance allows more behavioral exploration. The authors find that a moderate guidance scale (1.5–3.0) works best for locomotion, maintaining natural variability while tracking commands.

The multi-modal capability is evaluated by training on datasets containing multiple gait types and analyzing the distribution of generated actions. The authors use t-SNE visualization of action sequences to demonstrate that the diffusion policy maintains distinct gait clusters in its output distribution, unlike PPO which collapses to a single mode.

## Key Results & Numbers
- 15–25% improvement in gait smoothness metrics (joint jerk reduction) compared to PPO baselines
- 10–18% reduction in energy consumption (cost of transport) across tested gaits
- Competitive or superior velocity tracking accuracy (within 0.05 m/s of PPO) while maintaining gait quality improvements
- Successfully maintains 3–4 distinct gait modes in a single policy versus mode collapse in PPO
- Stable training across high-dimensional action spaces (18+ DoF) where PPO shows increased variance
- Ablation shows 8–16 step action horizon optimal for quadruped locomotion balance of smoothness and reactivity

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This work provides strong empirical motivation for exploring diffusion policies as an alternative or complement to PPO for Mini Cheetah locomotion. The demonstrated improvements in gait naturalness and energy efficiency are directly relevant—Mini Cheetah's 12 DoF action space falls within the regime where diffusion policies show clear advantages. The adaptive horizon scheme could be particularly valuable for handling the diverse terrain scenarios in the curriculum learning pipeline.

The energy efficiency improvements are significant for Mini Cheetah hardware deployment, where battery life and motor thermal limits constrain operation time. The multi-modal gait capability aligns with the project's goal of achieving diverse locomotion behaviors from a single policy, potentially simplifying the deployment pipeline compared to maintaining multiple specialist policies.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
For Cassie's hierarchical architecture, the diffusion policy's multi-modal action generation directly addresses the challenge of generating diverse locomotion primitives. The Primitives level, currently planned with DIAYN/DADS skill discovery, could benefit from diffusion-based action generation that naturally produces diverse behaviors without explicit diversity objectives. The classifier-free guidance mechanism provides a natural interface for the Planner level to modulate primitive behavior through guidance conditioning.

The adaptive horizon scheme is especially relevant for bipedal locomotion, where balance-critical situations require rapid reactive control while steady-state walking benefits from smooth, long-horizon planning. This maps well onto the Controller level's need to balance between tracking primitive targets and maintaining dynamic stability.

## What to Borrow / Implement
- Adopt the adaptive action horizon scheme that shortens prediction windows during perturbations for reactive control
- Use classifier-free guidance as the Planner→Primitives conditioning interface in Cassie's hierarchy
- Implement gait quality metrics (joint jerk, cost of transport) alongside task rewards for evaluating Mini Cheetah policies
- Apply v-prediction parameterization for more stable diffusion training in locomotion domains
- Leverage the multi-modal training dataset strategy (combining multiple specialist policies) for comprehensive gait coverage

## Limitations & Open Questions
- Paper focuses on simulation results without real hardware validation—sim-to-real gap for diffusion policies remains unclear
- Inference latency of diffusion models, even with DDIM acceleration, may be challenging for Cassie's higher control frequency requirements
- The dependence on demonstration data quality limits applicability when good demonstrations are not available
- Comparison baselines may not include the most recent PPO improvements (e.g., with symmetry losses, curriculum learning)
