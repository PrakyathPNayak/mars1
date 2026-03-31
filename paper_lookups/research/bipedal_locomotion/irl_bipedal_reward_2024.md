# Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse Reinforcement Learning

**Authors:** Wu et al. (Georgia Tech)
**Year:** 2024 | **Venue:** ICRA
**Links:** https://lab-idar.gatech.edu/wp-content/uploads/2023/09/ICRA2024_IRL_Reward_Shaping_Wu.pdf

---

## Abstract Summary
This paper addresses the challenge of designing reward functions for bipedal locomotion by leveraging Inverse Reinforcement Learning (IRL) to learn reward structures from expert demonstrations. Rather than hand-crafting complex reward functions that attempt to encode all aspects of effective bipedal walking, the authors use demonstrations from either human motion capture data or expert-trained bipedal policies to infer an underlying reward function that explains the demonstrated behavior. The learned reward function captures implicit strategies and preferences that are difficult to articulate manually, such as subtle weight-shifting patterns, anticipatory foot placement, and energy-efficient posture maintenance.

The key contribution is a two-phase pipeline: (1) IRL-based reward learning from a small set of expert demonstrations (as few as 20 trajectories), and (2) standard RL policy training using the learned reward. The authors demonstrate that policies trained with IRL-derived rewards outperform those trained with hand-crafted rewards on rough terrain bipedal walking tasks, particularly in terms of robustness to terrain perturbations and naturalness of the gait. The learned rewards transfer effectively to train entirely new agents, including those with different morphological parameters.

The work is validated on Cassie-like bipedal robot simulations, showing that IRL-derived rewards produce more robust and natural walking behaviors than even carefully-tuned expert-designed reward functions. The transferability of learned rewards to modified robots suggests the IRL process captures fundamental principles of efficient bipedal locomotion rather than robot-specific artifacts.

## Core Contributions
- A practical IRL pipeline for bipedal locomotion that learns reward functions from as few as 20 expert demonstration trajectories
- Demonstration that IRL-derived rewards capture implicit locomotion strategies (anticipatory weight shifting, terrain-adaptive foot clearance) that are missed by hand-crafted rewards
- Reward transfer experiments showing learned rewards generalize to train new agents with different mass distributions and leg lengths (±15% morphological variation)
- Comparison of Maximum Entropy IRL (MaxEnt IRL) and Adversarial IRL (AIRL) for locomotion reward learning, with AIRL showing superior transfer properties
- Policies trained with IRL rewards achieve 23% higher survival rate on rough terrain compared to expert-designed rewards
- Analysis of the learned reward function structure revealing previously unrecognized important features (hip abduction velocity, ankle torque patterns)
- Open-source release of the IRL reward learning pipeline with integration for MuJoCo bipedal environments

## Methodology Deep-Dive
The IRL pipeline uses Adversarial Inverse Reinforcement Learning (AIRL), which learns a reward function r_θ(s,a) by framing reward learning as a discriminative problem. A discriminator network D_θ distinguishes between expert demonstration state-action pairs and policy-generated pairs, and the learned discriminator directly defines the reward: r_θ(s,a) = log D_θ(s,a) - log(1 - D_θ(s,a)). The policy π_φ is trained to maximize this reward using PPO, creating a GAN-like training loop where the policy tries to fool the discriminator while the discriminator tries to distinguish expert from policy behavior.

Expert demonstrations are collected from two sources: (1) motion capture data of human walking, retargeted to the bipedal robot's kinematic structure, and (2) trajectories from a previously-trained expert policy on flat terrain. The authors find that combining both demonstration sources produces the highest-quality reward functions, as human demonstrations provide naturalness while expert policy demonstrations provide robot-specific feasibility.

The AIRL discriminator architecture uses a reward network decomposed as r_θ(s,a) = g_θ(s) + h_θ(s') - h_θ(s), where g_θ is the state-only reward component and h_θ is a shaping potential. This decomposition ensures that the learned reward is robust to differences in dynamics between the demonstration environment and the deployment environment—a critical property for transferring rewards to morphologically different robots or to sim-to-real transfer.

The state features provided to the reward network include: base position and velocity (6D), base orientation as quaternion (4D), joint positions (10 for Cassie-like biped), joint velocities (10), foot contact forces (2), and center-of-pressure location (2). The action features are the target joint position commands (10D). The reward network is a 3-layer MLP (128×128×64) with Tanh activations, chosen for smoothness of the learned reward landscape.

After AIRL training converges (approximately 500 discriminator-policy update cycles), the learned reward function r_θ is frozen and used to train fresh policies from scratch. This second-phase training uses standard PPO with the IRL-derived reward, running for 5000 iterations in MuJoCo with 512 parallel environments. The key validation is that policies trained with the frozen reward on rough terrain (height perturbations ±5 cm) achieve significantly better performance than policies trained with hand-crafted rewards.

## Key Results & Numbers
- Survival rate on rough terrain: 89% with IRL reward vs. 66% with hand-crafted reward (23% improvement)
- Gait naturalness score (human evaluation): 4.2/5.0 with IRL reward vs. 3.4/5.0 with hand-crafted reward
- Reward transfer to ±15% morphological variation: <8% performance degradation, vs. 25% degradation for hand-crafted rewards
- IRL reward learning requires only 20 expert trajectories (each ~5 seconds long) for effective reward inference
- AIRL training converges in ~500 discriminator-policy update cycles (~6 hours on single GPU)
- Second-phase policy training with IRL reward converges 30% faster than with hand-crafted reward
- Discovered important reward features: hip abduction velocity (weight: 0.18), ankle torque smoothness (weight: 0.14), center-of-pressure trajectory regularity (weight: 0.12)
- Energy efficiency: IRL-reward policies consume 12% less energy than hand-crafted reward policies on flat terrain

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The IRL concept is applicable to Mini Cheetah for learning reward functions from demonstrations. If expert Mini Cheetah locomotion data is available (from teleoperation, motion capture of quadruped animals, or expert-trained policies), AIRL can learn reward functions that capture nuanced locomotion quality aspects. However, the paper focuses on bipedal locomotion, so direct transfer of the learned reward features is limited. The methodology—rather than the specific reward structures—is what transfers to the quadruped domain.

The reward transfer experiments are particularly relevant: showing that IRL-derived rewards generalize across morphological variations suggests they could support Mini Cheetah's sim-to-real transfer, where the simulated and real robot have parameter mismatches.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critically relevant to Cassie's bipedal locomotion. The IRL-derived reward functions were specifically developed and validated on Cassie-like simulations, making the results directly transferable. The AIRL pipeline can generate high-quality reward functions for Cassie's Controller level from demonstrations, potentially surpassing hand-crafted rewards.

The reward decomposition into state-reward and shaping potential (g_θ + h_θ(s') - h_θ(s)) is particularly valuable for Cassie's hierarchical architecture, as the shaping potential provides a reward function that is invariant to dynamics changes across hierarchy levels. The demonstration that hip abduction velocity and ankle torque patterns are important reward features provides direct insight for Cassie's reward engineering. The 23% survival rate improvement on rough terrain would significantly enhance Cassie's outdoor deployment capability.

## What to Borrow / Implement
- Implement the AIRL pipeline for learning Cassie's Controller-level reward function from expert walking demonstrations
- Use the reward decomposition (g_θ + h_θ(s') - h_θ(s)) to ensure learned rewards transfer robustly across simulation-to-real dynamics differences
- Incorporate the discovered reward features (hip abduction velocity, ankle torque smoothness, CoP regularity) into Cassie's hand-crafted reward as additional terms
- Apply the reward transfer methodology to validate that learned rewards generalize across Cassie's morphological uncertainties
- Explore combining human motion capture demonstrations with expert policy demonstrations for richer reward learning

## Limitations & Open Questions
- AIRL training requires an iterative GAN-like process that can be unstable; mode collapse in the discriminator leads to degenerate reward functions
- The 20-trajectory demonstration requirement, while modest, still requires access to expert demonstrations which may not be available for novel robot morphologies
- The learned reward function is a neural network that lacks interpretability; understanding why certain behaviors are rewarded requires post-hoc analysis
- Extension to hierarchical reward learning (learning rewards for multiple hierarchy levels simultaneously) is not addressed—a critical need for Cassie's 4-level architecture
