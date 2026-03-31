# Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse RL

**Authors:** Feiyang Wu et al.
**Year:** 2024 | **Venue:** ICRA 2024
**Links:** https://arxiv.org/abs/2309.16074

---

## Abstract Summary
This paper uses inverse reinforcement learning to infer reward functions from expert demonstrations of bipedal walking, producing interpretable and transferable reward structures. The learned rewards enable bipedal robots to adapt walking strategies to unseen, uneven terrains without manual reward design. The approach demonstrates improved explainability over standard RL with hand-crafted rewards, while achieving comparable or superior locomotion performance.

## Core Contributions
- Inverse RL framework for inferring bipedal locomotion reward functions from demonstrations
- Interpretable reward structures that reveal which aspects of walking are most important
- Transfer of learned rewards to unseen terrain types without re-training the reward
- Improved explainability: the reward function itself is human-readable and analyzable
- Comparison showing IRL-derived rewards match or exceed manually designed rewards
- Adaptation capability: the inferred reward naturally encourages terrain-appropriate behavior
- Framework applicable to different bipedal platforms with minimal modification

## Methodology Deep-Dive
The approach follows a two-stage pipeline: first, inverse RL infers a reward function from expert demonstrations; second, forward RL uses this reward to train a locomotion policy. The expert demonstrations come from either motion capture of human walking, teleoperation of the robot, or a pre-existing high-quality controller. The key advantage over direct imitation learning is that the learned reward function generalizes—it captures the underlying objectives of good walking rather than mimicking specific trajectories.

The IRL method used is based on Maximum Entropy IRL (MaxEnt IRL), which finds the reward function that makes the expert demonstrations most likely under a Boltzmann-rational model. The reward is parameterized as a linear combination of hand-designed features (joint angles, velocities, foot contact patterns, CoM trajectory, energy consumption) with learned weights. This linear structure ensures interpretability—each weight indicates the importance of the corresponding feature in defining good walking.

After the reward function is inferred, standard RL (PPO) is used to train a locomotion policy that maximizes this reward. The crucial finding is that the IRL-derived reward transfers well: a reward learned from demonstrations on flat ground produces good walking on stairs, slopes, and rough terrain. This is because the reward captures general principles of bipedal stability (maintain upright posture, symmetric gait, appropriate foot clearance) rather than terrain-specific trajectories.

The adaptation mechanism works through the interplay between the general reward and the policy's terrain response. On uneven terrain, the policy must adjust its gait to maintain the reward criteria (stability, efficiency), naturally discovering terrain-appropriate strategies. This is more robust than trajectory-based imitation, which would fail when the terrain differs from the demonstration conditions.

Experiments are conducted on Cassie and humanoid robot models in simulation, with ablation studies comparing different IRL formulations, feature sets, and demonstration quality. The results show that even modest-quality demonstrations yield reward functions that produce robust locomotion, as the IRL procedure extracts the underlying objectives rather than copying specific motions.

## Key Results & Numbers
- IRL-derived rewards match or exceed hand-designed reward performance on flat ground
- Superior generalization to unseen terrains (stairs, slopes, rough) compared to imitation learning
- Interpretable reward weights: joint symmetry and CoM stability emerge as highest-weighted features
- Robust to demonstration quality: even sub-optimal demonstrations yield useful rewards
- Validated on Cassie bipedal robot model in simulation
- Reward transfer across terrain types without reward re-training
- 15-30% improvement in stability metrics on uneven terrain vs. flat-ground-only imitation
- Training time comparable to standard RL once reward is inferred

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
IRL is applicable to Mini Cheetah when demonstration data is available—either from an existing high-quality controller or from teleoperation. The interpretable reward structure could help diagnose and improve the current PPO training by revealing which locomotion features matter most. However, Mini Cheetah's current pipeline likely has sufficiently good hand-crafted rewards for basic locomotion, so the main benefit would be in reducing reward engineering effort for new tasks or in providing insights for reward improvement. The approach is straightforward to integrate with the MuJoCo training pipeline.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This paper is directly applicable to Project B as it specifically studies bipedal locomotion reward design for Cassie. The IRL-derived reward could replace or supplement hand-crafted rewards at the Controller level of the 4-level hierarchy. The interpretable reward weights provide actionable insights into what matters for Cassie's stability—information that could inform reward design at other hierarchy levels too. The terrain transfer capability is particularly valuable: a reward learned from flat-ground demonstrations could be used for the adversarial curriculum training where terrain difficulty varies. The MaxEnt IRL framework is compatible with PPO training already used in Project B. Additionally, the feature-based reward structure could interface with the Dual Asymmetric-Context Transformer by providing an interpretable objective function that the transformer can optimize against.

## What to Borrow / Implement
- Apply MaxEnt IRL to learn Cassie's reward function from existing controller demonstrations
- Use IRL-derived feature importance weights to validate and refine hand-crafted reward functions
- Implement the two-stage pipeline: IRL for reward learning → PPO for policy training
- Leverage reward transferability across terrains for the adversarial curriculum in Project B
- Use the interpretable reward structure for debugging and analysis of policy behavior
- Consider IRL at the Primitives level to infer rewards for different locomotion skills (DIAYN/DADS)

## Limitations & Open Questions
- Requires expert demonstrations, which may not be available for all target behaviors
- Linear reward structure may not capture complex, non-linear reward landscapes
- MaxEnt IRL assumes the expert is Boltzmann-rational, which may not hold for human demonstrations
- Feature engineering is still required—the IRL learns weights but the features are hand-designed
- How to apply IRL in a hierarchical setting where each level has different reward objectives?
- Scalability to very high-dimensional observation spaces (e.g., with vision inputs) is unclear
