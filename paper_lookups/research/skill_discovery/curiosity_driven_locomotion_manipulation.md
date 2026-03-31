# Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks

**Authors:** Various
**Year:** 2023 | **Venue:** CoRL 2023
**Links:** https://openreview.net/forum?id=QG_ERxtDAP-

---

## Abstract Summary
This paper applies curiosity-driven RL using Random Network Distillation (RND) to enable a wheeled-legged robot to learn joint locomotion and manipulation tasks. Intrinsic curiosity rewards encourage broad exploration with minimal extrinsic reward shaping, enabling the robot to learn complex behaviors like opening doors and moving objects. The approach dramatically reduces the need for manual reward engineering, relying primarily on sparse task-completion signals.

## Core Contributions
- Application of RND-based intrinsic motivation to joint locomotion-manipulation tasks
- Demonstrates that sparse extrinsic rewards plus curiosity suffice for complex mobile manipulation
- Reduces reward engineering burden from dense shaped rewards to simple sparse signals
- Real robot validation on wheeled-legged platform performing door opening and object pushing
- Analysis of exploration patterns driven by curiosity in locomotion-manipulation domains
- Comparison with dense reward baselines showing competitive or superior performance
- Insights into when curiosity-driven exploration is most beneficial (sparse reward, high-dimensional action spaces)

## Methodology Deep-Dive
The core approach replaces dense, hand-crafted reward functions with a combination of sparse task-completion rewards and intrinsic curiosity rewards from Random Network Distillation (RND). RND works by maintaining two neural networks: a fixed randomly-initialized target network and a predictor network trained to match the target's outputs. The prediction error serves as an intrinsic reward—states that have been visited frequently have low prediction error (the predictor has learned to match the target), while novel states have high prediction error, driving exploration.

For the locomotion-manipulation domain, the observation space includes proprioceptive data (joint positions, velocities, body pose), exteroceptive data (object positions, door state), and the robot's interaction forces. The RND networks operate on this observation space, providing curiosity bonuses for exploring new configurations of both the robot and the manipulated objects. This is particularly effective because the joint locomotion-manipulation space is enormous—the robot must coordinate whole-body motion, base locomotion, and arm manipulation simultaneously.

The training pipeline uses PPO with the combined reward: R_total = R_extrinsic + α · R_RND, where α is a scaling factor for the intrinsic reward. The extrinsic reward is intentionally kept sparse—e.g., +1 for opening the door, 0 otherwise. Without curiosity, this sparse reward makes learning nearly impossible due to the exploration challenge. With RND, the robot discovers intermediate behaviors (approaching the door, reaching for the handle, grasping and turning) through curiosity-driven exploration.

Domain randomization is applied to object positions, door properties (weight, friction, hinge stiffness), and robot dynamics. The curiosity reward naturally complements domain randomization—it encourages the policy to explore diverse strategies that work across the randomized conditions rather than overfitting to one configuration.

The wheeled-legged platform provides an interesting testbed where locomotion and manipulation are deeply coupled: the robot must position its base appropriately (locomotion) to reach and manipulate objects (manipulation). Pure locomotion or pure manipulation policies fail; the curiosity-driven approach discovers the necessary coordination.

## Key Results & Numbers
- Successfully learns door opening and object pushing from sparse rewards
- RND curiosity enables learning where dense reward shaping is absent
- Competitive with or superior to hand-designed dense reward baselines
- Real robot deployment on wheeled-legged platform
- Training converges in comparable wall-clock time to dense-reward approaches
- Exploration coverage (state space visited) significantly higher with RND
- Ablation: removing RND causes complete failure with sparse rewards
- Curiosity benefit most pronounced in high-dimensional action spaces

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
While Mini Cheetah is a locomotion-only platform without manipulation, curiosity-driven exploration is applicable when training in complex environments where the reward signal is sparse or delayed. For example, if Mini Cheetah is tasked with reaching goal locations in unknown environments, RND-based curiosity could drive exploration without dense waypoint rewards. The approach is also relevant if Mini Cheetah's training is extended to tasks beyond velocity tracking—such as recovery from falls or navigation through mazes. The RND module is lightweight and easily integrated with the existing PPO pipeline in MuJoCo.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
Intrinsic motivation from RND could enhance the skill discovery components (DIAYN/DADS) in Project B. While DIAYN discovers skills through mutual information maximization, RND provides complementary exploration pressure that encourages visiting novel states. This combination could lead to more diverse and useful locomotion primitives at the Primitives level. RND curiosity could also benefit the adversarial curriculum by encouraging exploration of failure modes that the adversary hasn't yet discovered. At the Planner level, curiosity could drive exploration of novel navigation strategies. However, the benefit is moderate since Project B already has structured exploration through DIAYN/DADS and adversarial training.

## What to Borrow / Implement
- Implement RND as an auxiliary exploration module alongside PPO for both projects
- Use RND curiosity to supplement DIAYN/DADS skill discovery in Project B's Primitives level
- Apply sparse reward + curiosity for Mini Cheetah navigation tasks in unknown environments
- Combine RND with adversarial curriculum to discover diverse failure modes
- Use RND prediction error as a diagnostic for training coverage—high error indicates under-explored states
- Consider RND for fall recovery training where the reward is naturally sparse (binary: recovered or not)

## Limitations & Open Questions
- RND curiosity can be distracted by stochastic environment elements (the "noisy TV problem")
- Intrinsic reward scaling (α) requires tuning relative to extrinsic reward magnitude
- Curiosity benefits diminish as the policy matures and most states become familiar
- RND operates on observation space—may not capture all relevant aspects of terrain/contact
- How to combine RND with DIAYN/DADS without redundant exploration signals?
- Curiosity-driven policies may learn fragile exploration behaviors that don't transfer to deployment
