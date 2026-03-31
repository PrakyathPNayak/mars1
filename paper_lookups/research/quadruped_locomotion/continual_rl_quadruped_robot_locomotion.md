# Continual Reinforcement Learning for Quadruped Robot Locomotion

**Authors:** Various
**Year:** 2024 | **Venue:** MDPI Entropy
**Links:** https://www.mdpi.com/1099-4300/26/1/93

---

## Abstract Summary
This paper addresses catastrophic forgetting in continual RL for quadruped locomotion, proposing methods to sequentially learn new locomotion tasks without forgetting previously learned skills. It uses elastic weight consolidation (EWC) and progressive networks to maintain performance across task sequences. The approach enables quadruped robots to incrementally learn new terrains and gaits while retaining mastery of earlier skills.

## Core Contributions
- Application of continual learning techniques (EWC, progressive networks, PackNet) to quadruped locomotion RL
- Systematic evaluation of catastrophic forgetting severity in locomotion task sequences (flat → slope → stairs → rough)
- Elastic weight consolidation adapted for RL: identifies important policy weights for each task and protects them during subsequent learning
- Progressive network architecture that allocates new capacity for each task while freezing previous task parameters
- Comparison of continual learning approaches against naive fine-tuning and multi-task baselines
- Analysis of task ordering effects on continual learning performance
- Demonstration that continual learning maintains >90% performance on old tasks while achieving >85% on new tasks

## Methodology Deep-Dive
Catastrophic forgetting occurs when a neural network trained on a new task loses performance on previously learned tasks. In locomotion, this manifests as a robot that learns to walk on stairs but forgets how to walk on flat ground. Naive sequential fine-tuning is the worst case — each new terrain completely overwrites the previous terrain's policy, achieving good performance only on the most recent task.

Elastic Weight Consolidation (EWC) addresses forgetting by identifying which network weights are important for each learned task and adding a penalty for changing those weights during future learning. The importance of each weight is estimated using the Fisher information matrix computed from the policy gradient at convergence. Weights with high Fisher information are critical for the current task and receive strong protection. During training on a new task, the loss function includes both the new task's RL objective and a quadratic penalty for deviating from the important weights of all previous tasks. The regularization strength (λ) controls the trade-off between learning new tasks and preserving old ones.

Progressive networks take a different approach: rather than protecting existing weights, they allocate entirely new network capacity for each task. The original network (trained on task 1) is frozen, and a new column (sub-network) is created for task 2. The new column receives lateral connections from the frozen column, allowing it to leverage previously learned features without modifying them. For task 3, another column is added with connections to both previous columns. This guarantees zero forgetting (frozen weights cannot change) at the cost of growing network size.

The paper evaluates on a quadruped locomotion task sequence: flat terrain → gentle slopes → stairs → rough terrain. Each task is trained for a fixed number of timesteps before moving to the next. The evaluation metrics track performance on all previous tasks after learning each new one, creating a "forgetting matrix" that quantifies exactly how much each task degrades. The authors also evaluate task ordering sensitivity — does learning slopes before stairs produce different outcomes than stairs before slopes?

The comparison includes: (1) naive fine-tuning (catastrophic forgetting baseline); (2) multi-task training (all tasks simultaneously — the gold standard but requires all task environments simultaneously); (3) EWC; (4) progressive networks; and (5) PackNet (another continual learning method that prunes and frees network capacity). Results show EWC and progressive networks significantly outperform naive fine-tuning, with progressive networks achieving the best anti-forgetting performance at the cost of larger networks.

## Key Results & Numbers
- Naive fine-tuning loses 60-80% performance on previous tasks after learning a new terrain
- EWC reduces forgetting to 10-20% performance loss on previous tasks
- Progressive networks achieve <5% forgetting (near-perfect retention) but require 2-4x more parameters
- PackNet achieves 5-15% forgetting with moderate parameter overhead
- Multi-task baseline (upper bound) achieves 95-100% performance on all tasks simultaneously
- EWC with optimized λ achieves 85-92% of multi-task performance with sequential training
- Task ordering matters: easier→harder sequences produce 10-15% better continual learning than random ordering
- Training time per task is comparable to single-task training (minimal overhead from continual learning)

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Continual learning directly addresses a practical challenge for Mini Cheetah: incrementally training on new terrains or gaits without losing existing capabilities. As Project A expands from flat-ground walking to slopes, stairs, and rough terrain, catastrophic forgetting would force expensive retraining from scratch. EWC can be integrated into Project A's PPO training pipeline with minimal architectural changes — it only adds a regularization term to the loss function. The terrain sequence evaluation (flat → slope → stairs → rough) maps directly to Project A's planned training progression. The finding that easier→harder ordering improves continual learning validates Project A's curriculum design. Progressive networks could provide guaranteed retention if the computational overhead is acceptable.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
Continual learning prevents skill degradation as new primitives are added to Cassie's hierarchy. When the primitives level acquires new skills (e.g., adding jumping after learning walking and running), EWC can protect the policy weights responsible for previously learned skills. This is particularly relevant because Project B's hierarchical structure means changes at one level can cascade and degrade other levels. However, Project B's use of DIAYN/DADS for skill discovery provides some natural protection against forgetting, as skills are represented in a shared latent space. The progressive network approach could be adapted for the primitives level, with each primitive getting its own network column. Task ordering insights inform the sequence in which new skills should be introduced.

## What to Borrow / Implement
- Integrate EWC regularization into Project A's PPO training loop for terrain curriculum progression
- Use the Fisher information matrix to identify and protect critical policy weights during curriculum transitions
- Evaluate progressive networks as an alternative architecture for Project B's primitives level
- Apply the easier→harder task ordering principle to both projects' curriculum designs
- Use the forgetting matrix evaluation methodology to monitor skill retention during training
- Consider PackNet as a middle ground between EWC and progressive networks for parameter-efficient continual learning
- Test continual learning for Project B's hierarchical levels — does training a new hierarchy level degrade existing levels?

## Limitations & Open Questions
- EWC's Fisher information matrix estimation can be noisy in RL settings, leading to suboptimal weight protection
- Progressive networks' growing parameter count becomes prohibitive for many tasks (>10 sequential tasks)
- Continual learning performance is sensitive to the regularization strength (λ in EWC), requiring careful tuning
- Does not address "backward transfer" — learning new tasks that improve performance on old tasks
- How does continual learning interact with hierarchical RL where different levels may need different retention strategies?
- Can continual learning techniques be combined with domain randomization, or do they conflict?
- What is the maximum number of sequential tasks before continual learning methods degrade significantly?
- Limited to proprioceptive locomotion tasks; continual learning for vision-based or manipulation tasks may behave differently
