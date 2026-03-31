# Two-Layered Reward Reinforcement Learning for Humanoid Robot Motion Tracking

**Authors:** Various
**Year:** 2025 | **Venue:** MDPI Mathematics
**Links:** https://www.mdpi.com/2227-7390/13/21/3445

---

## Abstract Summary
This paper proposes a two-layered reward structure for humanoid robot motion tracking: upper-level rewards govern the primary locomotion task while lower-level rewards manage auxiliary objectives such as energy efficiency and balance stability. The weights between layers adapt automatically during training using a learned scheduling mechanism. This hierarchical reward design supports efficient exploration and robust policy learning for complex humanoid motion.

## Core Contributions
- Two-layered reward hierarchy separating primary task objectives from auxiliary objectives
- Automatic reward weight adaptation between layers during training via learned scheduling
- Upper-level rewards focus on motion tracking; lower-level rewards handle energy and stability
- Improved training stability and convergence compared to flat multi-term rewards
- Robust humanoid motion tracking across diverse movement patterns
- Principled approach to multi-objective reward balancing without manual tuning
- Analysis of reward layer interactions and their temporal dynamics during training

## Methodology Deep-Dive
The key idea is structuring the reward function into two hierarchical layers rather than a flat weighted sum. The upper layer contains rewards directly related to the primary task—in this case, motion tracking objectives such as joint angle tracking error, end-effector position tracking, and root trajectory tracking. The lower layer contains auxiliary rewards that support but don't directly define the task: energy consumption, joint acceleration smoothness, contact force minimization, and balance stability metrics.

The total reward is computed as R_total = w_upper · R_upper + w_lower · R_lower, where w_upper and w_lower are automatically scheduled. Early in training, w_lower (auxiliary rewards) is weighted higher to encourage stable, energy-efficient exploration before the policy attempts precise motion tracking. As training progresses, w_upper (task rewards) gradually increases while w_lower decreases, shifting focus toward task performance while maintaining the already-learned stability behaviors.

The weight scheduling is not a simple linear function but is learned through a meta-learning mechanism. A small neural network observes training statistics (episode returns, constraint violation rates, gradient magnitudes) and outputs the layer weights. This network is trained with a slower learning rate than the policy, effectively performing online hyperparameter optimization.

The policy architecture is standard: an observation encoder processes proprioceptive state and reference motion, and an MLP actor-critic produces joint actions. PPO is the base RL algorithm. The two-layered reward structure modifies only the reward computation, not the policy architecture or training algorithm, making it easy to integrate with existing pipelines.

Experiments compare the two-layered approach against: (1) flat rewards with fixed weights, (2) flat rewards with manually scheduled weights, (3) single-objective tracking-only rewards, and (4) constraint-based approaches. The two-layered method achieves the best combination of tracking accuracy and auxiliary objective satisfaction. Importantly, it converges faster than flat rewards because the early emphasis on auxiliary objectives (stability, low energy) provides a good initialization before the policy focuses on precise tracking.

The ablation studies reveal that the automatic weight scheduling is critical—fixed two-layered weights provide some benefit over flat rewards, but the adaptive scheduling captures training phase-dependent trade-offs that static weights cannot.

## Key Results & Numbers
- Motion tracking error reduced by 20% compared to flat multi-term reward baselines
- Training convergence 30% faster than flat reward approaches due to structured exploration
- Energy consumption 15% lower while maintaining tracking accuracy
- Automatic weight scheduling outperforms fixed schedule by 10-15% across metrics
- Robust across diverse motion types: walking, turning, crouching, reaching
- Fall rate reduced from 12% to 3% compared to tracking-only rewards
- Meta-learned weight schedule generalizes across different motion patterns
- Validated on humanoid robot model with 20+ DoF in simulation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The two-layered reward structure is applicable to Mini Cheetah's PPO training where multiple objectives (velocity tracking, energy, smoothness, stability) must be balanced. The upper layer would contain velocity tracking rewards while the lower layer handles energy and stability. The automatic weight scheduling would replace manual reward weight tuning during training. For Mini Cheetah's 12-DoF system in MuJoCo, the approach requires minimal modification—only the reward computation changes. The early emphasis on stability through lower-layer weighting could improve initial training phases where Mini Cheetah often falls. However, the benefit over simpler approaches (e.g., constraint-based from Paper 4) is unclear for a relatively lower-DoF quadruped.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The hierarchical reward structure aligns naturally with Project B's multi-level hierarchy. Each level of the 4-level hierarchy (Planner, Primitives, Controller, Safety) can be viewed as a reward layer with different temporal priorities. The Planner level has long-horizon task rewards (upper layer), while Safety/Controller levels have immediate stability rewards (lower layer). The automatic weight scheduling mechanism could manage the training curriculum across hierarchy levels—initially emphasizing safety and balance, then progressively weighting task completion. The meta-learned scheduling is particularly relevant to the adversarial curriculum component, where the difficulty of the training environment changes and reward priorities should adapt accordingly. For Cassie's 20+ DoF, the multi-objective nature of the training (tracking, balance, energy, safety) makes the structured reward approach especially valuable compared to flat reward sums.

## What to Borrow / Implement
- Implement the two-layered reward structure for both projects' PPO training
- Use upper layer for task-specific objectives, lower layer for stability and efficiency
- Implement the meta-learned weight scheduling using a small auxiliary network
- Apply the phased training approach: stability focus first, then task focus
- Map the two-layer concept to Project B's 4-level hierarchy for level-specific reward management
- Combine with constraint-based approach: constraints for hard safety, layered rewards for optimization

## Limitations & Open Questions
- The meta-learning of weight schedules adds computational overhead and training complexity
- Only two layers may not capture the full complexity of multi-level hierarchical objectives
- The approach assumes a clear separation between "primary task" and "auxiliary" objectives, which may be ambiguous
- Weight scheduling learned on one motion type may not generalize to very different motions
- How to extend from two layers to N layers for Project B's 4-level hierarchy?
- Interaction between automatic weight scheduling and curriculum learning (terrain difficulty) unexplored
