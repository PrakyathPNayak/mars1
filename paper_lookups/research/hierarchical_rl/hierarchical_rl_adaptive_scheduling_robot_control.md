---
## 📂 FOLDER: research/hierarchical_rl/

### 📄 FILE: research/hierarchical_rl/hierarchical_rl_adaptive_scheduling_robot_control.md

**Title:** Hierarchical Reinforcement Learning with Adaptive Scheduling for Robot Control
**Authors:** Zheng Fang, Qi Zhao, Xin Xu, Haibin Duan
**Year:** 2023
**Venue:** Engineering Applications of Artificial Intelligence (Elsevier)
**arXiv / DOI:** 10.1016/j.engappai.2023.107160

**Abstract Summary (2–3 sentences):**
This paper proposes the Hierarchical RL with Adaptive Scheduling (HAS) algorithm, featuring a high-level controller that adaptively schedules low-level continuous options for robot control tasks. The adaptive scheduling mechanism dynamically balances exploration and exploitation across hierarchical levels, addressing the challenge of coordinating learning at different temporal scales. The approach is evaluated on continuous control benchmarks demonstrating improved sample efficiency and option utilization compared to flat RL and fixed-schedule hierarchical baselines.

**Core Contributions (bullet list, 4–7 items):**
- Hierarchical RL with Adaptive Scheduling (HAS) algorithm for multi-level robot control
- Adaptive option scheduling mechanism that dynamically balances exploration and exploitation at the high level
- Theoretical analysis of convergence properties under adaptive scheduling
- 35% improvement in sample efficiency over flat RL baselines on continuous control tasks
- Improved option utilization rates showing more balanced use of available options
- Evaluation on multiple robot control benchmarks including locomotion and manipulation
- Analysis of scheduling dynamics showing how the mechanism adapts across training phases

**Methodology Deep-Dive (3–5 paragraphs):**
The HAS algorithm extends the standard options framework with an adaptive scheduling mechanism at the high level. The architecture consists of a high-level controller (scheduler) that selects among K continuous options, each implemented as a parameterized sub-policy. Unlike standard hierarchical approaches where the high-level policy uses a fixed softmax selection over options, HAS introduces a scheduling function that modulates the option selection probabilities based on a combination of option performance history, option utilization statistics, and a time-varying exploration bonus. The scheduling function is designed to address a common failure mode in hierarchical RL: the high-level policy prematurely committing to a single option that provides moderate but sub-optimal performance, while neglecting other options that might be superior after further training.

The adaptive scheduling mechanism operates by maintaining a running estimate of each option's value and uncertainty. For each option ω_k, the scheduler tracks the exponential moving average of returns R̄_k obtained when the option is selected, as well as an uncertainty estimate σ_k based on the variance of recent returns. The option selection probability is computed as p(ω_k | s) ∝ exp(R̄_k + α · σ_k), where α is an exploration coefficient that controls the balance between selecting high-performing options (exploitation) and selecting uncertain options (exploration). This formulation resembles Upper Confidence Bound (UCB) strategies from multi-armed bandit theory, adapted to the non-stationary setting where option policies are continuously improving. The exploration coefficient α is annealed over training, starting high (encouraging broad option exploration) and decreasing to a low value (encouraging exploitation of the best-performing options).

The low-level continuous options are parameterized as neural network policies that map states to continuous actions. Each option is trained using PPO with option-specific reward shaping that encourages diverse and complementary behaviors. The option termination conditions are learned alongside the option policies using the termination gradient from the Option-Critic framework. A key design choice is that options share a common observation encoder but have separate policy heads, allowing feature sharing while maintaining distinct behaviors. The option policies are updated using trajectories collected while the option is active, with importance sampling corrections to account for the non-stationary option selection probabilities induced by the adaptive scheduler.

The interaction between the scheduler and the option policies creates a co-adaptation dynamic that the authors carefully analyze. When the scheduler increases the selection probability for an option, that option receives more training data and improves faster. This improvement, in turn, increases the option's estimated value R̄_k, further increasing its selection probability—a positive feedback loop that could lead to option collapse (all probability mass on one option). The uncertainty-based exploration bonus counteracts this by maintaining a minimum exploration level for all options. The authors provide a theoretical analysis showing that under mild conditions, the adaptive scheduling converges to a policy that selects options proportionally to their optimal value, and that the exploration bonus ensures all options receive sufficient data for convergence.

The experimental evaluation uses several continuous control benchmarks from the MuJoCo suite, including locomotion tasks (Ant, Humanoid), navigation with obstacles, and manipulation tasks. HAS is compared against flat PPO, Option-Critic with standard softmax selection, feudal networks (FuN), and hierarchical actor-critic (HAC). The results show that HAS achieves 35% better sample efficiency than flat PPO (measured as environment steps to reach 80% of optimal performance) and 15–20% improvement over standard Option-Critic. The option utilization analysis reveals that HAS maintains more balanced option usage throughout training compared to Option-Critic, where one or two options typically dominate. Visualization of the learned options shows that HAS discovers more diverse and complementary option behaviors, with different options specializing in different phases of the task (e.g., for locomotion: acceleration, steady-state, turning, recovery).

**Key Results & Numbers:**
- 35% improvement in sample efficiency over flat RL on continuous control benchmarks
- 15–20% improvement in sample efficiency over standard Option-Critic
- More balanced option utilization rates (Gini coefficient 0.2 vs 0.6 for Option-Critic)
- Successful option specialization: different options learn complementary locomotion behaviors
- Convergence guaranteed under mild conditions with adaptive scheduling
- Exploration coefficient annealing schedule: α from 2.0 to 0.1 over training
- Demonstrated on Ant (8 DoF), Humanoid (17 DoF), and manipulation tasks

**Relevance to Project A (Mini Cheetah):** LOW — More applicable to hierarchical architectures than the flat PPO approach currently used for Mini Cheetah. The adaptive scheduling concepts would become relevant if the Mini Cheetah project adopts a hierarchical structure with multiple locomotion options.

**Relevance to Project B (Cassie HRL):** HIGH — The adaptive scheduling mechanism is directly relevant to the Planner level's decision-making about when to switch between locomotion primitives. The UCB-inspired exploration-exploitation balance addresses a key challenge in hierarchical locomotion: ensuring that all primitives receive sufficient training while the planner learns to select the best primitive for each situation. The option utilization analysis methodology is also valuable for diagnosing primitive usage patterns in the Cassie system.

**What to Borrow / Implement:**
- UCB-inspired adaptive scheduling for the planner level's primitive selection
- Running value and uncertainty estimates for each locomotion primitive
- Exploration coefficient annealing schedule for training the hierarchical system
- Shared observation encoder with separate primitive policy heads
- Option utilization metrics (Gini coefficient) for monitoring primitive usage balance during training

**Limitations & Open Questions:**
- The adaptive scheduling adds hyperparameters (initial α, annealing schedule, moving average window) that require tuning
- Theoretical convergence results assume stationary option policies during analysis, but option policies are continuously updated in practice
- The method has been validated only on simulated MuJoCo tasks; real-robot deployment is not demonstrated
- Scaling to large numbers of options (>8) may require more sophisticated scheduling strategies
- The relationship between option diversity and task performance is empirically demonstrated but not theoretically characterized
- Integration with safety constraints (as needed for the Cassie LCBF safety filter) is not addressed
---
