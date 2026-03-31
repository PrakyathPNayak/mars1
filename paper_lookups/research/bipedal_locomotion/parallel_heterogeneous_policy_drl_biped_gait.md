---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/parallel_heterogeneous_policy_drl_biped_gait.md

**Title:** A Parallel Heterogeneous Policy Deep Reinforcement Learning Algorithm for Biped Robot Gait Optimization
**Authors:** Xingyang Liu, Diyuan Liu, Liang Hua, Jixin Chen
**Year:** 2023
**Venue:** Frontiers in Neurorobotics
**arXiv / DOI:** 10.3389/fnbot.2023.1205775

**Abstract Summary (2-3 sentences):**
This paper proposes a parallel heterogeneous policy DRL approach that simultaneously runs multiple RL algorithms (PPO, SAC, TD3) to optimize bipedal gait, with each algorithm exploring different regions of the solution space. A selection mechanism periodically evaluates the policies produced by each algorithm and promotes the best performers for further training, achieving faster convergence and more stable gaits than any single algorithm alone. The approach demonstrates 30% faster convergence than single-algorithm baselines while producing gaits with improved stability metrics in simulation.

**Core Contributions (bullet list, 4-7 items):**
- Novel parallel heterogeneous policy framework combining PPO, SAC, and TD3 for bipedal gait optimization
- Cross-algorithm policy selection mechanism that periodically evaluates and promotes best-performing policies
- Demonstration of 30% faster convergence compared to single-algorithm training baselines
- Improved gait stability metrics through diversity of exploration strategies across algorithms
- Analysis of how different algorithms specialize in different aspects of gait quality
- Scalable framework design that can incorporate additional algorithms without architectural changes

**Methodology Deep-Dive (3-5 paragraphs):**
The core innovation is running three fundamentally different RL algorithms in parallel, each maintaining its own policy network, value function, and replay mechanism. PPO operates with its on-policy clipped surrogate objective, collecting fresh trajectories and performing multiple epochs of updates with the clipping constraint. SAC uses its maximum entropy framework with twin Q-networks, soft policy updates, and automatic temperature tuning, exploring through stochastic policies that maximize both return and entropy. TD3 employs its twin delayed critics with deterministic policy gradients, target policy smoothing, and delayed policy updates. Each algorithm operates independently on copies of the same simulation environment, using its own exploration strategy to discover different regions of the gait solution space.

The cross-algorithm selection mechanism operates at fixed intervals (every N training episodes, typically N=100). At each selection point, all current policies from all three algorithms are evaluated on a standardized evaluation suite that measures multiple gait quality metrics: forward velocity accuracy, lateral stability (center of mass deviation), step symmetry, energy efficiency (total joint power), ground reaction force smoothness, and foot clearance. A composite score is computed as a weighted combination of these metrics, and the top-performing policies are identified. The selection mechanism then performs knowledge transfer: the best policy's trajectory data is shared with the other algorithms as additional training data (experience sharing), and the best policy's parameters can optionally be used to warm-start underperforming algorithms (parameter sharing). This creates a collaborative dynamic where algorithms benefit from each other's discoveries.

The simulation environment uses a custom bipedal robot model with 6 actuated joints per leg (hip roll, hip pitch, hip yaw, knee, ankle pitch, ankle roll) in a physics simulator. The observation space includes joint positions and velocities, torso orientation, angular velocity, foot contact booleans, and the phase variable of the gait cycle. The action space specifies target joint positions tracked by PD controllers. The reward function combines forward velocity tracking, upright posture maintenance, step regularity, energy minimization, and penalties for falling or excessive lateral sway. Training uses massively parallel environments (256 per algorithm, 768 total) distributed across GPU compute resources.

The authors conduct extensive ablation studies isolating the contribution of each component. Removing the cross-algorithm selection (running three algorithms independently without sharing) still provides some benefit from diversity but loses approximately 15% of the convergence speedup. Removing one algorithm at a time shows that PPO contributes most to training stability, SAC provides the best exploration of diverse gaits, and TD3 tends to find the most energy-efficient solutions. The full heterogeneous system combines these complementary strengths. The authors also compare against ensemble approaches where multiple instances of the same algorithm are run in parallel, finding that the heterogeneous approach significantly outperforms homogeneous ensembles of any single algorithm.

**Key Results & Numbers:**
- 30% faster convergence than single-algorithm baselines (PPO, SAC, or TD3 alone)
- Improved gait stability: 20% reduction in lateral center-of-mass deviation
- 15% improvement in energy efficiency compared to best single-algorithm policy
- Step symmetry index improved from 0.82 to 0.91 (1.0 being perfect)
- PPO-only convergence at 2M steps vs. heterogeneous at 1.4M steps
- Results demonstrated in simulation on a 12-DoF bipedal robot model
- 768 parallel environments used during training across all three algorithms

**Relevance to Project A (Mini Cheetah):** LOW — The bipedal-specific gait optimization is not directly applicable to quadruped locomotion. While the parallel algorithm concept is general, the specific implementation and results target bipedal robots with different kinematic and dynamic challenges.

**Relevance to Project B (Cassie HRL):** MEDIUM — The parallel heterogeneous training concept could be applied to training different levels of the hierarchical controller. For example, different algorithms might be more effective for different hierarchy levels: PPO for the high-level planner, SAC for the primitive-level exploration, and TD3 for the low-level controller. The cross-algorithm knowledge sharing concept could also be adapted for sharing information across hierarchy levels during training.

**What to Borrow / Implement:**
- Parallel heterogeneous training concept for exploring different algorithms at different hierarchy levels
- Cross-algorithm policy selection mechanism as a way to automatically choose the best training approach
- Multi-metric evaluation suite for gait quality assessment (applicable to Cassie gait evaluation)
- The insight that different algorithms specialize in different aspects of locomotion quality
- Experience sharing between parallel training runs as a form of data augmentation

**Limitations & Open Questions:**
- Simulation-only results with no real-robot validation, leaving sim-to-real transfer effectiveness unknown
- Additional computational overhead from running three algorithms simultaneously (approximately 3x compute)
- Selection mechanism hyperparameters (evaluation interval, sharing ratio) require tuning
- Limited to a specific bipedal robot model; generalization to Cassie or other platforms not demonstrated
- Cross-algorithm parameter sharing can cause training instability if policies are too different
- No analysis of how the approach scales with additional algorithms beyond three
---
