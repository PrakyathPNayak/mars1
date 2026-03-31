---
## 📂 FOLDER: research/hierarchical_rl/

### 📄 FILE: research/hierarchical_rl/effectively_learning_initiation_sets_hrl.md

**Title:** Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning
**Authors:** Akhil Bagaria, Jason Senthil, Matthew Slivinski, George Konidaris
**Year:** 2023
**Venue:** NeurIPS 2023
**arXiv / DOI:** via NeurIPS proceedings

**Abstract Summary (2–3 sentences):**
This paper addresses a core challenge in hierarchical reinforcement learning option frameworks: automatically discovering when options should be initiated. The authors propose learned initiation set classifiers that determine the subset of states from which each option can be reliably executed, improving option reusability and composability. The approach is evaluated on MuJoCo robotics tasks including manipulation and navigation, demonstrating significant improvements in option discovery efficiency and hierarchical policy performance.

**Core Contributions (bullet list, 4–7 items):**
- Novel method for learning initiation set classifiers that identify valid starting states for each option
- Theoretical analysis showing that properly learned initiation sets improve option composability and reduce negative transfer
- Practical algorithm integrating initiation set learning with option policy and termination function optimization
- Demonstration that restricting option availability to learned initiation sets prevents degenerate option usage patterns
- Evaluation on continuous MuJoCo control tasks showing 40% improvement in option discovery efficiency
- Analysis of how initiation set quality affects hierarchical policy performance and transfer learning
- Connection between initiation sets and safe/reliable option execution in continuous state spaces

**Methodology Deep-Dive (3–5 paragraphs):**
The paper begins by formalizing the problem of initiation set learning within the options framework. In the standard options framework (Sutton, Precup, Singh 1999), each option ω is defined by a tuple (I_ω, π_ω, β_ω) comprising an initiation set, intra-option policy, and termination function. Most prior work, including the Option-Critic architecture, simplifies the problem by setting I_ω = S (all states), making every option available everywhere. The authors argue that this simplification is harmful: it allows options to be initiated from states where they cannot be successfully executed, leading to wasted exploration, negative transfer between options, and degenerate hierarchical policies. The key insight is that learning accurate initiation sets—classifiers that predict whether an option can achieve its intended effect from a given state—is essential for building composable and reusable hierarchical policies.

The proposed approach learns initiation set classifiers as binary neural network classifiers I_ω(s) ∈ {0, 1} trained on data collected during option execution. The classifier is trained to distinguish between states from which the option successfully achieves its termination condition (positive examples) and states from which the option fails or produces undesirable outcomes (negative examples). The training signal comes from hindsight: after an option trajectory is collected, the initiation classifier is updated based on whether the option reached its intended subgoal. This creates a feedback loop where the initiation set becomes more accurate as more option trajectories are collected, and better initiation sets lead to more focused exploration and faster option improvement. The classifiers use a shared feature representation with the option policies to leverage common state representations.

The integration with option learning follows a three-phase iterative process. In the first phase, the current policy-over-options selects an option ω from the set of options whose initiation sets include the current state. In the second phase, the selected option executes using its intra-option policy until termination. In the third phase, all components are updated: the option policy π_ω is updated using the collected trajectory data, the termination function β_ω is updated using the termination gradient, and the initiation set classifier I_ω is updated based on option execution success. The policy-over-options μ(ω|s) is also restricted to only consider options for which I_ω(s) = 1, preventing the selection of options that cannot be meaningfully executed from the current state. This restriction is enforced through action masking during policy-over-options training.

The authors provide theoretical analysis showing that properly learned initiation sets satisfy two key properties: completeness (the initiation set includes all states from which the option can succeed) and soundness (the initiation set excludes states from which the option will fail). In practice, the learned classifiers approximate these properties, with the quality improving as more data is collected. The analysis shows that sound initiation sets are more important than complete ones—it is better to conservatively restrict option availability than to allow options to be attempted from states where they will fail. The authors also analyze the sample complexity of initiation set learning, showing that it scales logarithmically with the state space size under mild assumptions about the option policy's success region geometry.

Experimental evaluation is conducted on several MuJoCo continuous control benchmarks including ant navigation, manipulation, and multi-room tasks. The baselines include Option-Critic with full initiation sets (I_ω = S), fixed geometric initiation sets (e.g., state-space regions), and the proposed learned initiation sets. The results show that learned initiation sets improve option discovery efficiency by approximately 40%, measured as the number of environment interactions needed to discover a set of options that solves the task. Additionally, options learned with proper initiation sets transfer better to new tasks in the same domain, as each option has a well-defined region of competence. The analysis includes visualizations of learned initiation sets showing that they correspond to interpretable state-space regions (e.g., an option for navigating through a door has an initiation set corresponding to states near the door).

**Key Results & Numbers:**
- 40% improvement in option discovery efficiency compared to full-state initiation sets
- Better option composability demonstrated through improved transfer learning across MuJoCo tasks
- Learned initiation sets converge to near-optimal classifiers within 100K–500K environment steps
- Options with learned initiation sets show 25% higher success rates when composed in sequence
- Visualization confirms initiation sets correspond to interpretable state-space regions
- Sound (conservative) initiation sets consistently outperform complete (permissive) ones

**Relevance to Project A (Mini Cheetah):** LOW — More relevant to hierarchical approaches than the flat PPO architecture currently used for Mini Cheetah. However, if the Mini Cheetah project evolves toward a hierarchical structure with distinct locomotion modes, initiation set learning could determine when each mode is appropriate.

**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to learning when to activate and deactivate locomotion primitives in the options framework. For the Cassie HRL system, initiation sets would determine which locomotion primitives (walking, running, turning, recovery) are available in each state, preventing the planner from selecting inappropriate primitives. This is particularly important for safety-critical transitions—e.g., a running primitive should not be initiated from a state where the robot is off-balance.

**What to Borrow / Implement:**
- Initiation set classifier architecture for determining primitive availability in each state
- Hindsight-based training signal for updating initiation classifiers from option execution outcomes
- Action masking mechanism to restrict the planner to only selecting valid primitives
- Conservative (sound) initiation set strategy for safety-critical locomotion primitive selection
- Shared feature representation between initiation classifiers and option policies for parameter efficiency

**Limitations & Open Questions:**
- Binary initiation classifiers may be too coarse; a soft probability of success could enable more nuanced option selection
- The method requires sufficient exploration to discover both positive and negative examples for each option
- Scalability to large numbers of options (>10) with overlapping initiation sets has not been extensively studied
- The approach assumes stationary option policies; if option policies continue to improve, initiation sets may need continuous re-learning
- Integration with deep hierarchical architectures (more than two levels) is not explored
- Real-robot validation is missing; all experiments are in simulation
---
