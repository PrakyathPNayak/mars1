# RLIF: Interactive Imitation Learning as Reinforcement Learning

**Authors:** Various (UC Berkeley)
**Year:** 2024 | **Venue:** UC Berkeley Technical Report
**Links:** https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-17.pdf

---

## Abstract Summary
RLIF reframes interactive imitation learning by treating human intervention signals as implicit reward feedback for reinforcement learning. Rather than simply collecting expert corrections as supervised labels, the method interprets the presence or absence of human intervention as a binary reward signal, enabling the policy to surpass expert performance through RL optimization. This bridges the gap between imitation and reinforcement learning, demonstrated on manipulation tasks with implications for locomotion.

## Core Contributions
- Reinterprets human intervention in interactive IL as an implicit binary reward signal for RL
- Enables learned policies to surpass expert performance by optimizing the intervention-based reward
- Bridges the theoretical gap between imitation learning and reinforcement learning
- Achieves sample-efficient learning from minimal human interaction
- Demonstrates that non-expert humans can provide effective training signal through intervention timing
- Provides theoretical analysis showing intervention-based reward is a valid reward function under mild assumptions
- Shows the approach is robust to suboptimal or inconsistent human intervention patterns

## Methodology Deep-Dive
The key insight is that when a human expert intervenes to correct a robot's behavior, this intervention event itself carries reward information. Specifically, the absence of intervention implies the policy is performing acceptably, while intervention implies the policy is making an error. RLIF formalizes this by defining a reward function r(s, a) = 1 if no intervention occurs and r(s, a) = 0 (or negative) if the human intervenes. This transforms the interactive IL setup into a standard RL problem.

The training procedure alternates between two phases. In the data collection phase, the learned policy executes actions while a human observer monitors performance and intervenes when necessary. Each transition is labeled with the binary intervention signal. In the policy optimization phase, this labeled data is used to update the policy via standard RL algorithms (PPO or SAC). The intervention-based reward provides denser feedback than sparse task rewards, enabling efficient learning.

A critical theoretical contribution is showing that the intervention-based reward is aligned with the true task objective under reasonable assumptions about human behavior. If the human intervenes when the policy deviates from acceptable behavior and does not intervene otherwise, the intervention signal provides an unbiased (though noisy) estimate of policy quality. The paper provides formal bounds on the gap between the intervention-based reward and the true reward.

The method handles suboptimal experts by noting that the RL optimization is not bounded by expert performance. While standard IL methods can at best match the expert, RLIF uses the expert's intervention signal merely as a reward function. If the intervention signal correctly identifies bad behaviors (even if the expert's own corrections are imperfect), the RL optimization can discover better strategies than the expert demonstrates. This is the mechanism by which RLIF surpasses expert performance.

Practical implementation uses an off-policy RL algorithm with a replay buffer that stores intervention-labeled transitions. The replay buffer is populated incrementally through interactive sessions, and the policy is updated between sessions. A confidence-based scheduling mechanism determines when to request human monitoring vs. autonomous execution, reducing human effort over time.

## Key Results & Numbers
- Policy surpasses expert performance by 15-30% on manipulation benchmarks
- Requires 50% less human time compared to standard DAgger
- Intervention-based reward converges faster than sparse task reward RL
- Robust to 20% noise in human intervention timing
- Sample efficiency: achieves expert-level performance with 2-3 interactive sessions
- Theoretical guarantees on reward alignment under mild human behavior assumptions
- Non-expert humans can provide effective intervention signals with minimal training

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
RLIF could be applied when human teleoperators monitor Mini Cheetah locomotion and intervene during failures. The intervention signal (e.g., "robot is about to fall") provides a natural binary reward that supplements the PPO reward function from MuJoCo simulation. This is particularly valuable for sim-to-real fine-tuning, where real-world data is expensive but human monitoring is feasible. The ability to surpass expert performance means the policy could learn locomotion strategies better than what a human teleoperator demonstrates. However, the high-frequency control (500 Hz PD control) means interventions must be at a higher level (velocity/trajectory commands) rather than joint-level.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
The IL-as-RL framework is applicable at multiple hierarchy levels. At the Planner level, human intervention during navigation could train better path planning policies. At the Primitives level, intervention when Cassie loses balance could provide reward signal for gait improvement. The expert-surpassing property is especially valuable—a non-expert human monitoring Cassie could provide sufficient intervention signal for the RL optimization to discover superior locomotion strategies. The approach could complement the adversarial curriculum by providing real-world reward signal from human judgment. The off-policy replay buffer aligns with PPO training infrastructure already planned.

## What to Borrow / Implement
- Implement intervention-based reward as supplementary signal during real-world fine-tuning for both projects
- Use RLIF for sim-to-real adaptation where human monitors flag failure modes
- Apply at the Planner level of Project B for human-guided navigation refinement
- Combine with domain randomization: intervention signal identifies sim-to-real gaps
- Use the confidence-based scheduling to reduce human monitoring burden over time

## Limitations & Open Questions
- Requires human availability for interactive sessions, which may be impractical for continuous training
- Binary intervention signal is coarse—doesn't capture degree of error
- Latency between human observation and intervention may corrupt the reward signal
- Not demonstrated on locomotion tasks; extrapolation from manipulation is uncertain
- Open question: How to handle intervention signals at different hierarchy levels simultaneously?
- Real-time intervention for dynamic locomotion (running, jumping) may have unacceptable latency
- Unclear how intervention-based reward interacts with existing reward shaping from simulation
