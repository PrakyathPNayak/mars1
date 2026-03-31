# CHG-DAgger: Interactive Imitation Learning with Human-Policy Cooperative Control

**Authors:** Various
**Year:** 2024 | **Venue:** CoRL Workshop 2024
**Links:** https://openreview.net/pdf?id=nWiHRy3ZXy

---

## Abstract Summary
CHG-DAgger extends the DAgger framework with a cooperative human-guided control scheme where both the expert and the learned policy share control authority simultaneously. This progressive handoff mechanism reduces human intervention time while increasing visuomotor policy robustness. The method demonstrates faster error correction and more data-efficient learning compared to standard DAgger variants.

## Core Contributions
- Extends DAgger with cooperative human-policy control sharing during data collection
- Introduces a progressive handoff mechanism that gradually transfers authority from expert to policy
- Reduces required human intervention time compared to standard DAgger
- Improves data efficiency through targeted expert corrections on difficult states
- Demonstrates more robust visuomotor policies through cooperative training
- Provides a principled framework for blending expert and policy actions during interactive learning
- Shows faster convergence to competent policies compared to BC and vanilla DAgger

## Methodology Deep-Dive
Standard DAgger collects data by rolling out the learned policy and querying the expert for the correct action at each visited state. CHG-DAgger modifies this by allowing simultaneous execution of both expert and policy actions through a blending function. The blending coefficient α(t) controls the mix: α=1 means full expert control, α=0 means full policy control. During training, α decreases over rounds, progressively transferring control authority from expert to policy.

The cooperative control mechanism works as follows: at each timestep, both the expert and the policy propose actions. The executed action is a weighted combination: a_exec = α · a_expert + (1-α) · a_policy. When the policy makes errors, the expert's influence corrects the trajectory in real-time, preventing cascading failures. This is more graceful than DAgger's binary switch between expert and policy rollouts. The blending also means the expert only needs to provide coarse corrections rather than precise demonstrations, reducing cognitive load.

The human intervention signal is used adaptively. When the policy is performing well (low divergence from expert), the blending coefficient decreases faster. When the policy struggles (high divergence), the coefficient stays high, keeping more expert influence. This adaptive schedule is computed using a running estimate of the policy's performance on recent states, creating a curriculum-like effect where the policy gradually handles more challenging situations.

Data aggregation follows the DAgger principle but with richer labels. Each collected datapoint includes: the blended action executed, the pure expert action, the pure policy action, and the blending coefficient. This additional information can be used for more sophisticated loss functions that weight examples by their difficulty (higher α indicates harder states).

The visuomotor policy architecture uses a CNN for visual encoding and an MLP for action prediction. The visual observations are processed at a lower frequency than proprioceptive inputs, with a recurrent component to handle partial observability. The policy is trained on the aggregated dataset using a weighted BC loss that emphasizes high-α (difficult) states.

## Key Results & Numbers
- Reduces human intervention time by 40-60% compared to standard DAgger
- Achieves comparable or better policy performance with 30% less demonstration data
- Faster convergence: reaches 90% expert performance in 3 rounds vs. 5+ for vanilla DAgger
- Progressive handoff prevents catastrophic failures during data collection
- Robust to suboptimal expert corrections due to the blending mechanism
- Evaluated primarily on manipulation tasks with implications for locomotion

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
CHG-DAgger's cooperative control framework could be applied to fine-tuning Mini Cheetah locomotion policies after initial PPO training in MuJoCo. If human teleoperators can provide coarse corrections during real-world deployment, the progressive handoff would allow gradual policy refinement. The adaptive blending coefficient maps naturally to a curriculum where the policy handles easy terrain independently but receives human guidance on challenging terrain. The data efficiency gains are valuable since real-world Mini Cheetah data is expensive. However, the 500 Hz control frequency may make real-time human cooperation challenging at the joint level—it would need to operate at a higher abstraction (velocity commands).

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
The cooperative control framework could be applied at the Planner or Primitives level of Cassie's hierarchy, where the control frequency is lower and human input is more feasible. At the Planner level, a human could cooperatively guide path selection while the learned planner handles details. The progressive handoff aligns with curriculum learning—starting with heavy human guidance on challenging terrains and gradually reducing to autonomous operation. The DAgger-based refinement could fine-tune individual locomotion primitives using expert demonstrations. The adaptive blending coefficient could be useful during the transition from simulation to real-world deployment.

## What to Borrow / Implement
- Apply cooperative control at the Planner level for human-guided fine-tuning of Cassie navigation
- Use adaptive blending coefficient as a curriculum mechanism during sim-to-real transfer
- Implement weighted BC loss that emphasizes difficult states for policy refinement
- Adapt the progressive handoff for transitioning from teleop to autonomous locomotion
- Consider coarse velocity-level cooperative control for Mini Cheetah real-world refinement

## Limitations & Open Questions
- Human cooperation at joint-level control (500 Hz) is infeasible; must operate at higher abstraction
- Blending coefficient schedule requires tuning per task and may not generalize
- Primarily evaluated on manipulation; locomotion applicability is extrapolated, not demonstrated
- Expert cognitive load during cooperative control of dynamic locomotion is not studied
- Open question: How to handle the latency of human input in high-frequency locomotion control loops?
- Multi-skill cooperative control (multiple gait types) not addressed
