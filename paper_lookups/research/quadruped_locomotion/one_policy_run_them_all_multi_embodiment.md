# One Policy to Run Them All: An End-to-end Learning Approach to Multi-Embodiment Locomotion

**Authors:** Various
**Year:** 2024 | **Venue:** CoRL 2024
**Links:** https://openreview.net/forum?id=PbQOZntuXO

---

## Abstract Summary
This paper introduces a unified, multi-embodiment RL controller capable of robust locomotion across diverse robot morphologies — quadrupeds, bipeds, and hexapods — with successful transfer to unseen morphologies. It uses a morphology-agnostic architecture that encodes robot structure implicitly, hinting at "foundation models" for locomotion. The approach demonstrates that a single policy can match or exceed the performance of morphology-specific controllers.

## Core Contributions
- A single locomotion policy that controls quadrupeds, bipeds, and hexapods without morphology-specific tuning
- Morphology-agnostic network architecture that implicitly encodes robot structure through observation and action space design
- Zero-shot transfer to unseen robot morphologies not present during training
- Competitive or superior performance compared to morphology-specific baselines across diverse platforms
- Demonstration of "foundation model" principles applied to locomotion control
- Scalable training methodology that benefits from diversity of training morphologies
- Evidence that cross-morphology training acts as implicit regularization, improving individual robot performance

## Methodology Deep-Dive
The core insight is that locomotion control across morphologies shares fundamental principles — balance, periodic limb coordination, energy efficiency, and terrain adaptation — that can be captured by a single policy. Rather than training separate controllers for each robot, the authors design an architecture that processes morphology information alongside standard proprioceptive observations.

The observation space is carefully designed to be morphology-agnostic. Instead of a fixed-size vector tied to a specific robot, the architecture uses a per-joint representation that scales naturally with the number of limbs and joints. Each joint receives local proprioceptive information (position, velocity, torque) along with relative spatial information about its position in the kinematic chain. This per-joint representation is processed through shared weight networks, enabling the same parameters to be applied regardless of the number of joints.

Global body information (base orientation, angular velocity, linear velocity, gravity vector) is broadcast to all joint-level representations, providing each joint with context about the whole-body state. Command inputs (desired velocity, heading) are similarly broadcast. The policy outputs per-joint actions that are interpreted as target positions for PD controllers, maintaining compatibility across different actuator configurations.

Training uses massively parallel simulation with diverse robot morphologies sampled simultaneously. The training set includes multiple quadrupeds (varying sizes and proportions), bipeds, and hexapods. Domain randomization is applied per-instance, with each robot experiencing different physical parameters. The diversity of morphologies during training acts as a strong regularizer, preventing overfitting to any single robot's dynamics and encouraging the learning of general locomotion principles.

Transfer to unseen morphologies is evaluated by deploying the trained policy on robots with different proportions, masses, and limb configurations than those seen during training. The policy receives the new robot's proprioceptive information in the same per-joint format and must generalize its learned locomotion strategies to the novel body plan.

## Key Results & Numbers
- Single policy successfully controls quadrupeds, bipeds, and hexapods simultaneously
- Zero-shot transfer to unseen morphologies achieves competitive performance without fine-tuning
- Performance matches or exceeds morphology-specific baselines on individual robots
- Cross-morphology training improves robustness compared to single-morphology training
- Successful locomotion across flat terrain, slopes, and moderate rough terrain for all morphology types
- Training scales efficiently with the number of morphologies due to shared computation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
A foundation locomotion policy pre-trained on diverse morphologies could provide a powerful initialization for Mini Cheetah training. Rather than learning from scratch, the Mini Cheetah policy could start from a multi-embodiment checkpoint and fine-tune to its specific dynamics, potentially reducing training time and improving robustness. The per-joint observation architecture is directly compatible with Mini Cheetah's 12 DoF structure. The finding that multi-morphology training acts as regularization suggests that even if the goal is Mini Cheetah-only performance, including other morphologies during training could yield a more robust policy. The PD control interface matches Project A's 500 Hz PD control scheme.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The multi-embodiment approach validates morphology-agnostic architectures, directly relevant to Project B's graph-based embodiment encoding via MC-GAT (GATv2 on kinematic tree). This paper demonstrates that encoding morphology implicitly — rather than through hand-designed features — is sufficient for cross-morphology generalization, supporting the choice of graph neural networks over fixed architectures. The per-joint processing mirrors the message-passing in GATv2 where each node (joint) aggregates information from its kinematic neighbors. The finding that bipedal locomotion benefits from multi-morphology training is encouraging for Cassie, suggesting that including quadruped data could improve Cassie's policy. The zero-shot transfer capability aligns with Project B's sim-to-real goals.

## What to Borrow / Implement
- Adopt the per-joint observation representation for both projects' policy architectures
- Use multi-morphology pre-training as initialization for Project A (Mini Cheetah) and Project B (Cassie)
- Apply the morphology-agnostic design principles to validate Project B's MC-GAT architecture choices
- Investigate whether including quadruped data during Cassie training improves bipedal locomotion robustness
- Use the paper's evaluation methodology (cross-morphology transfer) to benchmark Project B's graph-based encodings
- Consider the broadcast mechanism for global body state as an alternative to concatenation in Project B's architecture

## Limitations & Open Questions
- Per-joint representation may lose important whole-body structural information that graph-based approaches (MC-GAT) could capture
- Unseen morphology transfer is limited to morphologies structurally similar to training set; radical changes may fail
- Does not address task-specific behaviors (manipulation, acrobatics) that may require morphology-specific strategies
- Scalability to very high-DoF systems (humanoids with hands) is unclear
- How does this approach compare to explicit graph-based morphology encodings (GNN, attention on kinematic tree)?
- Limited analysis of failure modes when transferring to morphologies far from the training distribution
