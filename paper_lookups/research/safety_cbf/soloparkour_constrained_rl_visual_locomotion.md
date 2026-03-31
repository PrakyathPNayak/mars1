# SoloParkour: Constrained Reinforcement Learning for Visual Locomotion from Privileged Experience Retargeting

**Authors:** Various (LAAS-CNRS / Gepetto team)
**Year:** 2024 | **Venue:** CoRL 2024
**Links:** https://openreview.net/forum?id=DSdAEsEGhE

---

## Abstract Summary
SoloParkour trains a quadruped (Solo-12) for parkour using constrained RL. The method first trains a privileged non-vision policy with access to full state information, then transfers skills to a visuomotor policy with explicit safety constraints. The constrained approach ensures safe deployment while maintaining agile parkour capabilities including jumping, climbing, and obstacle traversal.

## Core Contributions
- Introduces a two-stage training pipeline: privileged policy training followed by constrained visuomotor policy distillation
- Demonstrates constrained RL for agile locomotion tasks (parkour) where safety is critical during aggressive maneuvers
- Achieves real-world parkour deployment on the Solo-12 quadruped with explicit safety guarantees
- Develops privileged experience retargeting — transferring behaviors from a state-based expert to a vision-based student while maintaining constraints
- Shows that constrained RL can maintain both agility and safety simultaneously, avoiding the common trade-off
- Validates the approach across multiple parkour challenges: jumping gaps, climbing platforms, traversing obstacles

## Methodology Deep-Dive
The first stage trains a privileged teacher policy with full access to ground-truth state information including terrain geometry, contact forces, and exact robot state. This teacher policy is trained using RL with a rich reward function that encourages agile parkour behaviors. The privileged information allows the teacher to learn near-optimal strategies without the challenges of partial observability.

The second stage distills the teacher's behavior into a student visuomotor policy that only receives onboard camera images and proprioceptive feedback. This distillation is performed under explicit safety constraints using constrained RL methods. The constraints enforce joint limits, torque bounds, and stability margins during the transfer process, ensuring the student policy does not learn unsafe behaviors even when imitating the teacher's aggressive maneuvers.

The constrained RL formulation uses a Lagrangian approach with carefully tuned cost functions for each safety constraint. The key insight is that the privileged experience provides a strong behavioral prior, so the constrained optimization only needs to ensure safety rather than learning locomotion from scratch. This significantly simplifies the constrained optimization problem.

The vision system processes depth images to extract terrain information, which is combined with proprioceptive state in a neural network that outputs joint position targets. The network architecture includes a CNN for visual processing and an MLP for policy computation. The privileged retargeting ensures the visuomotor policy inherits the teacher's terrain-aware behaviors.

Real-world deployment uses the Solo-12 open-source quadruped robot with onboard computing and a depth camera. The constrained policy successfully performs parkour maneuvers while respecting hardware safety limits, demonstrating that the sim-to-real gap can be bridged even for aggressive locomotion tasks.

## Key Results & Numbers
- Successful real-world parkour on Solo-12 including jumping, climbing, and obstacle traversal
- Safety constraint satisfaction rate >95% during deployment
- Privileged-to-visual transfer maintains >85% of the teacher's task performance
- Constrained RL training converges within standard training budgets (comparable to unconstrained methods)
- Outperforms unconstrained baselines in safety metrics while matching agility performance
- Real-world deployment with zero catastrophic failures during testing sessions

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
SoloParkour's constrained RL framework for agile quadruped locomotion is directly applicable to Mini Cheetah. The two-stage privileged-to-visual transfer pipeline can be adopted for Mini Cheetah's MuJoCo training — first train with full state access, then distill to a deployable policy with safety constraints. The parkour-level agility demonstrated on Solo-12 (similar scale to Mini Cheetah) suggests the approach can handle the 12-DoF action space and 500 Hz control frequency. Domain randomization during the constrained distillation stage would further improve sim-to-real transfer.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The privileged experience retargeting directly parallels Project B's Dual Asymmetric-Context Transformer design, where the privileged critic has access to information unavailable to the actor. SoloParkour's constrained RL approach could inform how safety constraints are enforced during the distillation of privileged knowledge in Cassie's hierarchy. The two-stage pipeline maps onto Project B's architecture: the privileged teacher corresponds to the full-state planner, and the constrained student corresponds to the controller operating with limited observations. The safety constraint formulation is directly relevant to the LCBF layer.

## What to Borrow / Implement
- Adopt the two-stage privileged-to-constrained transfer pipeline for both projects
- Use constrained RL during policy distillation to maintain safety when transferring from privileged to deployment policies
- Apply the constraint formulation to enforce joint limits and torque bounds during Mini Cheetah training
- Integrate privileged experience retargeting with Project B's asymmetric actor-critic architecture
- Test the approach on progressively harder parkour challenges using curriculum learning
- Implement the vision-based policy distillation for depth camera integration in both platforms

## Limitations & Open Questions
- Solo-12 is lighter and has different dynamics than Mini Cheetah or Cassie — direct transfer of hyperparameters may not work
- Two-stage training adds complexity and potential information loss during distillation
- Constrained RL with Lagrangian methods still requires careful tuning of cost limits
- Vision-based policies may fail in conditions not seen during training (lighting, occlusions)
- The approach assumes the privileged teacher is near-optimal — poor teacher performance propagates to the student
- Scaling to bipedal platforms like Cassie with different balance requirements is an open question
