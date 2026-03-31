# Reinforcement Learning in Robotic Systems: A Review on Sim-to-Real Transfer Strategies

**Authors:** Various
**Year:** 2025 | **Venue:** Robotics and Autonomous Systems (Elsevier)
**Links:** https://www.sciencedirect.com/science/article/pii/S0921889025004245

---

## Abstract Summary
Comprehensive survey of sim-to-real transfer methods for RL in robotics, covering domain randomization, system identification, domain adaptation, meta-learning, and online adaptation. Compares techniques across legged locomotion, manipulation, and navigation, identifying best practices and open challenges. The survey provides a structured taxonomy and practical guidelines for practitioners designing sim-to-real transfer pipelines.

## Core Contributions
- Provides a comprehensive taxonomy of sim-to-real transfer methods organized by approach type and application domain
- Systematically compares domain randomization, system identification, domain adaptation, meta-learning, and online adaptation across multiple robotic tasks
- Identifies hybrid approaches combining multiple transfer strategies as consistently most effective
- Surveys application-specific insights for legged locomotion, manipulation, and navigation
- Analyzes the trade-offs between sample efficiency, robustness, and performance for each method class
- Highlights open challenges including sim-to-real for deformable objects, long-horizon tasks, and multi-robot systems
- Provides decision flowcharts for practitioners selecting appropriate transfer strategies

## Methodology Deep-Dive
The survey organizes sim-to-real transfer methods into five primary categories. Domain Randomization (DR) involves training policies across a distribution of simulation parameters to achieve robustness to real-world variation. The survey distinguishes between uniform DR, structured DR (physically plausible ranges), adaptive DR (automatically adjusting ranges during training), and visual DR (randomizing rendering for vision-based policies). Key findings indicate that adaptive DR methods like ADR and BayesSim consistently outperform uniform randomization.

System Identification (SysID) focuses on accurately matching simulation to reality by identifying physical parameters. The survey covers classical SysID (least squares, maximum likelihood), Bayesian SysID (posterior estimation over parameters), and neural SysID (learning residual dynamics models). A key insight is that SysID and DR are complementary—SysID narrows the parameter space while DR provides robustness within that space.

Domain Adaptation (DA) methods learn to map between source (simulation) and target (real) domains. The survey covers feature-level adaptation (learning domain-invariant representations), pixel-level adaptation (GAN-based visual translation), and dynamics-level adaptation (learning correction models). For locomotion tasks, dynamics-level adaptation is most relevant and is often implemented as learned residual models that correct simulation predictions.

Meta-learning approaches train policies that can rapidly adapt to new environments with minimal data. The survey reviews MAML-based methods, context-based meta-learning, and task-inference methods. For legged locomotion, context-based approaches that infer environment parameters from interaction history are most practical.

Online adaptation methods continue policy improvement during real-world deployment. The survey covers online fine-tuning, Bayesian optimization, and rapid motor adaptation (RMA). RMA-style approaches that use an adaptation module to estimate environment parameters from recent history are highlighted as particularly effective for locomotion.

## Key Results & Numbers
- Hybrid approaches (SysID + DR + adaptation) achieve 15-30% higher transfer success than any single method
- Adaptive DR outperforms uniform DR by 10-20% on locomotion tasks
- RMA-style online adaptation improves performance by 20-40% within first 1000 real-world steps
- Meta-learning methods require 10-100x less real-world data for adaptation compared to fine-tuning
- Vision-based transfer remains 20-30% less reliable than proprioceptive-based transfer for locomotion
- Domain adaptation most effective when sim-real gap is moderate; fails for large distributional shifts
- Survey covers 200+ papers spanning 2017-2025, with focus on post-2020 advances

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This survey provides the definitive reference for designing Mini Cheetah's sim-to-real pipeline. The decision flowcharts can guide selection of appropriate transfer strategies based on available resources (robot access time, computation budget, acceptable deployment risk). The finding that hybrid approaches are most effective supports combining system identification for Mini Cheetah's motors with targeted domain randomization and RMA-style online adaptation. The proprioceptive-based transfer findings are directly relevant since Mini Cheetah primarily relies on joint encoders and IMU. The PPO-specific transfer insights align with the project's training algorithm choice.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The survey's coverage of transfer strategies across different system complexities is invaluable for Project B's 4-level hierarchy. Different levels may benefit from different transfer strategies: the Planner level may use meta-learning for rapid terrain adaptation, the Primitives level may use DR for robust skill execution, the Controller level may benefit from accurate SysID, and the Safety level requires the highest-fidelity transfer for reliable constraint satisfaction. The survey's analysis of POMDP-aware transfer methods is relevant for the Dual Asymmetric-Context Transformer architecture. The discussion of multi-modal adaptation connects to the CPTE (contrastive terrain encoder) design.

## What to Borrow / Implement
- Use the decision flowcharts to systematically design sim-to-real pipelines for both projects
- Implement hybrid transfer: SysID for actuators + adaptive DR for environment + RMA for online adaptation
- Apply the survey's best-practice guidelines for PPO-based sim-to-real transfer
- Use the taxonomy to identify which transfer method is most appropriate for each hierarchy level in Project B
- Leverage the comparative analysis to justify design choices in publications and reports
- Adopt the survey's evaluation metrics for quantifying sim-to-real gap in experiments

## Limitations & Open Questions
- Survey breadth may sacrifice depth on specific techniques; original papers should be consulted for implementation details
- Comparative analysis across papers is inherently limited by different experimental setups and metrics
- Coverage of very recent 2025 methods may be incomplete due to publication timing
- Limited discussion of sim-to-real for high-frequency control loops (>500 Hz)
- Open question: how to optimally allocate computational resources between simulation fidelity and policy robustness
- Open question: how to certify or guarantee safety of transferred policies in safety-critical applications
