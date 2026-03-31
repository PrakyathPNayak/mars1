# Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2510.26067)

---

## Abstract Summary
This paper proposes a morphology-aware graph reinforcement learning framework specifically designed for tensegrity robot locomotion. Tensegrity robots, characterized by their compliant structures composed of rigid struts and elastic cables, present unique control challenges due to their high-dimensional, nonlinear dynamics and the tight coupling between structural morphology and behavior. The authors represent the robot's physical structure as a graph, where each physical component (strut, cable, node) corresponds to a graph node, and physical connections define edges.

The framework integrates Graph Neural Networks (GNNs) with Soft Actor-Critic (SAC) reinforcement learning to learn locomotion policies that respect the robot's morphological structure. By encoding both local and global physical couplings through message-passing operations, the GNN-based policy achieves significantly better sample efficiency compared to standard MLP-based policies. The approach is validated in simulation and demonstrated with sim-to-real transfer on physical tensegrity robots, confirming that morphology-aware graph representations yield more robust and transferable policies.

The key insight is that by building the robot's physical topology directly into the policy architecture, the learning algorithm can exploit structural priors—such as locality of actuation effects and symmetry of repeated structural motifs—to accelerate learning and improve generalization.

## Core Contributions
- Formulation of tensegrity robot morphology as a graph structure where nodes represent physical components and edges encode mechanical couplings
- Integration of GNN-based policy networks with SAC reinforcement learning for continuous control of compliant robots
- Demonstration that morphology-aware GNN policies are substantially more sample-efficient than flat MLP policies on tensegrity locomotion tasks
- Message-passing mechanism that captures both local physical interactions (adjacent strut-cable forces) and global structural coupling (propagated through multiple hops)
- Sim-to-real transfer validation on physical tensegrity hardware, confirming that graph-structured policies maintain performance across the reality gap
- Analysis showing that the inductive bias from graph structure improves robustness to perturbations and morphological variations
- Open framework applicable to other robots whose morphology can be naturally represented as a graph (e.g., legged robots, modular robots)

## Methodology Deep-Dive
The robot's morphology is encoded as an undirected graph G = (V, E), where vertices V represent physical components (rigid struts, cable attachment points, actuators) and edges E represent mechanical connections (cable links, rigid joints). Each node is assigned a feature vector containing local state information such as position, velocity, strain, and actuation state. Edge features encode coupling properties like stiffness coefficients, rest lengths, and damping parameters.

The GNN policy network uses a multi-round message-passing scheme. In each round, every node aggregates information from its neighbors through learned message functions, then updates its hidden state via a learned update function. After K rounds of message passing (typically K = 3–4), each node's hidden state encodes information from its K-hop neighborhood, capturing both local physical interactions and broader structural context. The final policy output is obtained by reading out action values from actuator nodes.

The RL component uses Soft Actor-Critic (SAC) with automatic entropy tuning. SAC's maximum entropy objective encourages exploration and produces robust policies that can handle the multimodal action distributions common in tensegrity control. The GNN policy serves as both the actor and the critic, with the critic using the same graph structure but a separate set of learned parameters. State inputs are decomposed per-node rather than concatenated into a flat vector, preserving structural information throughout the learning pipeline.

Training is conducted in MuJoCo-based simulation with domain randomization over cable stiffness, damping coefficients, friction, and mass distributions. The curriculum progresses from flat terrain to increasingly rough surfaces. The sim-to-real pipeline exports the trained GNN policy and deploys it on-board the physical tensegrity robot at approximately 50 Hz control frequency.

The authors compare against several baselines: standard MLP-SAC, MLP-PPO, and a manually designed graph policy without learned message functions. The morphology-aware GNN consistently outperforms all baselines in both learning speed (2–5× fewer environment steps to convergence) and final locomotion performance (velocity, stability, energy efficiency).

## Key Results & Numbers
- GNN-SAC achieves 2–5× better sample efficiency compared to MLP-SAC on tensegrity locomotion benchmarks
- Final locomotion velocity improved by ~30% over MLP baselines on rough terrain tasks
- Sim-to-real transfer success rate maintained at high levels with domain randomization
- Message-passing rounds K = 3 found optimal; K = 1 (local-only) significantly worse, K > 4 shows diminishing returns
- GNN policy generalizes to morphological variations (±15% cable stiffness changes) without retraining
- Control frequency of ~50 Hz achieved on embedded hardware for real-time deployment
- Energy efficiency (cost of transport) improved by ~20% compared to MLP policies

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper directly validates the use of GNN-based policy architectures for locomotion control of physically structured robots. The Mini Cheetah's kinematic tree—with its 4 legs, 3 joints per leg, and 12 DoF—is a natural graph. Encoding the Mini Cheetah's morphology as a graph and using message-passing to propagate information between adjacent joints and links could provide the same sample efficiency and generalization benefits demonstrated here. The domain randomization and curriculum learning strategies used in the tensegrity setting are directly applicable to the Mini Cheetah MuJoCo pipeline.

The sim-to-real transfer methodology is particularly relevant, as the Mini Cheetah project also targets real-world deployment. The finding that morphology-aware policies are more robust to physical parameter variations suggests that a GNN-based Mini Cheetah policy could transfer more reliably to hardware than a standard MLP policy.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This paper provides foundational validation for the MC-GAT (Morphology-Conditioned Graph Attention) approach central to Cassie's architecture. The demonstration that encoding robot morphology as a graph and using message-passing (here with GNN, in Cassie's case with GATv2) significantly improves learning efficiency and policy quality directly supports the MC-GAT design decision. The tensegrity setting, with its complex physical couplings, is arguably harder than legged robot control, making these results a strong positive signal.

The multi-hop message passing finding (K = 3 optimal) is directly informative for configuring GATv2 layers in Cassie's MC-GAT module. The graph-structured critic architecture could also be adopted for Cassie's PPO critic. The sim-to-real pipeline with domain randomization provides a template for Cassie's deployment workflow.

## What to Borrow / Implement
- Graph construction methodology: map Mini Cheetah/Cassie kinematic tree to graph with joint/link nodes and kinematic edges
- Multi-round message passing with K = 3–4 rounds as starting point for GATv2 layer depth in MC-GAT
- Per-node feature decomposition instead of flat state vector concatenation for both actor and critic
- Domain randomization strategy over physical parameters (stiffness, damping, mass) for sim-to-real robustness
- Graph-structured critic architecture to complement the graph-structured actor in PPO

## Limitations & Open Questions
- Tensegrity robots have fundamentally different dynamics (cable-driven, compliant) from rigid-body legged robots—results may not transfer directly
- SAC used here vs PPO in both projects; the GNN benefit may interact differently with on-policy vs off-policy algorithms
- Scalability of message-passing to larger/more complex morphologies (e.g., Cassie's full 20+ DoF kinematic tree) not explicitly tested
- No comparison with attention-based graph networks (GATv2), which are the specific architecture chosen for Cassie's MC-GAT
