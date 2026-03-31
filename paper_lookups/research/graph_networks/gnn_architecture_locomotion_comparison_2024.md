# Exploring Graph Neural Networks in Reinforcement Learning: A Comparative Study on Architectures for Locomotion Tasks

**Authors:** (2024)
**Year:** 2024 | **Venue:** UTRGV Thesis
**Links:** [ScholarWorks](https://scholarworks.utrgv.edu/etd/1493/)

---

## Abstract Summary
This thesis presents a systematic comparative study of graph neural network architectures applied to reinforcement learning for locomotion tasks. The work evaluates multiple GNN variants—including Graph Convolutional Networks (GCN), GraphSAGE, Graph Attention Networks (GAT/GATv2), and Graph Isomorphism Networks (GIN)—as policy and value function approximators in RL agents controlling articulated robots. The central research question is whether and how different GNN architectures capture the spatial and relational dependencies inherent in robot morphologies, and how this impacts learning efficiency and final locomotion performance.

The study uses standard MuJoCo locomotion benchmarks (Ant, HalfCheetah, Humanoid, Walker2d) with robot morphologies represented as graphs derived from their kinematic trees. Each GNN variant is integrated with PPO as the RL algorithm, and comparisons are made against MLP baselines. The thesis examines both static graph architectures (where the graph structure is fixed based on the robot's kinematic tree) and dynamic graph architectures (where edges are added or modified based on spatial proximity or learned attention).

Key findings include: (1) GNN-based policies consistently outperform MLPs on complex morphologies (Ant, Humanoid) but show smaller advantages on simpler ones (HalfCheetah); (2) attention-based architectures (GAT, GATv2) provide the best trade-off between performance and adaptability; (3) static kinematic-tree graphs outperform dynamic proximity-based graphs for locomotion; and (4) the depth of message passing (number of GNN layers) should match the diameter of the kinematic tree for optimal performance.

## Core Contributions
- Systematic comparison of GCN, GraphSAGE, GAT, GATv2, and GIN architectures for RL locomotion control
- Analysis of static vs dynamic graph construction strategies for articulated robot morphologies
- Demonstration that attention-based GNNs (GAT/GATv2) achieve the best performance-adaptability trade-off
- Finding that GNN benefits increase with morphological complexity: larger gains on Ant and Humanoid than HalfCheetah
- Message-passing depth analysis: optimal number of GNN layers matches kinematic tree diameter
- Comparison of per-node action output vs aggregated global action output for GNN policies
- Comprehensive ablation studies on hidden dimensions, attention heads, aggregation functions, and residual connections

## Methodology Deep-Dive
The graph representation maps each joint in the robot's URDF kinematic tree to a node, with edges connecting parent-child joints. Node features consist of the joint's local state: joint angle, joint velocity, and the joint's contribution to the global state (e.g., its Cartesian position relative to the root). For the root node, additional features include the base orientation (quaternion), base linear and angular velocities, and any external sensor readings (contact booleans). Edge features (where supported by the GNN variant) encode the relative position and orientation between connected joints.

Five GNN architectures are evaluated:

**GCN (Graph Convolutional Network):** Spectral-based convolution with normalized adjacency matrix. Node features are updated as h_v^{l+1} = σ(Σ_{u∈N(v)} (1/√(d_u · d_v)) · W^l · h_u^l), where d denotes node degree. Simple and computationally efficient but treats all neighbors equally.

**GraphSAGE:** Inductive learning framework that samples and aggregates neighbor features. Uses mean, max, or LSTM aggregators. More scalable than GCN for larger graphs and supports different aggregation strategies.

**GAT (Graph Attention Network v1):** Computes attention coefficients α_{vu} = softmax(LeakyReLU(a^T · [Wh_v || Wh_u])) that weight neighbor contributions. Allows the network to learn which neighbors are most informative for each node.

**GATv2 (Graph Attention Network v2):** Addresses the limited expressivity of GAT by using a modified attention mechanism: α_{vu} = softmax(a^T · LeakyReLU(W · [h_v || h_u])). This computes a dynamic, context-dependent attention that can distinguish between any pair of nodes, unlike GATv1 which computes a static ranking.

**GIN (Graph Isomorphism Network):** Based on the Weisfeiler-Leman graph isomorphism test. Updates node features as h_v^{l+1} = MLP((1 + ε) · h_v^l + Σ_{u∈N(v)} h_u^l). Maximally expressive among message-passing GNNs but may overfit on small graphs.

Each GNN variant is tested with 2, 3, and 4 layers, hidden dimensions of 64, 128, and 256, and for attention-based models, 1, 2, 4, and 8 attention heads. The policy network outputs per-node actions (each node outputs the action for its corresponding joint), while the value network uses a global readout (mean pooling over node embeddings) followed by a linear layer to produce the scalar state value.

PPO hyperparameters are held constant across all comparisons: learning rate 3e-4, γ = 0.99, λ = 0.95, clip ratio 0.2, 10 epochs, batch size 2048, entropy coefficient 0.005. Training runs for 10M environment steps with 5 random seeds per configuration.

The dynamic graph variant adds edges between joints that are spatially close (within a threshold distance) in addition to kinematic tree edges. This creates a graph that changes at each timestep as the robot's configuration changes, potentially capturing non-adjacent joint interactions (e.g., left and right feet during a double-support phase).

## Key Results & Numbers
- GATv2 achieves the highest average return across all environments, outperforming MLP by 15–35% on Ant and Humanoid
- GATv2 outperforms GATv1 by 5–10% on Humanoid, confirming the value of dynamic attention in complex morphologies
- GCN and GraphSAGE perform 5–15% better than MLP but 10–20% below GATv2 on complex morphologies
- GIN shows strong performance but is prone to overfitting on simpler morphologies (HalfCheetah, Walker2d)
- On HalfCheetah (simple, serial morphology), GNN advantage over MLP is marginal (<5%)
- Optimal GNN depth: 2 layers for HalfCheetah/Walker2d (tree diameter 2), 3 layers for Ant (diameter 3), 3–4 layers for Humanoid (diameter 4)
- Static kinematic-tree graphs outperform dynamic proximity-based graphs by 8–12% on average
- 4 attention heads provide the best results for GATv2; 8 heads show diminishing returns
- Per-node action output outperforms global aggregated output by 10–20% on multi-limbed robots (Ant, Humanoid)
- Sample efficiency: GATv2 reaches 80% of final MLP performance in 40–60% of the training steps on Ant/Humanoid

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The comparison results provide evidence-based guidance for choosing a GNN architecture for Mini Cheetah's policy network. The finding that GATv2 outperforms other GNN variants on locomotion tasks supports using attention-based GNNs for Mini Cheetah. The recommendation to match GNN depth to kinematic tree diameter (3 layers for a quadruped) and use 4 attention heads provides specific architectural parameters. However, Mini Cheetah's relatively simple morphology (compared to Humanoid) means the GNN advantage may be moderate rather than dramatic.

The per-node action output finding is directly applicable: each joint node in Mini Cheetah's kinematic graph should output its own action, rather than aggregating all node features into a global action vector.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This thesis provides the most direct empirical justification for the MC-GAT design choice of using GATv2 on Cassie's kinematic tree. The finding that GATv2 outperforms all other GNN variants on the Humanoid benchmark (morphologically closest to Cassie among the tested environments) directly validates using GATv2 for Cassie's policy. The specific architectural recommendations—3–4 GATv2 layers for Humanoid-class morphologies, 4 attention heads, 128-dimensional hidden features, per-node action output—can be directly adopted for MC-GAT.

The comparison between static and dynamic graphs is also informative: the result that static kinematic-tree graphs outperform dynamic proximity-based graphs suggests that MC-GAT should use Cassie's fixed kinematic tree as the graph structure rather than distance-based edge construction. The sample efficiency improvement (40–60% fewer steps to 80% performance) is significant for Cassie's complex training pipeline where each training run is expensive.

## What to Borrow / Implement
- GATv2 with 4 attention heads and 128-dimensional hidden features as the specific MC-GAT configuration
- 3–4 GATv2 layers matching Cassie's kinematic tree diameter for full information propagation
- Per-node action output: each joint node outputs its own action for decentralized, morphology-aware control
- Static kinematic-tree graph structure (not dynamic proximity-based) for stable, efficient training
- Residual connections between GNN layers to prevent over-smoothing in deeper networks

## Limitations & Open Questions
- Benchmark environments (MuJoCo Ant, Humanoid) are simpler than real robot control; results may not fully transfer to Mini Cheetah/Cassie complexity
- No hierarchical RL evaluation; all experiments use flat PPO, leaving open whether GATv2 benefits persist in hierarchical architectures
- No domain randomization or sim-to-real considerations; GNN robustness to parameter variation untested
- Single-task evaluation (forward locomotion); no assessment of GNN architecture impact on multi-skill or curriculum learning scenarios
