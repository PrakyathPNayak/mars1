# Enhanced Exploration in Reinforcement Learning Using Graph Neural Networks

**Authors:** (2025)
**Year:** 2025 | **Venue:** Nature Scientific Reports
**Links:** [Nature](https://www.nature.com/articles/s41598-025-23769-3)

---

## Abstract Summary
This paper introduces a GNN-based approach to enhancing exploration in reinforcement learning by constructing a graph representation of the state space and using it to compute intrinsic reward signals. The key observation is that the structure of the state-transition graph contains valuable information about which regions of the state space are well-explored and which remain novel. By representing visited states as nodes and transitions as edges, and then applying GNN operations to this graph, the method computes exploration bonuses that are geometrically informed by the global structure of the explored region.

The GNN processes the state-transition graph to produce per-node embeddings that capture each state's position within the broader exploration landscape. States in well-explored, densely connected regions receive low intrinsic rewards, while states on the frontier—near unexplored regions, connected to few other visited states, or structurally distinct from known clusters—receive high intrinsic rewards. This approach contrasts with count-based or prediction-error-based exploration methods by leveraging the relational structure of the state space.

Experiments across continuous control benchmarks, including locomotion tasks, demonstrate that GNN-guided exploration achieves significantly better state-space coverage and finds higher-performing policies faster than standard exploration strategies (ε-greedy, entropy bonus, RND, ICM). The approach is particularly effective in environments with deceptive rewards or sparse reward signals.

## Core Contributions
- Graph representation of the RL state-transition space where visited states are nodes and experienced transitions are edges
- GNN-based intrinsic reward computation that leverages global state-space structure for exploration guidance
- Frontier detection via graph topology: states on the boundary of explored regions are identified through low-degree nodes and structural novelty
- Superior state-space coverage compared to count-based, prediction-error-based (RND, ICM), and entropy-based exploration methods
- Improved sample efficiency on continuous control benchmarks, especially in sparse-reward and deceptive-reward environments
- Scalable graph construction using state-space discretization and approximate nearest-neighbor graph building
- Analysis of exploration patterns showing that GNN-guided agents systematically expand the exploration frontier rather than revisiting known regions

## Methodology Deep-Dive
The state-transition graph is constructed incrementally during training. As the agent visits new states, they are added as nodes with feature vectors equal to the state representation. Edges are added between states that are reachable through single-step transitions. To keep the graph manageable, the continuous state space is discretized using a learned state encoder that maps raw observations to a lower-dimensional embedding space, followed by locality-sensitive hashing (LSH) to group similar states into discrete bins. Each bin becomes a node, and transitions between bins become edges.

The GNN operates on this discretized state-transition graph. It uses a 3-layer Graph Attention Network (GAT) with multi-head attention (4 heads, 64-dimensional per head). Each node's initial feature vector is the centroid of its state-space bin in the embedding space. After message passing, each node's output embedding captures its structural role in the graph: hub nodes in densely connected regions produce embeddings similar to their neighbors, while frontier nodes produce embeddings dissimilar from the rest.

The intrinsic reward for visiting a state is computed from the GNN output embedding. Specifically, the intrinsic reward r_i(s) is proportional to the distance between the state's GNN embedding and the mean embedding of its K-nearest neighbors in the graph. States whose embeddings are far from their neighbors (structurally novel or on the frontier) receive high intrinsic rewards. An exponential decay factor reduces the intrinsic reward for states that have been visited many times, combining the structural signal with a count-based decay.

The total reward combines the extrinsic environment reward with the GNN-computed intrinsic reward, weighted by a coefficient β that is annealed during training (high β early for exploration, low β later for exploitation). The RL algorithm (PPO or SAC) optimizes the combined reward. The state-transition graph and GNN are updated periodically (every N environment steps) to keep computational costs manageable.

The graph construction and GNN inference are batched and parallelized on GPU. The LSH-based discretization ensures that graph size grows sub-linearly with the number of visited states, keeping memory and computation feasible for long training runs. The GNN forward pass on the state-transition graph takes approximately 10ms for graphs with 10K nodes on a single GPU.

## Key Results & Numbers
- 30–50% better state-space coverage compared to RND and ICM on continuous control exploration benchmarks
- 2–3× faster convergence to high-performing policies on sparse-reward locomotion tasks
- On deceptive reward environments, GNN-guided exploration finds the global optimum in 80% of runs vs 20% for RND
- Intrinsic reward computation adds <5% overhead to total training time (graph update every 1000 steps)
- GAT architecture outperforms GCN and GraphSAGE variants by 10–15% on exploration metrics
- Scalable to state-transition graphs with up to 100K nodes without significant performance degradation
- Ablation: removing structural intrinsic reward (using only count-based decay) reduces performance by 20–30%

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
GNN-guided exploration could enhance the Mini Cheetah training pipeline by encouraging the discovery of diverse locomotion behaviors early in training. In quadruped RL, the policy can easily converge to local optima (e.g., a shuffling gait) without discovering more efficient gaits (trot, gallop). The structural intrinsic reward could push the agent to explore dynamically distinct movement patterns by rewarding states that are topologically novel in the state-transition graph.

This is particularly relevant for curriculum learning: the intrinsic exploration signal could complement the extrinsic curriculum reward, ensuring that the agent doesn't exploit curriculum loopholes but instead genuinely explores the space of locomotion behaviors.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The GNN-based intrinsic reward mechanism is highly relevant to Cassie's skill discovery pipeline using DIAYN/DADS. Currently, DIAYN discovers skills by maximizing mutual information between skills and states, while DADS additionally considers dynamics-aware skill learning. The GNN-computed exploration bonus could serve as an additional intrinsic reward signal during skill discovery, encouraging the Primitives level to discover skills that access structurally novel regions of the state space.

The state-transition graph representation could also inform the Planner level (RSSM-based world model) by providing a discrete abstraction of the reachable state space. The Planner could use the graph structure to reason about which skill sequences lead to unexplored but potentially valuable regions, improving long-horizon planning efficiency.

## What to Borrow / Implement
- State-transition graph construction with LSH-based discretization for efficient state-space exploration tracking
- GAT-based intrinsic reward computation for encouraging diverse behavior discovery during DIAYN/DADS skill learning
- Frontier detection via graph topology as an additional signal for curriculum learning progression
- Annealed intrinsic reward weighting (high β early, low β later) for balancing exploration and exploitation
- Graph attention mechanism (GAT with 4 heads) as a reference architecture compatible with GATv2 in MC-GAT

## Limitations & Open Questions
- State-space discretization via LSH introduces quantization artifacts; states near bin boundaries may be incorrectly grouped
- The approach adds complexity (graph construction, GNN inference) that may not be justified for dense-reward environments where standard exploration suffices
- Scaling to very high-dimensional state spaces (e.g., image-based observations) requires effective state encoding before graph construction
- Interaction between GNN intrinsic reward and other intrinsic reward sources (DIAYN mutual information, DADS dynamics prediction) is unexplored
