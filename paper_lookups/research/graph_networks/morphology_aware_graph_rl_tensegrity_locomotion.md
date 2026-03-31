---
## 📂 FOLDER: research/graph_networks/

### 📄 FILE: research/graph_networks/morphology_aware_graph_rl_tensegrity_locomotion.md

**Title:** Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion
**Authors:** Jiayu Wen, et al.
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2510.26067

**Abstract Summary (2–3 sentences):**
This paper presents a GNN-based RL framework that explicitly encodes the physical topology of tensegrity robots as a graph to learn locomotion policies. By integrating structural priors from the robot's morphology into the policy network via graph neural networks trained with SAC, the framework achieves superior noise robustness, trajectory accuracy, and sim-to-real transfer compared to standard MLP-based approaches. The work validates the principle that morphology-aware graph encoding significantly benefits both learning efficiency and real-world deployment.

**Core Contributions (bullet list, 4–7 items):**
- Proposes a morphology-aware GNN-RL framework that encodes tensegrity robot topology as a graph for policy learning
- Integrates the GNN policy architecture with Soft Actor-Critic (SAC) for off-policy training with maximum entropy exploration
- Demonstrates superior noise robustness compared to MLP baselines, critical for sim-to-real transfer
- Validates the approach on real tensegrity robots, showing successful sim-to-real transfer
- Provides detailed analysis of how graph topology encoding improves trajectory tracking accuracy
- Shows faster convergence and better sample efficiency than flat policy architectures
- Extends graph-based RL from articulated robots (NerveNet) to tensegrity robots with more complex connectivity patterns

**Methodology Deep-Dive (3–5 paragraphs):**
The framework begins by constructing a graph representation of the tensegrity robot's physical structure. Unlike articulated robots with tree-like kinematics, tensegrity robots have a more complex topology with struts (rigid compression elements) and cables (tension elements) forming a network. Each strut endpoint and cable attachment point becomes a node in the graph, and edges represent physical connections (strut rigidity or cable tension paths). Node features include local state information such as position, velocity, and strain measurements. Edge features encode physical properties like stiffness coefficients and rest lengths. This graph construction captures the unique structural dynamics of tensegrity systems where forces propagate through the entire network.

The GNN architecture uses a multi-layer message-passing network where each layer performs neighborhood aggregation. Specifically, each node computes messages from its connected neighbors using a learned message function (MLP), aggregates them via a permutation-invariant operation (sum or attention-weighted mean), and updates its hidden state through a GRU-style update function. The use of GRU-based updates (rather than simple MLP updates) helps maintain temporal consistency in the node representations across sequential policy queries. After multiple message-passing layers, the node representations capture both local dynamics and the propagation of forces through the tensegrity structure.

The GNN policy is integrated with SAC (Soft Actor-Critic) as the RL algorithm. SAC's maximum entropy framework is particularly well-suited here because tensegrity robots have highly nonlinear dynamics with many local optima — the entropy bonus encourages exploration of diverse locomotion strategies. The actor network is the GNN that takes the graph-structured observation and produces per-node actions (cable tension commands). The critic network receives the full state-action pair and estimates Q-values; it can be either an MLP (receiving flattened graph features) or another GNN. Training uses standard SAC with twin critics and automatic entropy coefficient tuning.

A key contribution is the noise robustness analysis. The authors systematically inject observation noise (Gaussian perturbations to sensor readings) and actuation noise (random torque offsets) during evaluation and show that the GNN policy degrades much more gracefully than MLP baselines. The hypothesis is that the graph structure acts as a natural denoising mechanism: because each node's action depends on aggregated information from its neighborhood, local noise at one sensor is smoothed out by the structural aggregation. This is particularly important for sim-to-real transfer, where observation noise and model mismatch are inevitable.

For sim-to-real transfer, the authors train entirely in simulation using a MuJoCo model of the tensegrity robot, then deploy directly on hardware. The sim-to-real gap is addressed through domain randomization (randomizing cable stiffness, damping, and friction) combined with the inherent noise robustness of the GNN architecture. The real-robot experiments demonstrate successful locomotion with trajectory tracking, validating that graph-structured policies provide a practical advantage for deployment. The authors compare against MLP policies transferred with the same domain randomization, showing that the GNN policy achieves tighter trajectory tracking and more reliable gait patterns on hardware.

**Key Results & Numbers:**
- Faster learning convergence: GNN policy reaches target performance ~40% faster than MLP baseline
- Better sim-to-real transfer: GNN policy maintains ~85% of simulation performance on real robot vs ~60% for MLP
- Superior noise robustness: performance degradation under 10% sensor noise is ~5% for GNN vs ~20% for MLP
- Demonstrated on real tensegrity robot hardware with successful locomotion
- Trajectory tracking error reduced by ~30% compared to MLP baseline in real-world deployment
- Graph topology ablation confirms that correct physical topology outperforms random and fully-connected graphs

**Relevance to Project A (Mini Cheetah):** MEDIUM — While the robot type differs (tensegrity vs articulated quadruped), the core principle — encoding physical topology as a graph for RL policy learning — is directly applicable. The noise robustness finding is particularly relevant for Mini Cheetah sim-to-real transfer, where sensor noise and model mismatch are major challenges. The SAC integration (vs PPO used in Mini Cheetah) provides an alternative off-policy perspective.

**Relevance to Project B (Cassie HRL):** HIGH — Strongly validates the MC-GAT approach used in Cassie's HRL system. The key findings — that graph-based RL with kinematic structure encoding improves robustness, sim-to-real transfer, and sample efficiency — directly support the design choice of using GATv2 on Cassie's kinematic tree. The noise robustness analysis is particularly relevant since Cassie must handle real-world sensor noise and terrain uncertainty. The GRU-based node update mechanism could be explored as an alternative to the attention-based updates in MC-GAT.

**What to Borrow / Implement:**
- Noise robustness analysis methodology: systematically test GNN vs MLP policies under varying noise levels — apply to both Mini Cheetah and Cassie evaluation
- GRU-based node update functions as an alternative to purely MLP-based updates in the MC-GAT
- Domain randomization combined with graph-structured policies for sim-to-real transfer — complementary to existing domain randomization strategies
- The principle that structural aggregation acts as natural denoising — use this to justify and design the MC-GAT aggregation scheme
- Edge feature encoding (stiffness, damping) to incorporate physical parameters into the graph — applicable to encoding joint properties in the kinematic tree

**Limitations & Open Questions:**
- Tensegrity robots have fundamentally different dynamics than articulated robots, so direct architecture transfer may require adaptation
- The SAC-based training may not be directly comparable to PPO-based training used in most locomotion work
- Limited morphology transfer evaluation — only tested on a single tensegrity robot design
- The paper does not explore hierarchical control or multi-level policy architectures
- Scalability to robots with many more nodes (high-DoF humanoids) is not evaluated
- The interplay between domain randomization and graph structure robustness is not fully disentangled
---
