---
## 📂 FOLDER: research/graph_networks/

### 📄 FILE: research/graph_networks/nervenet_learning_structured_policy_graph_neural_networks.md

**Title:** NerveNet: Learning Structured Policy with Graph Neural Networks
**Authors:** Tingwu Wang, Renjie Liao, Jimmy Ba, Sanja Fidler
**Year:** 2018
**Venue:** ICLR 2018
**arXiv / DOI:** arXiv:1809.05052

**Abstract Summary (2–3 sentences):**
NerveNet proposes encoding a robot's physical structure directly as a graph neural network (GNN) for policy learning, where each node corresponds to a body part and edges reflect kinematic connections. Message-passing between joints captures inter-joint structural relationships, enabling policies that are morphology-aware. The approach demonstrates strong transfer across morphologies and improved sample efficiency compared to monolithic MLP policies.

**Core Contributions (bullet list, 4–7 items):**
- Introduces the idea of mapping a robot's kinematic tree directly to a GNN policy architecture, where each joint/link is a graph node
- Proposes a message-passing scheme where observations local to each body part are propagated through the graph to produce joint-level actions
- Demonstrates zero-shot and few-shot policy transfer across different robot morphologies (e.g., centipedes with different numbers of legs)
- Shows improved sample efficiency over flat MLP policies by leveraging structural inductive bias
- Validates the approach on a variety of MuJoCo locomotion tasks including centipede, snake, and humanoid environments
- Establishes that GNN-based policies generalize better to unseen morphological variations than unstructured alternatives
- Provides ablations showing the importance of graph topology matching the physical kinematic structure

**Methodology Deep-Dive (3–5 paragraphs):**
NerveNet constructs a graph where each node represents a rigid body or joint in the robot's kinematic tree, and edges represent physical connections (e.g., parent-child joint relationships). Each node receives as input the local proprioceptive observation relevant to that body part — for example, a knee joint node receives the knee angle, angular velocity, and any local contact information. This per-node input is first processed through a small embedding MLP to produce initial node features. The graph topology is defined once based on the robot's URDF or kinematic specification and remains fixed during training.

The core of NerveNet is a multi-round message-passing neural network. In each propagation step, every node aggregates messages from its neighbors (connected joints/links), transforms them through learned message functions, and updates its own hidden state. The message function is a shared MLP that takes the concatenation of the sender's and receiver's current hidden states and produces a message vector. After a fixed number of propagation rounds (typically 2–4), each node's final hidden state encodes both its local information and contextual information from the broader kinematic structure. The number of propagation rounds controls the receptive field — more rounds allow information to travel further across the kinematic tree.

After message passing, each node's updated hidden representation is passed through a per-node output MLP that produces the action (torque command) for the corresponding joint. This decentralized action generation naturally respects the robot's structure: each joint's action is informed by its local state and the states of structurally related joints, weighted by learned attention-like message functions. The entire architecture is end-to-end differentiable and trained with standard policy gradient methods (PPO or TRPO).

For transfer experiments, NerveNet exploits the fact that the message and update functions are shared across all nodes. When transferring to a new morphology (e.g., a centipede with more legs), the graph is simply extended with new nodes using the same shared parameters. This weight sharing across nodes is the key mechanism enabling morphology transfer — the policy learns general principles of joint coordination rather than morphology-specific mappings. The authors demonstrate that a policy trained on a 4-legged centipede can transfer to 6- or 8-legged variants with minimal or no fine-tuning.

The training procedure uses standard on-policy RL (TRPO) with the GNN policy as the function approximator. The authors compare against flat MLP baselines that receive the full concatenated observation vector and produce the full action vector. They also ablate the graph structure by testing random graphs and fully-connected graphs, showing that the correct kinematic topology provides the best inductive bias for locomotion tasks.

**Key Results & Numbers:**
- Policy transfer across different morphologies (4-leg to 6/8-leg centipede) with minimal performance loss
- Improved sample efficiency vs MLP baselines: ~30-50% fewer samples to reach comparable performance on locomotion tasks
- Demonstrated on centipede, snake, and humanoid locomotion in MuJoCo
- Ablations show correct kinematic graph topology outperforms random and fully-connected graphs
- Transfer performance degrades gracefully with increasing morphological distance from training morphology

**Relevance to Project A (Mini Cheetah):** MEDIUM — GNN structure encoding is directly applicable to Mini Cheetah's 12-DoF kinematic tree (4 legs × 3 joints), providing a structured inductive bias for the policy network. However, Mini Cheetah uses a single fixed morphology, so morphology transfer is not the primary benefit; the main advantage would be improved sample efficiency and structured inter-joint coordination.

**Relevance to Project B (Cassie HRL):** HIGH — Directly foundational for the MC-GAT (Multi-hop Cross-attention Graph Attention on kinematic tree) component. NerveNet establishes the core paradigm of encoding kinematic structure as a graph for policy learning. The MC-GAT in Cassie's HRL extends this with attention mechanisms (GATv2), but the fundamental idea of message-passing over a kinematic graph originates here.

**What to Borrow / Implement:**
- Graph construction from kinematic tree: map each joint to a node, each physical connection to an edge — directly applicable to both Mini Cheetah (12 nodes) and Cassie (20+ nodes)
- Per-node local observation embedding before message passing — this preprocessing step is used in MC-GAT
- Multi-round message-passing with shared message functions across nodes — the foundation for the MC-GAT propagation scheme
- The insight that correct kinematic topology matters more than arbitrary graph structures for locomotion policy learning
- Transfer learning approach could be used for sim-to-sim transfer between different MuJoCo models of the same robot

**Limitations & Open Questions:**
- Original NerveNet uses simple sum/mean aggregation rather than attention-based aggregation — GATv2 improves on this
- Fixed number of message-passing rounds may not optimally capture long-range dependencies in complex kinematic trees
- Does not address how to handle external inputs (terrain, commands) that are not naturally associated with a single joint
- Transfer across substantially different morphologies (e.g., quadruped to biped) remains challenging
- Computational overhead of GNN vs MLP may not be justified for fixed-morphology applications
---
