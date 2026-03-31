# MI-HGNN: Morphology-Informed Heterogeneous Graph Neural Network for Legged Robot Contact Perception

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [Paper](https://www.aimodels.fyi/papers/arxiv/mi-hgnn-morphology-informed-heterogeneous-graph-neural)

---

## Abstract Summary
MI-HGNN presents a morphology-informed heterogeneous graph neural network designed specifically for contact perception on legged robots. Contact detection—determining which feet are in contact with the ground and estimating contact forces—is critical for locomotion control but challenging due to sensor noise, compliant terrain, and the lack of direct force sensors on many platforms. The authors propose using the robot's kinematic tree structure as the backbone of a heterogeneous GNN that fuses proprioceptive signals (joint encoders, IMU) to predict ground contact states and reaction forces at each foot.

The heterogeneous graph encodes the robot's morphology with different node types for different physical components (hip joints, knee joints, ankle joints, feet, torso) and different edge types for kinematic relationships (revolute joints, fixed connections). By processing information through the kinematic tree structure, the network naturally captures the physical propagation of contact forces through the robot's body—a contact at the foot produces forces and torques that propagate up through the ankle, knee, and hip, which are observable through joint torque sensors and the IMU.

Experimental results demonstrate that MI-HGNN significantly outperforms morphology-agnostic baselines (MLPs, LSTMs, standard CNNs) on contact detection and force estimation tasks across multiple legged robot platforms, including quadrupeds and bipeds. The morphology-informed architecture also generalizes better to unseen terrains and locomotion gaits.

## Core Contributions
- Morphology-informed heterogeneous GNN architecture for contact perception that uses the kinematic tree as graph structure
- Heterogeneous node and edge typing reflecting the physical role of each component in force transmission
- Demonstration that kinematic-tree-based message passing naturally captures contact force propagation patterns
- Superior performance over morphology-agnostic baselines (MLP, LSTM, CNN) on contact detection and force estimation
- Generalization across terrains and gaits not seen during training, enabled by the structural inductive bias
- Extension of GNN applications in legged robotics from control to perception/state estimation
- Validation on multiple robot platforms including quadrupeds (A1, Mini Cheetah) and bipeds

## Methodology Deep-Dive
The MI-HGNN constructs a heterogeneous graph directly from the robot's URDF kinematic tree. Each joint in the tree becomes a node, typed according to its physical role: hip abduction/adduction joints, hip flexion/extension joints, knee joints, and ankle joints each constitute separate node types. Link nodes represent rigid body segments (torso, thigh, shin, foot). Edge types reflect the kinematic relationships: parent-child joint connections, joint-to-link attachments, and the root link's connection to the IMU sensor.

Node features are assigned based on type: joint nodes receive joint angle, joint velocity, and (if available) joint torque measurements. The torso/root node receives IMU data (orientation quaternion, angular velocity, linear acceleration). Link nodes can optionally receive estimated link poses from forward kinematics. All features are normalized using running statistics computed during training.

The message-passing architecture uses 4 layers of heterogeneous graph convolution. In each layer, messages are computed using type-specific linear transformations followed by ReLU activations. For each edge type (r, s, t), the message function takes the concatenation of source and target node features, applies a type-specific learned linear transformation, and produces a message vector. Messages arriving at each node are aggregated (sum or attention-weighted) and passed through a type-specific update function (GRU cell) that maintains a hidden state across message-passing rounds.

The key design principle is that message passing follows the kinematic chain. Information from foot contact propagates upward: foot nodes send messages to ankle joint nodes, which send messages to knee joint nodes, and so on up to the torso. Simultaneously, global context from the IMU propagates downward from the torso through the kinematic chain. After 4 rounds, information from any node can reach any other node (the kinematic tree depth for most quadrupeds/bipeds is ≤4), enabling full-body integration of contact-related signals.

The output layer reads out contact predictions from foot nodes (binary contact classification via sigmoid) and force estimates (3D force vector via linear projection). The model is trained with a combined loss: binary cross-entropy for contact detection and smooth L1 loss for force estimation, weighted equally. Training data consists of simulated locomotion trajectories with ground-truth contact labels from the physics engine, augmented with sensor noise injection and terrain randomization.

## Key Results & Numbers
- Contact detection accuracy: MI-HGNN achieves 94–97% accuracy across gaits (walk, trot, gallop) compared to 85–90% for MLP baselines
- Contact force estimation RMSE reduced by 25–40% compared to morphology-agnostic methods
- Generalization to unseen terrains (grass, gravel, slopes): MI-HGNN maintains >92% accuracy vs <85% for MLPs
- Generalization to unseen gaits: training on trot, testing on gallop shows <3% accuracy drop for MI-HGNN vs >10% for baselines
- Inference time: <1ms per prediction on GPU, suitable for real-time deployment at 1kHz+ control rates
- Validated on quadruped (Unitree A1, MIT Mini Cheetah) and biped (Cassie) platforms in simulation
- Ablation: removing heterogeneous typing (using homogeneous GNN) reduces accuracy by 5–8%, confirming the value of type-specific processing

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Contact perception is fundamental for Mini Cheetah locomotion control. The MI-HGNN architecture could serve as a learned contact estimator that feeds into the RL policy, replacing or augmenting hand-crafted contact detection heuristics. Since Mini Cheetah lacks direct ground force sensors, the ability to estimate contact states from proprioceptive signals (joint encoders + IMU) is directly valuable. The GNN-based approach would be more robust to terrain variations than threshold-based methods.

Integrating MI-HGNN as a perception module within the Mini Cheetah's PPO pipeline would provide the policy with richer, more reliable contact information, potentially improving gait quality and terrain adaptability.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This paper directly supports the MC-GAT module's role in Cassie's architecture. Contact perception via kinematic-tree GNN validates the core idea that GATv2 operating on Cassie's morphology graph can extract meaningful physical information. The MI-HGNN's heterogeneous graph structure—with type-specific nodes for Cassie's hip, knee, and ankle joints—maps directly to the MC-GAT design.

For Cassie's 4-level hierarchy, accurate contact perception is essential at multiple levels: the Safety layer (CBF-QP) needs contact state for constraint enforcement, the Controller needs contact for force distribution, and the Planner benefits from terrain-aware contact predictions. MI-HGNN could be integrated as a perception backbone within MC-GAT, providing contact-aware features to all hierarchy levels.

## What to Borrow / Implement
- Kinematic tree → heterogeneous graph construction pipeline with type-specific nodes for each joint type
- GRU-based node update functions for temporal integration of contact signals across message-passing rounds
- Combined contact classification + force estimation output heads from foot nodes
- Sensor noise injection and terrain randomization during training for robust contact perception
- 4-layer message-passing depth matching kinematic tree depth for full-body information propagation

## Limitations & Open Questions
- Trained in simulation with ground-truth contact labels; real-world deployment requires either real contact labels (from force plates) or self-supervised training approaches
- Binary contact detection may be insufficient; graduated contact states (sliding, partial contact) are not modeled
- Force estimation accuracy degrades on highly compliant/deformable terrains where rigid-body assumptions break down
- Integration with RL policy training (end-to-end vs modular) not explored; unclear if joint training would help or hurt
