# Morphological-Symmetry-Equivariant Heterogeneous Graph Neural Network for Robotic Dynamics Learning

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2412.01297)

---

## Abstract Summary
This paper introduces a Heterogeneous Graph Neural Network (HGNN) that incorporates morphological symmetry equivariance for learning robotic dynamics. Unlike standard GNNs that treat all nodes and edges uniformly, this approach defines distinct node types (joints, rigid links, sensors, actuators) and edge types (kinematic connections, functional couplings, sensor-actuator links), enabling the network to model the heterogeneous nature of robot morphologies more faithfully. The heterogeneous graph formulation allows different message-passing functions for different relationship types.

A central innovation is the integration of morphological symmetry as an equivariance constraint. Most legged robots exhibit bilateral or rotational symmetry—the left side mirrors the right side. By enforcing this symmetry directly in the network architecture through weight-sharing and equivariant message-passing operations, the model requires fewer parameters, generalizes better, and learns faster. The symmetry group is automatically extracted from the robot's URDF description.

Experiments demonstrate that the symmetry-equivariant HGNN significantly outperforms both standard MLPs and homogeneous GNNs on dynamics prediction tasks across multiple robot morphologies. The approach also shows strong cross-morphology generalization, where a model trained on one robot can predict dynamics for a morphologically similar but distinct robot with minimal fine-tuning.

## Core Contributions
- Heterogeneous graph formulation for robot morphology with distinct node types (joints, links, sensors) and edge types (kinematic, functional, sensor-to-actuator)
- Automatic extraction of morphological symmetry groups from URDF robot descriptions
- Symmetry-equivariant message-passing layers that enforce bilateral/rotational symmetry as architectural inductive bias
- Demonstrated improvement over homogeneous GNNs and MLPs on dynamics prediction for multiple robot platforms
- Cross-morphology transfer learning: models trained on one robot generalize to similar morphologies with minimal fine-tuning
- Theoretical analysis showing that symmetry equivariance reduces the effective parameter space and improves sample complexity bounds
- Open-source implementation with URDF-to-graph conversion utilities

## Methodology Deep-Dive
The heterogeneous graph is constructed from the robot's URDF (Unified Robot Description Format). Each joint becomes a joint-type node with features including joint angle, velocity, torque, and joint-axis orientation. Each rigid link becomes a link-type node with features such as mass, inertia tensor, and center-of-mass position. Sensors and actuators become additional node types. Edges are typed: kinematic edges connect parent-child links through joints, functional edges connect co-actuated components, and sensor edges link sensors to the components they observe.

Message passing operates type-specifically. For each edge type (r, s, t) connecting source type s to target type t through relation r, a separate message function φ_{r,s,t} computes messages. This allows the network to learn distinct interaction patterns—the information flow between a joint and its parent link differs fundamentally from the flow between a sensor and an actuator. Node update functions are also type-specific, with separate learned transformations for each node type.

Morphological symmetry is handled by identifying the symmetry group G of the robot (typically Z₂ for bilateral symmetry in legged robots). The authors define group actions on the heterogeneous graph that permute symmetric nodes and transform their features accordingly (e.g., reflecting left-leg joint angles to right-leg joint angles). The message-passing layers are constrained to be G-equivariant, meaning that applying a symmetry transformation to the input produces the correspondingly transformed output. This is achieved through parameter sharing: symmetric message functions share weights, and node updates respect the symmetry structure.

Training uses supervised learning on dynamics prediction: given current state and action, predict next state. Training data is collected from MuJoCo simulations with diverse initial conditions and random actions. The loss function is the mean squared error on predicted joint angles and velocities, with an additional equivariance regularization term that penalizes violations of the symmetry constraint during training.

The architecture uses 3 heterogeneous message-passing layers with 128-dimensional hidden features per node type. Attention mechanisms (similar to GATv2) are optionally applied within each message-passing layer to weight neighbor contributions based on learned relevance. The model is trained with Adam optimizer, learning rate 1e-4, batch size 256, for 500 epochs.

## Key Results & Numbers
- Symmetry-equivariant HGNN reduces dynamics prediction MSE by 35–50% compared to standard homogeneous GNNs
- 60–70% MSE reduction compared to MLP baselines of equivalent parameter count
- Parameter efficiency: symmetry weight-sharing reduces model parameters by ~40% with no performance loss
- Cross-morphology transfer: fine-tuning from quadruped to hexapod requires only 10% of the training data to reach equivalent performance
- Bilateral symmetry enforcement alone (without heterogeneous typing) provides ~20% improvement, suggesting both contributions are valuable
- Training convergence is 2–3× faster with symmetry equivariance compared to unconstrained HGNN
- Tested on A1 quadruped, Cassie biped, and hexapod morphologies in MuJoCo

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The Mini Cheetah has strong bilateral symmetry (left-right) and additional front-back approximate symmetry. Encoding these symmetries as equivariance constraints in a GNN-based policy could significantly reduce the parameter space and improve learning efficiency. The heterogeneous node typing is directly applicable: Mini Cheetah's 12 joints, 13 links, and IMU sensor each have distinct physical roles. The URDF-to-graph pipeline can be directly applied to the Mini Cheetah's existing URDF model.

The dynamics prediction capability could also serve as an auxiliary task during PPO training, providing additional gradient signal that encourages the policy network to build a morphology-aware internal representation.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
This paper is directly foundational for Cassie's MC-GAT (Morphology-Conditioned Graph Attention) module. The heterogeneous graph formulation with distinct node and edge types maps precisely to what MC-GAT needs: Cassie's joints, links, and sensors should be different node types with type-specific message passing. The use of attention mechanisms within heterogeneous message passing aligns with GATv2.

Cassie has strong bilateral symmetry (left-right legs), and enforcing this as an equivariance constraint would halve the effective parameter space in the MC-GAT layers. The URDF-to-graph conversion utility can be applied to Cassie's model. The cross-morphology transfer results suggest that MC-GAT trained on simulated Cassie could potentially transfer to similar bipedal platforms.

## What to Borrow / Implement
- Heterogeneous node typing: define separate node types for Cassie's joints (hip, knee, ankle), links (thigh, shin, foot), and sensors (IMU, encoders)
- Type-specific message functions: learn distinct interaction patterns for kinematic vs functional vs sensor edges
- Bilateral symmetry equivariance: enforce Z₂ symmetry in MC-GAT's GATv2 layers via weight sharing between left and right leg subgraphs
- URDF-to-heterogeneous-graph conversion pipeline for automatic graph construction from robot descriptions
- Attention-weighted heterogeneous message passing as the specific implementation of GATv2 in MC-GAT

## Limitations & Open Questions
- Dynamics prediction (supervised) vs policy learning (RL): the benefits of symmetry equivariance in supervised dynamics may not fully transfer to the RL policy optimization setting
- Assumes perfect symmetry; real robots often have manufacturing asymmetries that could make strict equivariance suboptimal
- Computational overhead of type-specific message functions could be significant for real-time control at high frequencies
- Cross-morphology transfer tested only on similar morphologies; transfer between fundamentally different structures (quadruped↔biped) remains open
