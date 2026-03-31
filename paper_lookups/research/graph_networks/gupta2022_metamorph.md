# MetaMorph: Learning Universal Controllers with Transformers

**Authors:** Agrim Gupta, Linxi Fan, Surya Ganguli, Li Fei-Fei
**Year:** 2022 | **Venue:** ICLR
**Links:** https://openreview.net/forum?id=Opmqtk_GvYL

---

## Abstract Summary
MetaMorph proposes a Transformer-based universal controller that generalizes across diverse robot morphologies by treating morphology as a modality analogous to language. Instead of training separate policies for each robot design, MetaMorph encodes both the robot's morphological description (limb lengths, joint types, connectivity) and its proprioceptive observations into a unified sequence of tokens, which is then processed by a standard Transformer to produce actions. This formulation enables a single policy to control robots with vastly different body plans — from simple planar walkers to complex 3D quadrupeds and hexapods.

The key insight is that a robot's morphology can be tokenized in a principled way: each limb or joint becomes a token with associated morphological features (limb length, joint range, relative position in the kinematic tree) concatenated with the corresponding proprioceptive observation (joint angle, velocity). The Transformer's self-attention mechanism naturally learns inter-limb coordination patterns that generalize across morphologies, as the attention mechanism can dynamically route information based on morphological context.

MetaMorph is pre-trained on a large dataset of procedurally-generated modular robots (100+ distinct morphologies) performing locomotion and manipulation tasks in MuJoCo. The resulting universal policy achieves zero-shot transfer to unseen morphologies without any fine-tuning, often outperforming morphology-specific policies that were individually trained. This demonstrates that Transformer-based architectures can learn compositional motor control primitives that recombine across body plans.

## Core Contributions
- **Morphology as a modality:** Introduced the paradigm of encoding robot morphology as a token sequence alongside observations, enabling Transformers to jointly reason about body structure and control
- **Universal locomotion policy:** Trained a single policy across 100+ morphologies that generalizes zero-shot to unseen body plans
- **Morphology tokenization scheme:** Designed a principled method for converting URDF-like robot descriptions into Transformer-compatible token sequences with positional encodings reflecting kinematic tree structure
- **Combinatorial generalization:** Demonstrated that the Transformer learns compositional limb-level primitives that recombine for novel morphologies (e.g., training on 4-leg and 6-leg robots enables generalization to 5-leg robots)
- **Large-scale morphology pre-training:** Established the value of training on diverse morphologies as a pre-training strategy, analogous to language model pre-training
- **Open-source codebase:** Released training code, environments, and procedural morphology generation tools

## Methodology Deep-Dive
MetaMorph's architecture processes a robot as an ordered sequence of limb tokens. Each token `x_i` consists of: (1) a morphological descriptor vector `m_i ∈ R^d_m` encoding limb length, joint type (revolute/prismatic), joint axis, parent-child relationships, and mass; (2) a proprioceptive observation vector `o_i ∈ R^d_o` encoding current joint angle, joint velocity, and local body-frame orientation. These are concatenated and projected to the Transformer's hidden dimension: `h_i = W_proj [m_i; o_i] + PE_i`, where `PE_i` is a positional encoding derived from the limb's depth in the kinematic tree.

The Transformer encoder processes the token sequence with standard multi-head self-attention and feed-forward layers. The self-attention mechanism is unrestricted (all tokens attend to all others), allowing the model to learn arbitrary inter-limb coordination patterns. The authors experiment with both absolute positional encodings (based on tree depth) and relative positional encodings (based on shortest path in kinematic tree), finding that relative encodings improve generalization to morphologies with different tree depths.

For action generation, each limb token's output representation is passed through a per-token MLP action head that produces the joint torque or target angle for that limb. This per-token design ensures that the action dimensionality automatically matches the number of limbs in any morphology. A shared value function head averages all token representations and outputs a scalar state value for PPO training.

Training uses PPO with GAE (λ=0.95) across a curriculum of morphologies. In each training batch, environments are randomly populated with different morphologies. The Transformer parameters are shared across all morphologies, with the token sequence length varying by robot. Reward is a standard locomotion reward (forward velocity - energy cost - alive bonus). The authors find that training on more diverse morphologies improves generalization, with diminishing returns beyond ~100 training morphologies.

A critical design decision is the morphology description encoding. The authors compare three approaches: (1) raw URDF parameters, (2) learned morphology embeddings from a separate auto-encoder, and (3) a graph neural network encoding of the kinematic tree. They find that raw URDF parameters with proper normalization perform competitively while being simplest to implement. However, the GNN-based encoding shows advantages for highly heterogeneous morphology sets.

## Key Results & Numbers
- Zero-shot transfer to unseen morphologies: MetaMorph achieves 85-95% of the performance of morphology-specific PPO policies without any fine-tuning
- Training on 100+ morphologies: 2 billion environment steps total, ~20 million per morphology, using 256 parallel MuJoCo environments
- Combinatorial generalization: Training on 4-leg and 6-leg robots enables 78% performance on unseen 5-leg morphologies
- Transformer scale: 6-layer Transformer with 256 hidden dim, 8 attention heads, ~5M parameters
- Outperforms morphology-specific policies on 40% of test morphologies (the universal policy sometimes learns better coordination than individual training)
- With fine-tuning (10% additional training): achieves 98% of morphology-specific performance on held-out morphologies

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
MetaMorph's universal controller concept is directly applicable to the Mini Cheetah project. If policies need to transfer across quadruped variants (different leg lengths, mass distributions, or actuator characteristics from domain randomization), the morphology-conditioned Transformer provides a principled framework. The Mini Cheetah's morphological parameters (link lengths, masses, motor parameters) can be tokenized per-limb and fed alongside proprioceptive observations. This enables: (1) more robust sim-to-real transfer by treating sim and real as different "morphologies," (2) rapid adaptation when physical parameters change (e.g., payload), and (3) pre-training on diverse quadruped variants in MuJoCo before fine-tuning on Mini Cheetah specifically. The per-limb tokenization is especially natural for a quadruped with 4 identical legs.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
MetaMorph provides a complementary and potentially alternative approach to MC-GAT for morphology-aware control. While MC-GAT uses GATv2 on the kinematic graph, MetaMorph uses Transformers on a morphology token sequence — both achieve morphology-conditioned control but through different mechanisms. For Project B's Dual Asymmetric-Context Transformer, MetaMorph's morphology tokenization can inform how to inject Cassie's kinematic structure into the Transformer's input. Specifically, the technique of concatenating morphological descriptors with proprioceptive tokens could be used in the Transformer's "privileged context" or "deployment context" streams. The zero-shot transfer capability is relevant if Project B aims to generalize Cassie's policy to similar bipedal platforms. The per-token action head design maps naturally to Cassie's per-joint control output at the Controller level of the hierarchy.

## What to Borrow / Implement
- **Morphology tokenization scheme:** Adapt MetaMorph's limb-to-token mapping for Cassie's kinematic tree — each of Cassie's 10 actuated joints becomes a token with its URDF parameters concatenated with proprioceptive state
- **Kinematic-tree positional encoding:** Use relative positional encoding based on shortest path in the kinematic tree, which is more generalizable than absolute depth-based encoding
- **Morphology-conditioned training:** Apply domain randomization over morphological parameters (link lengths ±10%, mass ±20%) using MetaMorph's framework to improve sim-to-real robustness
- **Per-token action heads:** Consider replacing the monolithic action MLP in the Controller level with per-joint action heads conditioned on joint-specific Transformer output tokens
- **Pre-training strategy:** Pre-train on diverse bipedal morphologies before specializing to Cassie, leveraging MetaMorph's finding that morphological diversity improves universal policy quality

## Limitations & Open Questions
- MetaMorph operates as a flat policy without hierarchical structure; integrating its morphology tokenization into Project B's 4-level hierarchy (Planner→Primitives→Controller→Safety) requires architectural design work
- The Transformer's quadratic attention cost (`O(n²)`) may be excessive for Cassie's relatively small kinematic tree (~20 nodes); GATv2 on the sparse kinematic graph may be more computationally efficient
- MetaMorph was evaluated primarily on locomotion; its effectiveness for the complex hierarchical tasks in Project B (planning, safety, multi-objective) is unvalidated
- The paper uses MuJoCo environments; adaptation to Project B's specific simulation setup and reward structure requires non-trivial engineering
