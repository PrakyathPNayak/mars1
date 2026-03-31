# How Attentive are Graph Attention Networks?

**Authors:** Shaked Brody, Uri Alon, Eran Yahav
**Year:** 2022 | **Venue:** ICLR
**Links:** https://arxiv.org/abs/2105.14491

---

## Abstract Summary
This paper identifies a fundamental and previously overlooked limitation in the widely-used Graph Attention Network (GAT): its attention mechanism computes what the authors term *static attention*, meaning the ranking of attention coefficients is the same for all query nodes. Specifically, in GAT the learned attention function applies a linear transformation followed by concatenation and then a LeakyReLU, which results in attention scores that depend monotonically on the key node alone, regardless of the query. This means GAT cannot truly compute query-dependent (dynamic) attention, severely limiting its expressiveness in tasks where the relative importance of neighbors should vary per node.

The authors propose GATv2, a simple yet provably more expressive graph attention variant that achieves *dynamic attention*. The key modification is reordering the internal operations: instead of applying the non-linearity after concatenation with the attention vector, GATv2 first concatenates the transformed features of source and target, then applies the non-linearity, and finally projects with the attention vector. This subtle change allows the attention function to be a universal approximator over the input features, enabling truly query-dependent attention scoring.

GATv2 is shown to be strictly more expressive than GAT — every attention function computable by GAT can also be computed by GATv2, but not vice versa. The authors provide both theoretical proofs and empirical evidence across multiple benchmarks, demonstrating that GATv2 consistently matches or outperforms GAT at equivalent computational cost. GATv2 is implemented as `GATv2Conv` in PyTorch Geometric, making it a drop-in replacement.

## Core Contributions
- **Identification of static attention limitation:** Formally proved that GAT's attention mechanism produces rankings that are independent of the query node, making it a "static" attention that cannot capture dynamic relational patterns.
- **Theoretical expressiveness proof:** Showed that GAT can only compute a restricted family of attention functions, while GATv2 can approximate any attention function (universal approximation over node pairs).
- **GATv2 architecture:** Proposed a modified attention mechanism that simply reorders operations — applying LeakyReLU after concatenation of both source and target transformed features — to achieve dynamic attention.
- **No additional parameters:** GATv2 uses the same number of parameters as GAT, with identical computational cost per layer, making it a strict upgrade.
- **Empirical validation:** Demonstrated improvements on multiple graph benchmarks including OGB datasets and synthetic tasks specifically designed to require dynamic attention.
- **Open-source implementation:** Released GATv2Conv in PyTorch Geometric, enabling easy adoption.

## Methodology Deep-Dive
The original GAT attention mechanism computes attention coefficients as: `e(hi, hj) = LeakyReLU(a^T [Whi || Whj])`. The authors show that because `a` can be decomposed into `[a_l || a_r]`, this becomes `LeakyReLU(a_l^T W hi + a_r^T W hj)`. Since LeakyReLU is a monotonic function, the ranking of attention scores for a fixed query node `i` depends only on `a_r^T W hj`, making it independent of the query node's features. This means all nodes assign the same attention ranking to their neighbors — a property the authors call "static attention."

GATv2 modifies this by computing: `e(hi, hj) = a^T LeakyReLU(W [hi || hj])`. Here, the non-linearity is applied element-wise to the concatenated and transformed features before the final linear projection. Because LeakyReLU is applied before the dot product with `a`, the attention score is no longer decomposable into independent source and target components. This allows the network to learn attention functions where the importance of a neighbor genuinely depends on the query node's features.

The authors construct a rigorous theoretical framework to compare expressiveness. They define the notion of an "attention ranking" — the ordering of neighbors by their attention coefficients — and show that GAT's ranking is a function of the key alone (static), while GATv2's ranking depends on both key and query (dynamic). They prove that GATv2 is a universal approximator for the class of scoring functions over node pairs, assuming sufficient hidden dimensionality.

Empirically, the authors design a synthetic "Dictionary Lookup" task where the correct answer requires dynamic attention (each query node must attend to a different key node based on its own features). GAT fundamentally fails on this task while GATv2 solves it easily. On real-world benchmarks (OGB-Code2, QM9, and VarMisuse), GATv2 consistently improves over GAT, with particularly notable gains on tasks where relational reasoning between node pairs is critical.

The implementation is straightforward and backward-compatible. In PyTorch Geometric, users simply replace `GATConv` with `GATv2Conv` while keeping all hyperparameters the same. The forward pass remains `O(|E| * d)` where `|E|` is the number of edges and `d` is the feature dimension, identical to GAT.

## Key Results & Numbers
- On the synthetic Dictionary Lookup task, GAT achieves ~50% accuracy (random chance) while GATv2 achieves 100%
- OGB-Code2 (code summarization): GATv2 improves F1 by ~0.5 points over GAT
- QM9 molecular property prediction: GATv2 reduces MAE on multiple targets
- VarMisuse (program analysis): GATv2 improves accuracy over GAT
- Zero additional parameters or computational overhead compared to GAT
- Attention distribution analysis shows GATv2 produces genuinely different attention patterns per query node, while GAT's attention rankings are identical across queries

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For the Mini Cheetah RL pipeline, graph-based representations of the robot's kinematic tree could benefit from GATv2's dynamic attention. A quadruped's kinematic graph (torso → hips → knees → feet) could use GATv2 to learn context-dependent joint relationships — for instance, during a turning maneuver, the attention between front-left and rear-right legs should differ from straight walking. However, the Mini Cheetah's relatively simple 12-DOF structure may not fully exploit GATv2's dynamic attention advantage, as simpler architectures (MLPs, standard message passing) may suffice for the limited graph size.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This is the foundational paper for the MC-GAT (Morphology-Conditioned Graph Attention) component in Project B's architecture. MC-GAT uses `GATv2Conv` from `torch_geometric` to process Cassie's kinematic tree, where each node represents a joint/link and edges follow the kinematic chain. Dynamic attention is critical here because Cassie's bipedal morphology requires asymmetric attention patterns — during single-leg stance, the stance leg joints should receive different attention than swing leg joints, and this must change dynamically based on gait phase. The static attention of original GAT would assign the same attention to both legs regardless of gait state. GATv2's dynamic attention enables the MC-GAT module to learn phase-dependent, context-aware joint feature aggregation across the kinematic tree. The Dual Asymmetric-Context Transformer downstream from MC-GAT benefits from richer, more discriminative joint-level features.

## What to Borrow / Implement
- **Direct use of GATv2Conv:** Import `from torch_geometric.nn import GATv2Conv` as the backbone for MC-GAT in Project B's kinematic graph processing
- **Multi-head attention configuration:** Use 4-8 attention heads per GATv2 layer to capture diverse joint-interaction patterns (stance/swing, proximal/distal)
- **Kinematic tree edge construction:** Build edge index from Cassie's URDF/kinematic chain and pass to GATv2Conv with bidirectional edges for message passing
- **Attention visualization for debugging:** Extract and visualize per-head attention weights to verify that the network learns meaningful gait-phase-dependent joint attention patterns
- **Benchmarking against GATConv:** Run ablation replacing GATv2Conv with GATConv to quantify the benefit of dynamic attention for locomotion-specific representations

## Limitations & Open Questions
- GATv2's theoretical advantage is most pronounced when tasks genuinely require dynamic attention; for simple kinematic structures, the improvement over GAT may be marginal and should be empirically validated
- The paper does not explore temporal/recurrent graph attention, which may be needed for locomotion where the graph state evolves over time steps
- Scalability to very deep GATv2 networks (>4 layers) is not extensively studied; over-smoothing may still be an issue on small kinematic graphs
- The interaction between GATv2 attention and downstream RL optimization (PPO gradients flowing through attention) is unexplored in the original paper and may require careful learning rate scheduling
