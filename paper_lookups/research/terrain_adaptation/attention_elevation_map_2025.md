# Attention-Based Map Encoding for Learning Generalized Legged Locomotion

**Authors:** Zita et al.
**Year:** 2025 | **Venue:** Science Robotics
**Links:** [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.adv3604)

---

## Abstract Summary
This paper proposes an attention-based elevation map encoding mechanism for learning generalized legged locomotion across diverse, challenging terrains. The core innovation is conditioning attention weights on the robot's proprioceptive state, enabling the terrain encoder to dynamically focus on task-relevant regions of the elevation map — particularly potential foothold locations. Unlike standard CNN-based terrain encoders that process the entire map uniformly, this approach learns interpretable attention patterns that reveal the robot's neural topographic reasoning.

The attention mechanism operates on a discretized local elevation map centered on the robot, where each grid cell is treated as a token in a transformer-style attention layer. The proprioceptive state (joint positions, velocities, body orientation, gait phase) serves as the query, while terrain map patches serve as keys and values. This enables state-dependent terrain processing: when the robot is about to place its front-right foot, attention naturally shifts to the terrain ahead and to the right. The resulting attention maps are highly interpretable, showing clear foothold selection patterns.

Extensive experiments on a quadruped robot demonstrate that attention-based terrain encoding significantly outperforms conventional approaches on sparse, risky terrains (stepping stones, narrow beams, cliff edges) while maintaining comparable performance on standard terrains. The approach achieves precise foot placement on terrains where a single misstep leads to failure, something that uniform terrain processing cannot reliably accomplish.

## Core Contributions
- **Proprioceptive-conditioned attention** over elevation maps that dynamically focuses on task-relevant terrain regions based on the robot's current state and gait phase
- **Interpretable neural topographic reasoning** — attention maps clearly reveal foothold selection logic, providing transparency into the policy's terrain processing decisions
- **Generalization across terrain types** from a single unified policy, handling everything from flat ground to stepping stones to narrow beams without terrain-specific modules
- **Token-based elevation map processing** that treats terrain patches as a sequence, enabling variable-resolution terrain encoding with spatial relationship preservation
- **Demonstrated superiority on sparse/risky terrains** where precise foothold selection is critical and failure is catastrophic
- **Real-world deployment** validating that attention-based terrain reasoning transfers from simulation to physical hardware

## Methodology Deep-Dive
The elevation map encoding begins with a local heightmap discretized into an N×N grid (typically 20×20 at 5cm resolution, covering a 1m×1m area around the robot). Each grid cell contains the terrain height value, optionally augmented with local surface normal and curvature estimates. These cells are linearized into a sequence of terrain tokens, each embedded through a small MLP that maps the raw terrain features to a d-dimensional token embedding. Positional encodings (2D sinusoidal) are added to preserve spatial relationships.

The attention mechanism follows a cross-attention design inspired by the transformer decoder architecture. The query is derived from the robot's proprioceptive state vector, projected through a learned linear layer. The keys and values come from the terrain token embeddings, projected through their respective linear layers. Multi-head attention (typically 4-8 heads) is applied, allowing different heads to attend to different aspects of the terrain (e.g., one head for foothold selection, another for obstacle avoidance, another for slope assessment). The attention output is concatenated across heads and passed through a feed-forward network to produce the final terrain embedding.

The proprioceptive conditioning is crucial. The query vector encodes not just the current joint configuration but also gait phase information (which foot is about to be placed), body velocity, and orientation. This allows the attention to shift dynamically with the gait cycle. During the swing phase of a particular leg, attention weights concentrate on the terrain region where that foot will land. During double-support phases, attention distributes more broadly for path planning.

The complete policy architecture stacks the attention-based terrain encoder with a standard RL policy network. The terrain embedding is concatenated with the proprioceptive observation and passed through an MLP policy trained with PPO. Training uses Isaac Gym with massive parallelism (4096 environments) and a terrain curriculum that progressively introduces more challenging scenarios. The terrain curriculum specifically emphasizes sparse terrains (stepping stones with increasing gap size, narrowing beams) to force the policy to develop precise foothold selection.

Domain randomization is applied to terrain geometry (height variation, gap placement, surface friction), robot dynamics (mass, joint friction, motor delay), and sensor noise (heightmap measurement noise, proprioceptive noise). An important training detail is the elevation map noise model, which simulates realistic depth sensor artifacts including missing data, edge artifacts, and drift.

## Key Results & Numbers
- **92% success rate** on stepping stones (30cm spacing) vs. 65% for CNN-based terrain encoder and 40% for blind locomotion
- **Interpretable attention maps** show foothold-aligned attention peaks correlating >85% with actual foot placement locations
- **15-20% improvement** in foot placement precision (measured as distance from foot contact to nearest safe foothold center)
- **Generalization** to unseen terrain configurations with <5% performance degradation compared to seen terrains
- **Real-world deployment** on ANYmal quadruped traversing stepping stones, gaps, and narrow beams with onboard terrain mapping
- **Inference speed**: attention-based encoder adds only **1.5ms** latency compared to standard CNN encoder (total policy inference <5ms)
- **Multi-head attention analysis**: distinct heads specialize in foothold selection (head 1-2), obstacle avoidance (head 3), and slope assessment (head 4)
- Training requires approximately **12 hours** on a single A100 GPU with 4096 parallel environments

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to the Mini Cheetah project as it provides a state-of-the-art terrain encoding mechanism for quadruped locomotion. The attention-based elevation map processing could replace or augment standard CNN-based terrain encoders in the Mini Cheetah's perception pipeline. The proprioceptive-conditioned attention is particularly relevant as it integrates naturally with the Mini Cheetah's 12 DoF joint state information. The demonstrated real-world transfer on similar quadruped hardware (ANYmal) suggests feasibility for Mini Cheetah deployment.

The interpretability of attention maps is a practical advantage for debugging and analyzing the Mini Cheetah's terrain reasoning during development. Understanding where the policy "looks" when making foothold decisions can accelerate the sim-to-real debugging process.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critical to Project B as the attention-over-terrain-maps mechanism is directly applicable to Cassie's terrain encoder module. The proprioceptive-conditioned cross-attention architecture could serve as the foundation for CPTE or as a complementary terrain encoding approach. For bipedal locomotion, the foothold selection aspect is even more critical than for quadrupeds since Cassie has only two contact points and each foothold decision directly impacts balance.

The multi-head attention specialization (foothold selection, obstacle avoidance, slope assessment) maps to different levels of Cassie's hierarchy. Foothold attention could inform the Controller level, obstacle attention could feed the Planner level, and slope assessment could influence the Safety level. The interpretability is especially valuable for the hierarchical architecture, enabling diagnosis of terrain processing at each level. The GATv2 attention mechanism already planned for Cassie's architecture is conceptually aligned with this work's transformer-style attention.

## What to Borrow / Implement
- **Cross-attention terrain encoder** — implement proprioceptive-conditioned cross-attention over elevation map tokens as an alternative or complement to CPTE's contrastive approach
- **Multi-head attention specialization** — design attention heads with specific roles (foothold, obstacle, slope) for different hierarchy levels in Cassie's architecture
- **Gait-phase-aware query conditioning** — encode gait phase in the attention query to shift terrain focus dynamically with the locomotion cycle
- **Sparse terrain curriculum** — design training scenarios emphasizing stepping stones and narrow supports to force precise bipedal foot placement learning
- **Attention map visualization pipeline** — implement real-time attention visualization for debugging terrain reasoning during sim-to-real transfer

## Limitations & Open Questions
- **Quadruped-only validation** — all experiments use quadruped robots; bipedal attention patterns may differ significantly due to reduced support polygon and higher balance sensitivity
- **Fixed-resolution elevation map** — the 5cm grid resolution may be insufficient for fine foothold selection on very sparse terrains; adaptive resolution could improve precision
- **Computational scaling** — attention complexity is O(N²) in the number of terrain tokens; larger maps or higher resolution may require efficient attention variants
- **No temporal attention** — the current mechanism processes a single snapshot; temporal attention over terrain map history could improve prediction of upcoming terrain features
