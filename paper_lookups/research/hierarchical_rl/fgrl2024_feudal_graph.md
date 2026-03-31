# Feudal Graph Reinforcement Learning

**Authors:** Tommaso Marzi, Babel Raafat, Tudor Barbu
**Year:** 2024 | **Venue:** arXiv / OpenReview
**Links:** https://arxiv.org/abs/2304.05099

---

## Abstract Summary
Feudal Graph Reinforcement Learning (FGRL) extends classical feudal hierarchical reinforcement learning by imposing graph-structured communication between policy modules. The architecture builds a pyramidal hierarchy where each layer corresponds to a different level of abstraction: lower layers mirror the agent's physical morphology (e.g., individual joints and limbs), while higher layers encapsulate increasingly abstract task-level components. Inter-layer communication follows the feudal principle where higher-level nodes issue commands to subordinates, but intra-layer communication uses graph message-passing, allowing coordinated behavior among peer modules.

The core innovation is representing the agent's body as a kinematic graph where each node corresponds to a body part (link or joint), and then constructing a hierarchical graph on top of this morphology. At the lowest level, each joint has its own local policy that receives messages from neighboring joints and commands from its hierarchical superior. Mid-level nodes aggregate information from limb groups, and the top-level node makes task-level decisions. This graph-structured feudal hierarchy naturally handles agents with different morphologies by simply changing the graph topology.

FGRL is evaluated on MuJoCo continuous-control locomotion benchmarks, including HalfCheetah, Ant, and Humanoid, where it demonstrates competitive or superior performance compared to both flat RL baselines and prior hierarchical RL methods. The compositional nature of the approach enables transfer across morphologies and facilitates learning in high-dimensional action spaces by decomposing them into coordinated local decisions.

## Core Contributions
- **Graph-structured feudal hierarchy** that combines feudal HRL's top-down goal-setting with GNN-based message-passing for peer coordination
- **Morphology-aware policy decomposition** where the lowest hierarchy level mirrors the agent's kinematic tree, with each joint having a local policy
- **Pyramidal hierarchy construction** algorithm that automatically groups joints into limbs and limbs into higher-level abstractions based on kinematic distance
- **Committee of decentralized policies** that coordinate through message-passing rather than centralized control, improving scalability
- **Compositional transfer** across different agent morphologies by re-using trained limb-level and joint-level policies
- **Evaluation on MuJoCo locomotion** demonstrating SOTA or competitive performance on standard benchmarks

## Methodology Deep-Dive
FGRL constructs a multi-level graph hierarchy from the agent's kinematic description. At the base level (Level 0), each actuated joint in the agent's body corresponds to a graph node. Edges connect joints that are physically linked in the kinematic chain. Each Level-0 node maintains a local policy network that outputs the torque for its corresponding joint. The local policy receives as input: (1) local proprioceptive observations (joint angle, velocity), (2) messages from neighboring Level-0 nodes via graph attention, and (3) a goal embedding from its Level-1 parent node.

Level-1 nodes correspond to limb groups. For a quadruped, these would be the four legs plus the torso. Each Level-1 node aggregates information from its child Level-0 nodes using a learned attention mechanism and produces a goal embedding that is broadcast down to its children. Level-1 nodes also communicate with each other via message-passing, enabling inter-limb coordination (e.g., diagonal leg pairs synchronizing for trotting gaits). The construction of Level-1 groups is done by partitioning the kinematic graph based on structural proximity, where joints belonging to the same limb are grouped together.

Level-2 (and potentially higher) nodes represent abstract task-level controllers. The top-level node receives a global state summary and the task objective, producing high-level commands that propagate down through the hierarchy. Each level uses a form of graph attention network for intra-level communication, where attention weights determine how much each node attends to its peers' messages.

Training uses a combination of techniques. The lowest-level policies are trained with PPO using a combination of the global task reward and local intrinsic rewards that encourage each limb to contribute to the overall motion. Higher-level policies are trained with a variant of the transition policy gradient, where the reward signal measures whether the commanded behavior was achieved by subordinate nodes. The entire system can be trained end-to-end, though the authors report benefits from pre-training lower levels before introducing higher levels.

The message-passing mechanism at each level uses multi-head attention similar to Graph Attention Networks (GAT). Each node computes attention weights over its neighbors, aggregates their hidden states, and combines with its own state to produce an updated representation. This enables dynamic communication patterns that can adapt to the current locomotion phase, for example, stance-phase legs might attend more strongly to each other than to swing-phase legs.

## Key Results & Numbers
- **HalfCheetah-v3:** Achieved average return of approximately 8000, competitive with SAC and superior to HIRO and HAM baselines
- **Ant-v3:** Average return of approximately 5500, outperforming flat PPO (~4500) and HIRO (~4800)
- **Humanoid-v3:** Average return of approximately 6200, showing the approach scales to high-dimensional morphologies (17 joints)
- **Transfer experiments:** Limb-level policies pre-trained on Ant successfully transferred to a 6-legged variant with minimal fine-tuning
- **Sample efficiency:** Approximately 30% fewer environment interactions than flat PPO on Ant and Humanoid due to the structured decomposition enabling more efficient credit assignment
- **Ablation on message-passing:** Removing intra-level communication reduced Ant performance by approximately 15%, confirming the importance of peer coordination

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
FGRL's graph-based morphology representation is directly applicable to the Mini Cheetah's 12-DoF kinematic structure. The Mini Cheetah has four legs, each with three actuated joints (hip abduction/adduction, hip flexion/extension, knee flexion/extension). FGRL would map this to a Level-0 graph with 12 nodes, Level-1 grouping into 4 limb nodes plus a body node, and Level-2 providing the task-level command. This decomposition naturally encourages each leg to learn its local control while coordinating with other legs through message-passing.

For the MuJoCo-based PPO training pipeline, FGRL's architecture could be integrated by replacing the monolithic policy network with the graph-structured hierarchy. Domain randomization would apply to each local policy's observations (e.g., randomizing individual joint friction or mass), and curriculum learning could progressively increase terrain difficulty while the graph structure maintains stable inter-limb coordination. The compositional transfer capability is especially attractive for transferring between simulation and hardware variations of the Mini Cheetah.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High (Critical)**
FGRL directly validates Project B's approach of using graph attention on the kinematic tree. The MC-GAT (Multi-Context Graph Attention) component in Project B uses GATv2 on Cassie's kinematic tree to process proprioceptive information at the Primitives level, and FGRL provides strong evidence that this graph-structured approach works for locomotion. The pyramidal hierarchy construction in FGRL closely mirrors Project B's 4-level hierarchy, with FGRL's levels roughly mapping to the Controller (Level-0), Primitives (Level-1), and Planner (Level-2) in Project B.

Key architectural insights from FGRL for Project B include: (1) the importance of intra-level message-passing, which supports the MC-GAT's use of attention between joint groups within the Primitives level; (2) the benefit of morphology-aware decomposition, which aligns with Cassie's natural grouping of left leg, right leg, and pelvis; and (3) the hierarchical pre-training strategy, which could inform the training schedule for Project B's levels. FGRL's results on Humanoid-v3 are particularly relevant since Cassie shares similar morphological complexity.

## What to Borrow / Implement
- **Kinematic graph construction** algorithm for automatically building the hierarchical graph from Cassie's or Mini Cheetah's URDF/MJCF description
- **Multi-head graph attention** for intra-level peer coordination, directly applicable to MC-GAT implementation with GATv2 attention weights
- **Hierarchical pre-training schedule:** train low-level joint controllers first, then freeze and train limb-level coordinators, then full hierarchy
- **Local intrinsic rewards** per limb group to encourage each leg to actively contribute to locomotion rather than free-riding on other limbs
- **Morphology-conditioned transfer** for moving policies between simulation variants (e.g., different link masses, joint friction)

## Limitations & Open Questions
- **Scalability of graph hierarchy construction:** The automatic grouping algorithm may not produce optimal hierarchies for all morphologies; Cassie's unusual spring-loaded passive joints may not fit neatly into the kinematic partitioning scheme
- **Communication overhead:** Message-passing at every control step adds latency and compute; unclear if this is feasible at the 1kHz+ control rates needed for real hardware deployment
- **Limited to locomotion benchmarks:** FGRL was evaluated only on standard MuJoCo locomotion; unclear how well it handles the contact-rich, impact-heavy dynamics of real-world legged locomotion
- **No sim-to-real validation:** All results are in simulation; the gap between simulated and real graph-attention dynamics is unexplored
