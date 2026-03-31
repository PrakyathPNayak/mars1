# MoELoco: Mixture of Experts for Multitask Locomotion

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** https://moe-loco.github.io/

---

## Abstract Summary
MoELoco introduces a Mixture of Experts (MoE) framework for training a single locomotion policy that handles diverse terrains, gaits, and locomotion modes across both bipedal and quadrupedal robots. Rather than training separate specialist policies for each terrain type or gait pattern and switching between them, MoELoco uses a gating network to dynamically route inputs to specialized expert sub-networks, enabling soft composition of locomotion skills. This approach mitigates the task conflict problem that arises when a single monolithic network must simultaneously optimize for walking on flat ground, climbing stairs, traversing rough terrain, and performing multiple gait types.

The framework demonstrates that MoE architectures naturally partition the locomotion skill space: different experts specialize in different terrain-gait combinations, with the gating network learning to blend experts for novel or transitional scenarios. Importantly, the routing is learned end-to-end through RL training, requiring no manual expert assignment or curriculum design. The system is validated on both quadruped (Go1, A1) and biped (Cassie-like) platforms in simulation, with successful real-world transfer on quadruped hardware.

MoELoco achieves significantly better multi-task performance compared to single-network baselines, while using a comparable total parameter count. The sparse activation pattern (only a subset of experts active per input) also provides computational efficiency at inference time, as inactive expert parameters are not evaluated.

## Core Contributions
- Proposes MoE architecture for multi-task locomotion, where expert sub-networks specialize in different terrain-gait combinations
- Demonstrates end-to-end learning of expert routing through RL training without manual assignment or curriculum design
- Mitigates task conflict in multi-task locomotion—expert specialization prevents gradient interference between dissimilar tasks
- Achieves unified policy handling both bipedal and quadrupedal locomotion within a single framework
- Shows that MoE routing naturally discovers semantically meaningful expert specializations (terrain-specific, gait-specific)
- Validates real-world transfer on quadruped hardware with maintained multi-task performance
- Demonstrates computational efficiency through sparse expert activation—only 2-4 of 8 experts active per inference step

## Methodology Deep-Dive
The MoE architecture replaces the standard MLP policy network with a collection of N expert networks and a learned gating function. Each expert is a small MLP (e.g., 2 layers of 128 units), and the gating network is a separate MLP that takes the observation as input and outputs a softmax distribution over experts. The top-K experts (typically K=2) are selected per input, and the policy output is the weighted sum of the selected experts' outputs, weighted by the gating probabilities.

Formally, given observation o, the gating network produces g(o) ∈ R^N. The top-K indices are selected: I = TopK(g(o), K). The policy output is π(o) = Σ_{i∈I} g_i(o) · E_i(o), where E_i is the i-th expert network and g_i is the normalized gate value. This sparse routing ensures that only K expert forward passes are computed per step, providing computational savings proportional to K/N.

Training uses PPO with a modified objective that includes a load balancing loss. Without explicit balancing, the gating network may collapse to always selecting the same 1-2 experts, wasting the capacity of remaining experts. The load balancing loss penalizes uneven expert utilization: L_balance = N · Σ_{i=1}^{N} f_i · P_i, where f_i is the fraction of inputs routed to expert i and P_i is the average gate probability for expert i. This loss encourages uniform utilization across training while allowing specialization to emerge.

The observation space includes proprioceptive state (joint positions, velocities, base orientation, angular velocity), terrain information (heightmap or terrain type encoding), and command inputs (desired velocity, gait type, heading). The action space is desired joint positions for PD controllers. For the unified bipedal-quadrupedal setting, the observation and action spaces are padded to the maximum dimensionality, with binary masks indicating which joints are active for the current morphology.

Curriculum learning is employed across terrain difficulty levels. Training begins on flat terrain and progressively introduces stairs, slopes, rough terrain, and gaps. The curriculum is automated—terrain difficulty increases when the agent achieves a threshold success rate. The MoE architecture naturally adapts to the curriculum: early experts specialize in flat-ground locomotion, while later-activated experts handle challenging terrains introduced in subsequent curriculum stages.

Domain randomization covers physical parameters (mass ±20%, friction 0.4-1.5, motor strength ±15%), sensor noise (joint encoder noise ±0.01 rad, IMU noise), communication delays (5-25ms), and terrain variations (height perturbations, random obstacles). The MoE framework shows improved robustness to domain randomization compared to monolithic networks, as different experts can specialize in different randomization regimes.

## Key Results & Numbers
- Multi-terrain success rate: MoELoco 92.3% vs. single MLP 76.8% vs. separate specialists 88.1% (with switching overhead)
- Stair climbing: 95% success (MoE) vs. 71% (single MLP) on 15cm stairs
- Rough terrain traversal: 89% success (MoE) vs. 62% (single MLP) with ±8cm height variation
- Expert specialization analysis: 6/8 experts show >70% activation on specific terrain types, confirming meaningful specialization
- Inference time: 0.4ms (MoE, K=2 of 8 experts) vs. 0.3ms (single MLP)—minimal overhead despite higher capacity
- Total parameters: 2.1M (MoE) vs. 1.8M (single MLP)—comparable, but MoE achieves far better performance
- Gait transition smoothness: 23% reduction in torque peaks during gait transitions compared to specialist switching
- Real-world quadruped (Go1): 87% multi-terrain success vs. 91% in simulation
- Bipedal locomotion: 84% flat-ground walking success, 72% stair climbing in simulation

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
MoELoco is directly applicable to Mini Cheetah's need for a versatile locomotion policy. Instead of training separate policies for different terrains and gaits and managing switching logic, MoELoco provides a single unified policy with natural expert specialization. This simplifies the deployment pipeline while improving performance on diverse terrains. The end-to-end learned routing eliminates the need for terrain classification as a separate module—the gating network implicitly classifies the terrain and selects appropriate locomotion strategies.

For Mini Cheetah specifically, the MoE framework could handle the diverse locomotion requirements of the project (trotting, bounding, pronking, walking on flat/rough/stairs) within a single policy. The sparse activation provides computational efficiency critical for the limited onboard compute of Mini Cheetah. The demonstrated real-world transfer on comparable quadruped platforms (Go1, A1) provides confidence in deployability.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
MoELoco's MoE routing mechanism is directly relevant to Cassie's Primitives level, where the system must select and compose locomotion skills. The MoE framework provides a differentiable, learned alternative to the discrete skill selection that the Planner level would otherwise need to provide. Each expert could correspond to a locomotion primitive (walking, turning, standing, stepping over obstacles), with the gating network implementing soft skill blending.

The unified bipedal-quadrupedal framework demonstrates that MoE architectures generalize across morphologies, validating the approach for Cassie's bipedal locomotion. The load balancing loss ensures all locomotion primitives are well-trained, preventing the collapse to only commonly-used skills. The curriculum learning strategy could be adopted for progressively training Cassie on increasingly complex terrains and locomotion modes.

## What to Borrow / Implement
- Implement MoE policy architecture with top-K routing for both Mini Cheetah (terrain/gait experts) and Cassie (locomotion primitive experts)
- Adopt the load balancing loss to prevent expert collapse during RL training
- Use the automated terrain curriculum with MoE to progressively build locomotion skills
- Apply the sparse activation mechanism for computational efficiency on resource-constrained onboard computers
- Leverage the expert specialization analysis tools (activation histograms, expert-terrain correlation) for interpretability and debugging

## Limitations & Open Questions
- Load balancing can conflict with natural specialization—forcing equal expert utilization may prevent optimal routing patterns from emerging
- Top-K routing introduces non-differentiable selection, requiring straight-through estimators or Gumbel-Softmax, which can introduce gradient bias
- Expert count (N) and active expert count (K) are fixed hyperparameters; adaptive expert allocation remains an open challenge
- The paper's bipedal results are less mature than quadruped results, with lower success rates on challenging terrains—further development is needed for Cassie-class bipedal tasks
