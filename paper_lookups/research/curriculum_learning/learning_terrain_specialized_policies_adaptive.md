---
## 📂 FOLDER: research/curriculum_learning/

### 📄 FILE: research/curriculum_learning/learning_terrain_specialized_policies_adaptive.md

**Title:** Learning Terrain-Specialized Policies for Adaptive Locomotion in Legged Robots
**Authors:** F. Angarola, M. Posa, et al.
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2509.20635

**Abstract Summary (2–3 sentences):**
This work proposes a hierarchical terrain-specialized policy framework where a high-level selector chooses among sub-policies individually trained for specific terrain types. Each specialist policy is trained via curriculum-based reinforcement learning on its designated terrain, and the selector network learns to identify terrain conditions and route commands to the appropriate specialist. The approach outperforms monolithic generalist policies by approximately 30% on challenging terrains including low-friction surfaces and discontinuous ground.

**Core Contributions (bullet list, 4–7 items):**
- Introduces a hierarchical architecture with terrain-specialized sub-policies and a learned selector
- Demonstrates that specialist policies outperform generalist policies by ~30% on challenging terrains
- Uses curriculum-based training for each specialist, progressively increasing terrain difficulty
- Achieves smooth policy switching between terrain specialists without gait discontinuities
- Provides a principled method for terrain classification that drives policy selection
- Validates the approach in high-fidelity simulation across multiple terrain types
- Shows modular extensibility — new terrain specialists can be added without retraining existing ones

**Methodology Deep-Dive (3–5 paragraphs):**
The framework is structured as a two-level hierarchy. At the lower level, multiple specialist locomotion policies are trained independently, each targeting a specific terrain class (flat ground, slopes, stairs, low-friction surfaces, discontinuous terrain, etc.). Each specialist is trained using PPO with a terrain-specific curriculum that progressively increases the difficulty parameters of its assigned terrain. For example, the low-friction specialist starts training on moderate friction coefficients and gradually decreases them, while the stairs specialist progresses from shallow to steep step heights. This curriculum-per-specialist approach allows each policy to develop terrain-appropriate strategies without the catastrophic forgetting that plagues single-policy multi-terrain training.

The high-level selector network is a separate neural network that takes as input the robot's proprioceptive observations and recent trajectory history, then outputs a probability distribution over the available specialist policies. The selector is trained in a second phase after all specialists are frozen, using a reward signal that combines task performance (velocity tracking) with a smoothness penalty for frequent policy switching. The selector learns implicit terrain classification from proprioceptive signals — it does not receive explicit terrain labels but instead infers terrain type from patterns in joint torques, foot contact forces, and body acceleration.

Policy switching is handled through a gating mechanism that blends actions from adjacent specialists during transitions, preventing discontinuous jumps in commanded joint positions. When the selector's confidence in a new specialist exceeds a threshold, a weighted interpolation period (typically 0.2–0.5 seconds) smoothly transitions from the current to the new specialist's actions. This blending is critical for maintaining stability during terrain transitions, as abrupt policy changes can cause the robot to stumble.

Each specialist policy uses the same network architecture: a two-layer MLP (256×128 hidden units) with proprioceptive inputs (joint positions, velocities, body orientation, angular velocity, foot contact booleans) and joint position targets as outputs. Domain randomization is applied within each specialist's training, but randomization ranges are narrower than those used for generalist policies, allowing specialists to exploit terrain-specific structure. The observation space also includes a short history window (5 previous timesteps) to help the policy infer terrain properties from temporal patterns.

Evaluation is conducted in simulation across a suite of terrain types, comparing the specialist ensemble against a single generalist policy trained on all terrains simultaneously with a unified curriculum. The specialist system achieves higher velocity tracking accuracy, lower energy consumption, and fewer falls, with the largest gains on the most challenging terrains where the generalist policy struggles to find a single strategy that works.

**Key Results & Numbers:**
- ~30% performance improvement over generalist policy on challenging terrains
- Smooth policy switching with <0.5 second transition periods
- Individual specialists achieve near-optimal performance on their assigned terrains
- Selector achieves >90% terrain classification accuracy from proprioception alone
- Modular architecture allows adding new specialists without retraining
- Validated in simulation across 6+ terrain types
- Energy efficiency improved by 15–20% compared to generalist on matched terrains

**Relevance to Project A (Mini Cheetah):** HIGH — Terrain specialization is directly relevant to Mini Cheetah outdoor deployment where the robot encounters varying surfaces. The curriculum-per-specialist approach can be adapted for the Mini Cheetah's 12 DoF PPO training pipeline.
**Relevance to Project B (Cassie HRL):** HIGH — The terrain-specialized sub-policies map directly to the primitives level of the Cassie 4-level HRL. The selector mechanism parallels the planner's role in choosing appropriate primitive behaviors, and the modular specialist design aligns with the hierarchical decomposition philosophy.

**What to Borrow / Implement:**
- Adopt the specialist-per-terrain training paradigm for Mini Cheetah outdoor deployment
- Use the proprioception-based terrain classifier as input to the Cassie planner level
- Implement the smooth policy blending mechanism for transitions between primitives in Cassie HRL
- Apply narrow domain randomization ranges within each specialist for better per-terrain performance
- Use the modular architecture pattern to allow incremental addition of new locomotion capabilities

**Limitations & Open Questions:**
- Requires pre-defined terrain categories; does not handle truly novel/unseen terrains gracefully
- Selector accuracy depends on terrain having distinctive proprioceptive signatures
- Policy blending during transitions may be suboptimal for rapid terrain changes
- Simulation-only validation; no real-world hardware experiments reported
- Scalability to large numbers of terrain types not thoroughly explored
- Fixed specialist set — no mechanism for online learning of new specialists
---
