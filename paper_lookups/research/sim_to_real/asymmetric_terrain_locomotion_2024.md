# Learning Quadrupedal Locomotion on Tough Terrain Using an Asymmetric Actor-Critic Framework

**Authors:** (2024)
**Year:** 2024 | **Venue:** Applied Intelligence (Springer)
**Links:** [Springer](https://link.springer.com/article/10.1007/s10489-024-05782-7)

---

## Abstract Summary
This paper presents an asymmetric actor-critic framework specifically designed for quadrupedal locomotion over challenging terrain types including stairs, slopes, gaps, and debris fields. The central architectural choice is a critic network that receives privileged terrain features — specifically a multi-resolution terrain feature pyramid computed from ground-truth heightmaps — while the actor receives only proprioceptive observations (joint positions, velocities, IMU, and command velocities). The terrain feature pyramid uses a hierarchical encoding that captures both fine-grained local terrain geometry (step edges, small obstacles) and coarse global terrain structure (overall slope, terrain type), providing the critic with a rich representation of the environment state.

A key contribution is the Terrain Feature Mining (TFM) module within the critic, which uses attention-based feature aggregation over the terrain pyramid to dynamically weight terrain features based on the robot's current state and action context. Rather than naively concatenating all terrain features, TFM learns which terrain features are most informative for value estimation at each timestep. For example, during stair climbing, TFM attends strongly to local edge features and step heights, while during slope traversal, it attends to global gradient features. This dynamic weighting reduces the effective dimensionality of the critic's input, improving value estimation accuracy and training stability.

Experiments on a simulated quadruped (modeled after Unitree A1) across five terrain types demonstrate that the proposed approach significantly outperforms symmetric baselines (both standard PPO and PPO with terrain observation) and naive asymmetric baselines (critic with full heightmap but no attention). Real-world deployment on the Unitree A1 validates sim-to-real transfer quality, with the robot successfully traversing stairs (18cm rise), 25° slopes, and random debris.

## Core Contributions
- Multi-resolution terrain feature pyramid that captures both local and global terrain structure for privileged critic input
- Terrain Feature Mining (TFM) attention module that dynamically selects relevant terrain features based on robot state and action
- Demonstration that structured privileged terrain encoding (pyramid + attention) outperforms naive heightmap concatenation by 20–35%
- Curriculum learning schedule that progressively increases terrain difficulty synchronized with the actor's capability
- Comprehensive ablation study comparing terrain encoding strategies: raw heightmap, CNN features, pyramid, pyramid + attention
- Successful sim-to-real transfer on Unitree A1 across stairs, slopes, and debris
- Analysis of TFM attention patterns revealing interpretable terrain-aware value estimation

## Methodology Deep-Dive
The asymmetric architecture follows the standard paradigm: the actor π(a_t | o_t) takes proprioceptive observations o_t = [q, q̇, ω_body, g_proj, v_cmd, a_{t-1}] where q ∈ R^12 are joint positions, q̇ ∈ R^12 are joint velocities, ω_body ∈ R^3 is body angular velocity, g_proj ∈ R^3 is gravity projection in body frame, v_cmd ∈ R^3 is commanded velocity, and a_{t-1} ∈ R^12 is the previous action. The critic V(o_t, f_terrain) additionally receives terrain features f_terrain computed by the TFM module.

The terrain feature pyramid is constructed from a 10m × 10m ground-truth heightmap centered on the robot, sampled at three resolutions: fine (0.02m/pixel, 50×50 patch around the robot for local detail), medium (0.1m/pixel, 100×100 patch for neighborhood context), and coarse (0.5m/pixel, 20×20 patch for global terrain structure). Each resolution is processed by a separate CNN encoder producing feature vectors f_fine ∈ R^64, f_medium ∈ R^64, and f_coarse ∈ R^32. These features, concatenated with the proprioceptive observation, are fed to the TFM module.

The TFM module uses a cross-attention mechanism where the query is derived from the proprioceptive state and previous action Q = MLP([o_t, a_{t-1}]), and the keys/values are the terrain features K = V = [f_fine; f_medium; f_coarse]. The attention output is: f_TFM = softmax(QK^T / √d) V, producing a state-dependent weighted combination of multi-resolution terrain features. This allows the critic to focus on the most relevant terrain scale for the current locomotion phase. The attention weights are interpretable: during stance phases, the model attends to fine-grained features under the support feet; during swing phases, it attends to medium-resolution features along the swing trajectory.

Training uses PPO with GAE (λ=0.95, γ=0.99) and a curriculum learning schedule. The curriculum defines five terrain levels (flat → gentle slopes → stairs → steep slopes → random debris), with promotion triggered when the agent achieves >80% of maximum reward on the current level for 5 consecutive evaluation episodes. Domain randomization includes: friction (0.4–1.2), mass (±15%), motor strength (±10%), proprioceptive noise (Gaussian σ=0.01), and action delay (0–2 steps). The actor and critic are separate MLPs with hidden sizes [512, 256, 128], trained with Adam optimizer (lr=3e-4).

The sim-to-real transfer uses the trained actor directly with a simple low-pass filter on actions (cutoff 30Hz) to reduce high-frequency jitter. No additional fine-tuning or adaptation module is used. The proprioceptive observations on the real robot are provided by onboard joint encoders, IMU, and a velocity estimator based on leg kinematics.

## Key Results & Numbers
- Asymmetric with TFM outperforms symmetric baselines by 35–50% average return across all five terrain types
- Asymmetric with TFM outperforms naive asymmetric (raw heightmap to critic) by 20–35%
- Stair climbing: 95% success rate on 18cm steps vs. 60% for symmetric baseline
- 25° slope traversal: maintained target velocity within 10% vs. 30% deviation for symmetric
- Random debris: 88% traversal success vs. 52% for symmetric
- TFM attention analysis: fine-grained features receive 55% attention weight during stair climbing, coarse features receive 60% during slope traversal
- Training time: 4 hours on single GPU with 4096 parallel environments
- Sim-to-real transfer: successful on Unitree A1 without additional fine-tuning

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to the Mini Cheetah quadruped locomotion project. The multi-resolution terrain feature pyramid and TFM attention module provide a structured approach to using privileged terrain information in the critic, which is more principled than providing raw heightmaps. The terrain types tested (stairs, slopes, debris) match Mini Cheetah's target deployment scenarios. The curriculum learning schedule, domain randomization parameters, and sim-to-real transfer approach (direct deployment with action low-pass filter) are directly transferable.

The TFM module's interpretable attention patterns provide useful debugging information during training, showing whether the critic is attending to appropriate terrain features for each locomotion phase. This interpretability is valuable for diagnosing training failures and tuning the curriculum.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The asymmetric framework design principles translate to Cassie's Dual Asymmetric-Context architecture. The multi-resolution terrain pyramid can serve as the privileged context stream input, while the TFM attention mechanism is analogous to the cross-attention in the Dual Asymmetric-Context Transformer. The state-dependent attention weighting (different terrain scales for different locomotion phases) is especially relevant for bipedal locomotion, where stance and swing phases have dramatically different terrain interaction patterns.

For Cassie's hierarchical architecture, the TFM concept can be applied at the Controller level for terrain-aware low-level control, while a coarser version provides terrain context to the Planner. The GATv2 graph attention in Cassie's primitive selection can incorporate similar multi-resolution terrain features.

## What to Borrow / Implement
- Implement multi-resolution terrain feature pyramid (fine/medium/coarse) as privileged critic input for Mini Cheetah training
- Adopt TFM cross-attention mechanism for Cassie's Dual Asymmetric-Context Transformer as a terrain feature aggregation module
- Use the curriculum learning schedule (flat → slopes → stairs → debris) with 80% reward threshold promotion for both platforms
- Apply domain randomization parameters (friction 0.4–1.2, mass ±15%, motor ±10%) as baseline ranges
- Leverage TFM attention weight visualization for debugging and interpretability of terrain-aware value estimation

## Limitations & Open Questions
- The 10m × 10m heightmap assumption requires accurate global terrain mapping, which may not be available on real robots without external perception
- TFM attention is trained end-to-end with PPO; there is no guarantee that attention patterns are optimal for value estimation (they are locally optimal)
- The paper only tests quadruped locomotion; bipedal locomotion has different terrain interaction dynamics that may require modified pyramid resolutions
- Sim-to-real results are demonstrated on relatively controlled indoor terrain; outdoor unstructured environments with deformable terrain remain challenging
