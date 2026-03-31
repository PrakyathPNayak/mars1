# Rethinking Decision Transformer via Hierarchical Reinforcement Learning

**Authors:** (OpenReview/ICLR 2024)
**Year:** 2024 | **Venue:** OpenReview
**Links:** https://openreview.net/forum?id=WsM4TVsZpJ

---

## Abstract Summary
This paper identifies a fundamental limitation of Decision Transformer (DT): its inability to "stitch" together sub-optimal trajectory segments to compose optimal behaviors. In standard DT, the model is trained to reproduce trajectories from the offline dataset conditioned on returns. If the dataset contains only sub-optimal trajectories (e.g., two trajectories that each solve half of a task well but not the other half), DT cannot combine the best segments to create a trajectory that solves the full task. This limitation is inherent to the sequence modeling paradigm, which conditions on observed returns rather than performing dynamic programming to discover optimal compositions.

To address this, the paper introduces a Hierarchical Decision Transformer (HDT) that decomposes the policy into two levels: a high-level policy that selects sub-goals (intermediate states to reach) and a low-level policy that generates actions to reach each sub-goal. Both levels are implemented as transformer-based sequence models. The high-level policy operates on a coarser temporal scale, selecting sub-goals every H timesteps. The low-level policy conditions on the current sub-goal in addition to the return-to-go, generating actions at the environment's native frequency.

The hierarchical decomposition enables trajectory stitching: the high-level policy can combine sub-goals from different trajectory segments, while the low-level policy executes short-horizon behaviors reliably. HDT significantly outperforms standard DT on offline RL benchmarks that require stitching, particularly on AntMaze navigation tasks where composing partial trajectories is essential for reaching distant goals.

## Core Contributions
- Identifies and formally analyzes the trajectory stitching limitation of Decision Transformer
- Proposes Hierarchical Decision Transformer (HDT) with high-level sub-goal selection and low-level action generation, both implemented as transformers
- Demonstrates that hierarchical decomposition enables trajectory stitching in the sequence modeling paradigm, bridging the gap with value-based offline RL methods
- Achieves significant improvements over standard DT on AntMaze tasks (requiring stitching) while maintaining competitive performance on locomotion tasks
- Provides theoretical analysis showing that the stitching capability arises from the high-level policy's ability to condition on sub-goals from different trajectories
- Shows that the temporal abstraction (sub-goals every H steps) naturally handles the credit assignment problem over long horizons
- Introduces a sub-goal representation learning method that discovers compact, informative sub-goal spaces from offline data

## Methodology Deep-Dive
HDT consists of three components: a sub-goal encoder, a high-level transformer policy, and a low-level transformer policy. The sub-goal encoder φ maps states to a low-dimensional sub-goal space: g = φ(s). This encoder is trained via a VQ-VAE (Vector Quantized Variational Autoencoder) objective on the offline dataset states, learning a discrete codebook of sub-goal embeddings. The discrete codebook ensures that sub-goals are a finite set of meaningful waypoints rather than arbitrary continuous values, which improves the high-level policy's ability to compose sub-goals from different trajectories.

The high-level policy π_H is a causal transformer that operates on a coarse temporal scale. Every H timesteps (e.g., H=10 or H=25), it selects the next sub-goal g_{t+H} conditioned on the return-to-go R̂_t, the current state s_t, and the history of previous sub-goals (g_{t-H}, g_{t-2H}, ...). The transformer architecture allows the high-level policy to attend to the full history of sub-goals, enabling long-horizon planning. The output is a distribution over the VQ-VAE codebook, from which the next sub-goal is sampled.

The low-level policy π_L is a separate causal transformer that generates actions at the environment's native frequency (e.g., every timestep). It conditions on the current sub-goal g_target (set by the high-level policy), the return-to-go, the current state, and the history of recent states and actions within the current sub-goal interval. The low-level policy's context window is limited to H timesteps (the interval between sub-goal updates), keeping its horizon short and tractable. The action prediction follows the standard DT approach: MSE loss for continuous actions.

Training proceeds in three stages. Stage 1: Train the VQ-VAE sub-goal encoder on all states in the offline dataset. This learns a meaningful sub-goal space where nearby embeddings correspond to similar locomotion states. Stage 2: Train the high-level policy to predict the sub-goal at timestep t+H given the history up to timestep t. The target is the VQ-VAE encoding of the actual state at t+H in the dataset, and the loss is cross-entropy over the codebook. Stage 3: Train the low-level policy to predict actions given the sub-goal and recent history, using the standard DT action prediction loss.

The critical design choice is the sub-goal horizon H. Short horizons (H=5) provide fine-grained sub-goals but limit the high-level policy's planning horizon. Long horizons (H=50) allow long-term planning but require the low-level policy to handle more complex behaviors. The paper finds H=10-25 provides the best trade-off across environments. An adaptive horizon mechanism is also explored, where H is dynamically adjusted based on the sub-goal achievement rate during training.

Stitching occurs through the high-level policy. Consider two sub-optimal trajectories: τ_A reaches a waypoint W but then fails, and τ_B starts near W and reaches the goal. Standard DT cannot combine these because it conditions on per-trajectory returns. HDT's high-level policy can learn: from the start state, select the sub-goal "reach W" (from τ_A), then from W, select "reach goal" (from τ_B). The low-level policy, trained on segments of both trajectories, can execute both sub-goal segments. This composition creates a new trajectory not present in the dataset.

## Key Results & Numbers
- AntMaze-Medium-v2: HDT 74.2% vs. DT 28.5% vs. IQL 87.5% (massive stitching improvement over DT)
- AntMaze-Large-v2: HDT 48.3% vs. DT 4.8% vs. IQL 56.7%
- HalfCheetah-Medium-v2: HDT 43.1% vs. DT 42.6% (comparable on non-stitching tasks)
- Hopper-Medium-Expert-v2: HDT 98.4% vs. DT 107.6% (slight regression on high-quality data)
- VQ-VAE codebook size: 512 sub-goals provides best balance between expressiveness and tractability
- Sub-goal horizon H=20 is optimal for AntMaze; H=10 for locomotion tasks
- Training convergence: Stage 1 (VQ-VAE) 50K steps, Stage 2 (high-level) 100K steps, Stage 3 (low-level) 100K steps
- Inference overhead: <2% additional compute compared to standard DT (high-level runs infrequently)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
HDT's hierarchical decomposition could improve offline policy learning for Mini Cheetah by enabling trajectory stitching from sub-optimal demonstration data. If the available Mini Cheetah datasets contain partial successes (e.g., trajectories that navigate obstacles well but fall at the end, or trajectories that maintain balance but move slowly), HDT could compose the best segments. The sub-goal representation could encode meaningful locomotion waypoints (desired base position, velocity milestones) for the high-level policy.

However, continuous locomotion tasks (HalfCheetah results) show minimal improvement over standard DT, suggesting that stitching is less critical for periodic locomotion behaviors compared to navigation. HDT is most valuable when Mini Cheetah must perform goal-directed navigation rather than steady-state locomotion.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
HDT directly mirrors Cassie's 4-level hierarchical architecture and provides a principled framework for implementing the Planner→Primitives interface. The HDT's high-level policy corresponds to Cassie's Planner level (selecting sub-goals / locomotion objectives), while the low-level policy corresponds to the Primitives level (executing locomotion skills to achieve sub-goals). The VQ-VAE sub-goal encoder provides a learned, discrete skill library—directly analogous to the Primitives level's locomotion primitives.

The sub-goal horizon H could be adapted to the Planner's planning frequency. The VQ-VAE codebook could encode Cassie's locomotion modes (walking, turning, standing, recovery) as discrete sub-goals. The stitching capability is critical for Cassie's offline learning from diverse demonstration datasets where no single trajectory demonstrates complete task execution. The multi-stage training pipeline (VQ-VAE → high-level → low-level) provides a structured recipe for training the hierarchical architecture.

## What to Borrow / Implement
- Implement VQ-VAE sub-goal encoder for learning discrete locomotion primitives from Cassie's offline trajectory data
- Adopt the two-level transformer policy structure as the backbone for Cassie's Planner→Primitives architecture
- Use the sub-goal horizon adaptation mechanism for dynamically adjusting the Planner's decision frequency based on task complexity
- Leverage the multi-stage training pipeline (sub-goal learning → high-level planning → low-level control) for structured hierarchical training
- Apply the trajectory stitching capability for composing optimal Cassie locomotion policies from sub-optimal demonstration datasets

## Limitations & Open Questions
- Three-stage training introduces complexity and potential error propagation (VQ-VAE errors affect high-level policy, which affects low-level policy)
- The VQ-VAE codebook size is a critical hyperparameter; too small limits expressiveness, too large makes high-level policy learning difficult
- Performance regresses slightly on high-quality non-stitching datasets (Hopper-ME), suggesting the hierarchical decomposition adds unnecessary overhead when stitching is not needed
- The method assumes sub-goals can be meaningfully defined in the state space; for tasks with complex dynamics (contacts, impacts), state-space sub-goals may be insufficient and task-space sub-goals may be needed
