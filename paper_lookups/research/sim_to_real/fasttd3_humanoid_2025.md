# FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Locomotion

**Authors:** arXiv Authors (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2505.22642)

---

## Abstract Summary
FastTD3 challenges the prevailing assumption that on-policy methods like PPO are necessary for high-throughput legged robot training in massively parallel simulators. The paper demonstrates that TD3 (Twin Delayed DDPG), when properly scaled with large batch sizes, parallel simulation, and distributional critic enhancements, can match or exceed PPO's performance on humanoid locomotion tasks while maintaining the sample efficiency advantages inherent to off-policy methods. The result is stable humanoid walking learned in hours instead of days.

The key innovations are threefold: (1) scaling TD3 to use large replay buffers (10M+ transitions) with batch sizes of 4096-16384, enabling efficient utilization of parallel simulator data; (2) replacing TD3's standard twin critics with distributional critics (quantile regression) that provide richer gradient signal and better uncertainty estimation; (3) a simplified training recipe that eliminates many hyperparameters commonly needed for stable off-policy training (no prioritized replay, no n-step returns, no complex exploration schedules). The simplicity is a feature—FastTD3 uses fewer hyperparameters than PPO while achieving comparable results.

Validated on humanoid locomotion (walking, running, turning) with sim-to-real transfer to a physical humanoid robot, FastTD3 demonstrates that the off-policy paradigm is viable for large-scale legged robot training. Training from scratch to stable walking takes approximately 2-4 hours on a single GPU with 4096 parallel environments, compared to 8-16 hours for PPO-based approaches.

## Core Contributions
- Demonstration that off-policy TD3 can match PPO throughput when scaled with parallel simulation, challenging the on-policy dominance in legged robot RL
- Distributional critic (quantile regression with 32 quantiles) replacing standard twin critics, providing richer gradient signal and implicit pessimism
- Large-batch off-policy training recipe: batch size 16384, replay buffer 10M, no prioritized replay, simple uniform sampling
- 4-8x sample efficiency over PPO while matching wall-clock training time through parallel simulation scaling
- Simplified hyperparameter set: fewer tunable parameters than PPO (no clip ratio, no GAE lambda, no entropy bonus, no KL penalty)
- Sim-to-real transfer demonstrated on physical humanoid robot with standard domain randomization
- Open-source implementation enabling reproducibility

## Methodology Deep-Dive
FastTD3's architecture builds on standard TD3 with three critical modifications. First, the twin critic is replaced with a distributional critic using Quantile Regression (QR-TD3). Each critic network outputs N=32 quantile values {θ_i}_{i=1}^{32} representing the distribution of returns at evenly spaced quantile levels {τ_i = (2i-1)/64}. The quantile regression loss is: L_QR = Σ_i Σ_j ρ_{τ_i}(δ_{ij}) where δ_{ij} = r + γθ'_j(s',a') - θ_i(s,a) and ρ_τ(u) = |τ - I(u<0)| · |u| is the asymmetric Huber loss. The actor uses the mean of the distributional critic for gradient computation, while the conservative quantile (e.g., 25th percentile) is used for action selection, providing implicit pessimism without explicit min-of-two-critics.

Second, the training scales to massive batch sizes. With 4096 parallel environments in Isaac Gym, each simulation step produces 4096 transitions. FastTD3 performs one gradient update per simulation step with batch size 16384 (sampling 4x the per-step data from the replay buffer). This high update-to-data ratio, combined with large batches, stabilizes off-policy learning that would typically suffer from instability at such scale. The replay buffer holds 10M transitions (~2500 episodes), ensuring sufficient data diversity.

Third, exploration is simplified. Instead of complex exploration schedules (OU noise, linearly decaying Gaussian), FastTD3 uses fixed Gaussian noise σ=0.1 on the actor output throughout training. The distributional critic provides implicit exploration incentive—quantile disagreement in uncertain states naturally drives the policy toward under-explored regions. Target policy smoothing uses clipped noise (σ=0.2, clip=0.5) following standard TD3.

The network architecture uses simple 3-layer MLPs (1024-512-256) for both actor and (distributional) critic, with LayerNorm after each hidden layer for training stability at large batch sizes. No recurrence or attention is used, keeping the architecture minimal. The actor outputs joint position targets (PD control), with action space corresponding to the humanoid's joint angles. Observations include joint positions, velocities, body orientation (quaternion), angular velocity, and commanded velocity.

Domain randomization for sim-to-real follows standard practices: friction (0.5-2.0), mass (±20%), motor strength (0.8-1.2), PD gains (±20%), and observation noise (±5%). The randomization is applied uniformly across the 4096 parallel environments, and no curriculum is used—all randomization is active from the start of training. This simplicity is intentional, demonstrating that the off-policy method handles randomization without the careful staging often needed for PPO.

The training recipe eliminates common RL training complexities: no learning rate schedule (fixed at 3e-4 for both actor and critic), no reward normalization (raw rewards used), no observation normalization (only clipping to [-5, 5]), no gradient clipping beyond default PyTorch behavior, and no warm-up period (training starts immediately).

## Key Results & Numbers
- Humanoid walking: learned in 2-4 hours on single A100 GPU with 4096 parallel environments (vs 8-16 hours for PPO)
- Forward velocity tracking: 0.04 m/s error at 1.0 m/s command (comparable to PPO-based approaches)
- Sample efficiency: reaches PPO-equivalent performance using 4-8x fewer environment transitions
- Sim-to-real transfer: 85% velocity retention on physical humanoid (comparable to PPO-trained policies at 82-87%)
- Training stability: 4/5 random seeds produce walking policies (vs 5/5 for PPO, 3/5 for standard TD3)
- Distributional critic improves over standard twin critics by 15-20% in final return and 30% in training stability
- Wall-clock breakdown: 60% simulation, 25% gradient updates, 15% data transfer/replay sampling
- Total hyperparameters to tune: 6 (lr, batch_size, buffer_size, noise_σ, τ_target, n_quantiles) vs 10+ for PPO

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
FastTD3 provides an alternative training paradigm for Project A's Mini Cheetah locomotion. If PPO training is taking too long or not achieving desired performance, FastTD3's recipe offers a simpler, potentially faster approach. The 4-8x sample efficiency improvement could significantly reduce Mini Cheetah training iterations, though wall-clock time may be similar given parallel simulation.

The distributional critic is the most transferable innovation—it can be applied to any off-policy method and provides better gradient signal for locomotion reward functions with multiple components. However, switching from PPO to TD3 for Project A requires validating that the exploration behavior is sufficient for quadruped locomotion (which has a different structure than humanoid walking). The simplified hyperparameter set is attractive for reducing the tuning burden. The domain randomization approach (all-at-once, no curriculum) contradicts Project A's curriculum-based approach but may be worth testing as a simpler baseline.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
FastTD3 is highly relevant to Project B's Cassie training, particularly at the Controller level where joint-tracking policies must be trained efficiently. The humanoid-specific results are directly transferable to Cassie's bipedal morphology. The distributional critic's ability to handle multi-modal return distributions is valuable for Cassie's complex reward landscape where different locomotion modes produce very different return profiles.

The massive batch training approach is compatible with Project B's training infrastructure (likely using Isaac Gym or similar parallel simulator). The 2-4 hour training time for humanoid walking suggests Cassie Controller training could achieve similar efficiency. The simplified training recipe reduces the hyperparameter burden in an already complex hierarchical system. The finding that off-policy methods can match on-policy throughput opens the possibility of using replay buffers across hierarchy levels—storing Planner commands and Controller responses for more sample-efficient hierarchical learning. However, integration with the Dual Asymmetric-Context Transformer may require architectural modifications to accommodate TD3's deterministic policy (vs PPO's stochastic policy).

## What to Borrow / Implement
- Implement distributional critic (QR-TD3) as an enhancement for off-policy training at the Controller level of Project B
- Test FastTD3's simplified training recipe as a baseline for Mini Cheetah training, comparing against current PPO setup
- Apply large-batch training (16384) with Isaac Gym parallel environments to accelerate training for both projects
- Use the conservative quantile (25th percentile) for action selection as an alternative to min-of-two-critics for pessimistic value estimation
- Evaluate the all-at-once domain randomization approach vs curriculum-based randomization for both platforms

## Limitations & Open Questions
- Slightly lower training stability (4/5 seeds) compared to PPO (5/5 seeds) suggests off-policy methods still have reliability issues at scale
- Deterministic policy may limit exploration in highly multi-modal locomotion tasks (e.g., discovering multiple gaits)
- No evaluation on rough terrain or highly dynamic tasks (jumping, recovery); performance on complex locomotion modes unknown
- Distributional critic adds 32x parameter and compute overhead in the critic; impact on training throughput at very large scale unclear
