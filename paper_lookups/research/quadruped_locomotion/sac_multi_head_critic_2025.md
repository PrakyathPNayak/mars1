# Quadruped Robot Locomotion via Soft Actor-Critic with Multi-Head Critic

**Authors:** Applied Intelligence Authors (2025)
**Year:** 2025 | **Venue:** Applied Intelligence
**Links:** [Paper](https://link.springer.com/article/10.1007/s10489-025-06584-1)

---

## Abstract Summary
This paper introduces a multi-head critic architecture for Soft Actor-Critic (SAC) specifically designed for quadruped robot locomotion. Standard SAC uses twin critics (Q₁, Q₂) to address overestimation bias, but these critics learn a single composite value function that conflates multiple locomotion objectives (speed, stability, energy efficiency, gait regularity). The multi-head critic decomposes the value function into objective-specific heads, each estimating the expected return for a single reward component, enabling more nuanced multi-objective optimization.

The architecture features K critic heads, each associated with a specific reward component: r_velocity (forward speed tracking), r_stability (body orientation and angular velocity penalties), r_energy (torque and power consumption), r_symmetry (gait regularity), and r_contact (foot contact pattern rewards). Each head is trained with its own TD target, and the actor receives gradients from all heads weighted by configurable objective priorities. This enables explicit trade-off control between competing locomotion objectives without modifying the reward function weights.

Evaluated on quadruped locomotion tasks in simulation (Unitree A1 and Go1 models in Isaac Gym), the multi-head SAC achieves faster convergence, better Pareto-optimal trade-offs between objectives, and higher overall locomotion quality compared to standard SAC, TD3, and PPO baselines. The approach demonstrates that decomposed value estimation improves both learning dynamics and final policy quality.

## Core Contributions
- Multi-head critic architecture where each head estimates the return for a specific reward component (velocity, stability, energy, symmetry, contact)
- Objective-specific TD targets enabling independent learning rates and discount factors per objective, accounting for different temporal scales
- Configurable priority weights for actor gradients, enabling post-training trade-off adjustment without retraining
- Faster convergence (20-35%) compared to standard SAC on quadruped locomotion due to reduced objective interference in the critic
- Superior Pareto-optimal trade-offs between speed and energy efficiency compared to single-critic baselines
- Compatible with SAC's entropy regularization, maintaining exploration benefits while decomposing the value function
- Ablation studies demonstrating the contribution of each critic head to overall locomotion quality

## Methodology Deep-Dive
The standard SAC critic learns Q(s,a) = E[Σ_t γ^t r_t] where r_t = Σ_k w_k r_k,t is the weighted sum of reward components. The multi-head variant instead learns K separate Q-functions: Q_k(s,a) = E[Σ_t γ_k^t r_k,t], one per reward component k. Each head can have its own discount factor γ_k reflecting the temporal scale of its objective: velocity tracking uses γ_vel = 0.99 (long-horizon), while contact pattern uses γ_contact = 0.95 (shorter horizon, more immediate feedback needed). The composite Q-function is Q_total(s,a) = Σ_k w_k Q_k(s,a).

The twin critic mechanism (from standard SAC) is applied per head: each of the K heads has two sub-networks, and the minimum is used for the actor update: Q_k^min = min(Q_k¹, Q_k²). The actor loss becomes: L_actor = E_s[Σ_k w_k · α_k · Q_k^min(s, π(s)) - α_entropy · log π(a|s)], where w_k is the objective weight and α_k is a per-head scaling factor. The entropy term remains shared across all heads.

Implementation uses a shared encoder backbone (3-layer MLP with 512-512-256 units) that processes the state observation, followed by K head modules (each 2-layer MLP with 256-128 units). This shared-backbone-multi-head design reduces total parameter count compared to K independent critics while allowing objective-specific feature learning in the head layers. The actor is a standard SAC Gaussian policy network (3-layer MLP, 512-512-256) that receives gradients from all heads.

Reward decomposition for quadruped locomotion uses five components: (1) r_velocity = -||v_cmd - v_actual||² (velocity tracking error), (2) r_stability = -(||ω||² + ||θ_body||²) (angular velocity and body tilt penalties), (3) r_energy = -Σ_j |τ_j · q̇_j| (joint power consumption), (4) r_symmetry = -||q_left - q_right||² (left-right gait symmetry), (5) r_contact = Σ_feet I(correct_phase) (binary contact pattern reward based on desired gait timing). Each component has an associated weight w_k that determines its relative importance.

A key advantage of the multi-head architecture is the ability to adjust trade-offs post-training without retraining. By changing the priority weights w_k at deployment time, the same trained multi-head critic can produce different policies along the Pareto front. In practice, this is implemented by re-optimizing the actor for a few hundred gradient steps with new weights w_k, using the frozen multi-head critic. This enables rapid adaptation to different deployment scenarios (e.g., prioritize speed on flat terrain, prioritize stability on rough terrain).

The training procedure handles the different temporal scales through per-head target networks with independent update rates: τ_velocity = 0.005 (slow update for long-horizon objective), τ_energy = 0.01 (faster update for shorter-horizon objective). The replay buffer stores (s, a, {r_k}_{k=1}^K, s') tuples, enabling independent TD updates for each head from the same experience.

## Key Results & Numbers
- Convergence speed: Multi-head SAC reaches 90% peak performance in 2M steps vs 3M for standard SAC (33% faster)
- Forward velocity tracking error: 0.05 m/s (multi-head) vs 0.08 m/s (standard SAC) vs 0.12 m/s (PPO)
- Energy efficiency: 15% lower power consumption than standard SAC at comparable speed, due to explicit energy head
- Gait symmetry index: 0.92 (multi-head) vs 0.85 (standard SAC) vs 0.88 (PPO), scale 0-1 where 1 is perfect symmetry
- Pareto front coverage: multi-head SAC discovers 40% more Pareto-optimal policies in the speed-energy trade-off space
- Post-training trade-off adjustment requires only 500 actor gradient steps (~2 minutes) to shift between speed-priority and efficiency-priority policies
- Stability under perturbations: 20% fewer falls in push-recovery tests compared to standard SAC
- Training wall-clock time: ~15% longer than standard SAC due to multi-head computation (5 heads × 2 twins = 10 Q-networks)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The multi-head critic is directly applicable to Mini Cheetah training in Project A, where multiple competing objectives (speed, stability, energy efficiency, gait quality) must be balanced. Currently, these are likely combined into a single scalar reward with manually tuned weights, making it difficult to control trade-offs. The multi-head architecture would enable Project A to: (1) train once with decomposed objectives, (2) analyze which objectives conflict, and (3) adjust trade-offs for different deployment scenarios without retraining.

The five reward components (velocity, stability, energy, symmetry, contact) align directly with Mini Cheetah's locomotion objectives. The per-head discount factors are particularly valuable—Mini Cheetah's velocity tracking needs long-horizon credit assignment while foot contact patterns benefit from shorter horizons. The 33% convergence speed improvement would significantly reduce training time. The post-training trade-off adjustment is ideal for transitioning between lab testing (prioritize stability) and performance evaluation (prioritize speed).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The multi-head critic approach is applicable to Cassie's Controller level, where joint-tracking policies must balance multiple objectives (tracking accuracy, energy efficiency, smoothness, safety margins). The decomposed value function could help the Controller learn more nuanced joint-level behaviors. However, the multi-head architecture adds complexity to an already complex hierarchical system.

For Project B, the most valuable insight may be the per-objective discount factor and the Pareto-front exploration capability. At the Planner level, different locomotion modes (fast walking vs careful stepping) represent different points on the speed-stability Pareto front, and the multi-head critic could enable the Planner to explicitly reason about these trade-offs. The post-training adjustment capability is less relevant for Cassie since the hierarchy handles mode switching at a higher level. The MC-GAT (Multi-Critic Graph Attention) already provides a form of multi-critic reasoning—the multi-head approach could be integrated within MC-GAT's architecture.

## What to Borrow / Implement
- Implement multi-head critic in SAC for Project A's Mini Cheetah training with 5 locomotion objective heads
- Use per-head discount factors to handle different temporal scales of locomotion objectives (γ_vel=0.99, γ_contact=0.95)
- Apply post-training trade-off adjustment for rapid adaptation between deployment scenarios without retraining
- Integrate the reward decomposition scheme (velocity, stability, energy, symmetry, contact) into both projects' reward functions
- Consider combining multi-head critic with MC-GAT in Project B's Controller level for enhanced multi-objective reasoning

## Limitations & Open Questions
- 10 Q-networks (5 heads × 2 twins) significantly increases memory and compute requirements; scalability to 10+ objectives unclear
- Per-head discount factors and update rates introduce additional hyperparameters that may require extensive tuning
- Post-training adjustment assumes convexity of the Pareto front; non-convex trade-offs may require retraining
- Reward decomposition requires careful engineering to avoid correlated objectives that confuse individual heads
