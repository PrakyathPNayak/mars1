# Reward Shaping for Reinforcement Learning with An Assistant Reward Agent

**Authors:** Ma, T., Chen, Z., Wu, Y., Wang, Z., Liang, X.
**Year:** 2024 | **Venue:** ICML 2024
**Links:** [Proceedings](https://proceedings.mlr.press/v235/ma24l.html)

---

## Abstract Summary
This paper introduces a dual-agent framework for reward shaping where a separate "reward agent" learns to generate auxiliary dense reward signals for a "policy agent" operating in sparse-reward environments. The reward agent observes the policy agent's trajectories and predicts which states and transitions are most informative for learning, converting sparse binary success/failure signals into continuous, temporally dense reward landscapes. This approach eliminates the need for manual reward engineering while providing stronger learning signals than intrinsic motivation methods like curiosity or count-based exploration.

The key innovation is the reward agent's use of future-oriented trajectory information: it can look ahead in a trajectory to determine whether a particular state ultimately leads to success, and retroactively assign credit through dense auxiliary rewards. This hindsight-aware reward shaping naturally balances exploration and exploitation—early in training when few trajectories succeed, the reward agent encourages broad exploration; as success rate increases, it shifts to providing fine-grained guidance for policy refinement.

The framework is evaluated across continuous control tasks (MuJoCo locomotion, robotic manipulation) and discrete control tasks (navigation, Atari games), consistently outperforming hand-designed rewards, curiosity-driven exploration, and other automatic reward shaping baselines. The reward agent is trained end-to-end alongside the policy agent, requiring no additional supervision beyond the sparse task reward.

## Core Contributions
- Dual-agent architecture separating reward generation from policy optimization, enabling specialized learning dynamics for each
- Future-oriented reward prediction using hindsight trajectory information for temporally dense credit assignment
- Automatic balance between exploration and exploitation via adaptive reward agent behavior across training stages
- Elimination of manual reward engineering while outperforming hand-designed dense rewards on multiple benchmarks
- Theoretical analysis showing the shaped reward preserves the optimal policy of the original sparse MDP (potential-based shaping guarantee)
- Generalization across both continuous (MuJoCo) and discrete (Atari, navigation) domains
- End-to-end training with no additional supervision beyond sparse task rewards

## Methodology Deep-Dive
The framework consists of two neural networks trained in tandem: a policy agent π(a|s) that interacts with the environment and selects actions, and a reward agent R_aux(s, a, s') that observes transitions and outputs auxiliary rewards. The policy agent optimizes the combined reward r_total = r_sparse + α · R_aux, where r_sparse is the original environment reward and α is a mixing coefficient. The reward agent is trained to maximize the policy agent's learning progress, creating a meta-learning dynamic.

The reward agent architecture is a transformer-based sequence model that processes trajectory segments. Given a trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, ..., s_T), the reward agent attends over the full sequence (including future states, since reward assignment happens post-hoc) to predict R_aux(s_t, a_t, s_{t+1}) for each transition. This future-oriented attention is the core technical innovation: by observing the eventual outcome of a trajectory, the reward agent can accurately assign credit to intermediate states that contributed to success or failure.

The training procedure alternates between three phases: (1) the policy agent collects trajectories using the current combined reward; (2) the reward agent is updated using a meta-gradient that measures how much its auxiliary rewards improved the policy agent's performance on the sparse objective; and (3) the policy agent is updated using PPO with the new combined reward. The meta-gradient computation requires differentiating through the policy update step, which is approximated using the policy gradient theorem to avoid expensive second-order derivatives.

A critical design choice is the mixing coefficient α, which decays over training. Initially, α is large (1.0–5.0) to provide strong guidance when the sparse reward is rarely achieved. As the policy improves and sparse rewards become more frequent, α decays exponentially to avoid the auxiliary reward dominating and potentially distorting the original objective. The decay schedule is adaptive: α is reduced when the policy agent's sparse reward rate exceeds a threshold.

The potential-based shaping guarantee is maintained by constraining the reward agent's output to be expressible as R_aux(s, a, s') = γ · Φ(s') - Φ(s) for a learned potential function Φ. This constraint is implemented architecturally by having the reward agent output Φ(s) for each state, and computing the shaped reward as the temporal difference of Φ. This ensures that the optimal policy under the shaped reward is identical to the optimal policy under the original sparse reward, preventing reward hacking.

For locomotion tasks specifically, the sparse reward is defined as a threshold on forward distance (e.g., reward = 1 if distance > 10m within episode). The reward agent learns to provide intermediate guidance: early in training it rewards any forward movement, then progressively refines to reward efficient, stable forward progress. This naturally creates a curriculum without explicit curriculum design.

## Key Results & Numbers
- MuJoCo Ant (sparse): 3.2× faster convergence than PPO with hand-designed dense reward, 8× faster than PPO with sparse reward alone
- MuJoCo Humanoid (sparse): achieves 85% success rate where PPO with curiosity plateaus at 40%
- Manipulation (sparse pick-and-place): 95% success rate vs. 75% for hand-designed rewards and 60% for curiosity
- Reward agent training overhead: ~15% additional compute per iteration
- α decay schedule: starts at 3.0, reaches 0.1 by 70% of training
- Meta-gradient approximation error: <5% compared to exact second-order computation
- Ablation: removing future-oriented attention reduces performance by 25–40% across tasks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The dual-agent reward shaping framework addresses a core challenge in Mini Cheetah training: designing dense reward functions that guide the policy toward desired locomotion behaviors without introducing unintended optima. For sparse objectives like "traverse rough terrain without falling" or "reach a goal position," the reward agent can automatically generate intermediate guidance rewards. This is particularly valuable for the domain randomization curriculum, where the reward landscape changes as randomization difficulty increases.

The potential-based shaping guarantee is important for Mini Cheetah because it ensures that the auxiliary rewards don't change the optimal policy—the robot will converge to the same behavior regardless of the reward agent's specific outputs, just faster. This eliminates a common failure mode where shaped rewards produce fast-training but suboptimal gaits. The 15% computational overhead is acceptable given the convergence speedup.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The dual-agent framework is highly applicable to the Cassie 4-level hierarchy, where each level could benefit from a dedicated reward agent. At the Planner level, the reward agent could provide dense feedback for long-horizon goal reaching that the sparse navigation reward cannot. At the Primitives level (Option-Critic), the reward agent could complement DIAYN/DADS diversity objectives with task-relevant guidance. At the Controller level, it could shape the joint tracking reward to accelerate convergence. At the Safety level (LCBF), it could provide dense rewards for constraint satisfaction.

The future-oriented attention mechanism is particularly relevant for the Planner level, where the RSSM needs to assign credit across long temporal horizons. The automatic exploration-exploitation balance mirrors the adversarial curriculum design at the Primitives level. The potential-based shaping guarantee is critical for the hierarchical setting, where reward distortion at one level could cascade to produce pathological behavior at other levels. The reward agent's adaptive α decay could be coordinated across hierarchy levels, reducing the auxiliary reward contribution as each level matures.

## What to Borrow / Implement
- Implement a reward agent for the Mini Cheetah sparse terrain traversal objective to automatically generate dense intermediate rewards
- Deploy separate reward agents at each Cassie hierarchy level, coordinating their α decay schedules
- Use the potential-based shaping constraint to ensure reward agents don't distort optimal policies at any hierarchy level
- Adopt the meta-gradient training procedure for end-to-end reward agent optimization alongside policy training
- Leverage the future-oriented attention mechanism for long-horizon credit assignment in the Planner level's RSSM

## Limitations & Open Questions
- Meta-gradient computation adds non-trivial training complexity and requires careful implementation to avoid instability
- The potential-based shaping constraint limits the expressiveness of the reward agent—in practice, the constraint is enforced softly (via regularization), not exactly
- Scalability to 4-level hierarchical systems is untested; reward agent interactions across levels could introduce new failure modes
- The framework assumes the sparse reward is well-defined and achievable; for tasks with unclear success criteria, the reward agent has no signal to learn from
