# CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions

**Authors:** Yang et al.
**Year:** 2025 | **Venue:** ICRA
**Links:** [arXiv](https://arxiv.org/abs/2510.14959)

---

## Abstract Summary
CBF-RL proposes a paradigm shift in how safety constraints interact with reinforcement learning: rather than applying Control Barrier Function (CBF) safety filters only at deployment time, CBF-RL enforces CBF constraints during the RL training process itself. The key insight is that if the RL policy learns in the presence of CBF constraints, it internalizes safety-aware behavior, producing policies that are inherently safe without requiring a separate runtime safety filter.

Traditional approaches train an RL policy freely (potentially experiencing unsafe states in simulation), then add a CBF-QP filter at deployment. This creates a distribution mismatch: the policy was trained under different dynamics (unconstrained) than it experiences at deployment (constrained). CBF-RL eliminates this mismatch by embedding the CBF-QP into the training loop. The policy's actions are filtered through the CBF-QP before being applied in simulation, and the filtered actions (not the raw policy outputs) determine the next state. Crucially, the policy gradient is computed with respect to the filtered actions, enabling the policy to learn to work with (rather than against) the safety constraints.

Applied to humanoid robot locomotion and manipulation tasks, CBF-RL demonstrates safer exploration during training (zero constraint violations by construction), faster convergence (the policy doesn't waste samples on unsafe regions), and deployment-ready policies that maintain safety without a separate filter. The approach is compatible with PPO, SAC, and TD3 as base RL algorithms.

## Core Contributions
- Integration of CBF-QP safety filtering into the RL training loop, creating a unified train-and-deploy framework with consistent safety guarantees
- Gradient computation through the CBF-QP layer using differentiable optimization, enabling end-to-end policy gradient updates that account for safety filtering
- Demonstration that training-time safety filtering improves sample efficiency by 20-40% by eliminating exploration in unsafe regions
- Zero constraint violations during training (by construction), enabling safe sim-to-real transfer of training behaviors
- Elimination of deployment-time safety filter: the trained policy inherently respects safety constraints, reducing deployment complexity and latency
- Compatible with multiple RL algorithms (PPO, SAC, TD3) and multiple CBF formulations (standard, high-order, exponential)
- Application to humanoid locomotion tasks demonstrating real-world applicability to legged robots

## Methodology Deep-Dive
The core mechanism wraps the RL policy's action output with a differentiable CBF-QP layer. At each timestep, the policy π_θ(s) outputs a nominal action u_nom. This action is passed through the CBF-QP: u_safe = argmin_u ||u - u_nom||² subject to L_f h_i(x) + L_g h_i(x)·u + α_i(h_i(x)) ≥ 0 for all i ∈ {1,...,k}. The safe action u_safe is applied to the environment, producing the next state s' = f(s, u_safe). The key technical contribution is computing ∂u_safe/∂θ for the policy gradient update.

The gradient through the QP is computed using the implicit function theorem applied to the KKT conditions of the QP. At the optimal solution, the KKT conditions are: u_safe - u_nom + A^T λ = 0 (stationarity), λ_i(A_i u_safe - b_i) = 0 (complementary slackness), A_i u_safe ≥ b_i (primal feasibility), λ_i ≥ 0 (dual feasibility). Differentiating the active constraints through the KKT system yields: ∂u_safe/∂u_nom = I - A_act^T(A_act A_act^T)^{-1} A_act, where A_act is the matrix of active constraint normals. This is the projection matrix onto the feasible action space. The full policy gradient becomes: ∂J/∂θ = ∂J/∂u_safe · ∂u_safe/∂u_nom · ∂u_nom/∂θ, where the middle term is the QP Jacobian.

For implementation, the authors use the OptNet differentiable QP solver or the analytical gradient for simple cases. When no constraints are active (u_nom is already safe), ∂u_safe/∂u_nom = I (identity), and the gradient passes through unmodified. When constraints are active, the gradient is projected to respect the constraint geometry, teaching the policy to avoid actions near constraint boundaries.

The training loop with CBF-RL proceeds as: (1) Sample state s from environment, (2) Compute u_nom = π_θ(s), (3) Solve CBF-QP to get u_safe, (4) Apply u_safe to environment, get reward r and next state s', (5) Store (s, u_safe, r, s') in buffer (or use for on-policy update), (6) Compute policy gradient with QP Jacobian and update θ. The reward function uses the standard task reward without any safety penalty terms, as safety is enforced structurally rather than through reward shaping.

For humanoid locomotion, the CBF constraints include: (1) Joint position limits: h_joint = (q_max - q)(q - q_min) ≥ 0, (2) Joint velocity limits: h_vel = v_max² - v² ≥ 0, (3) Body orientation bounds: h_orient = cos(θ_max) - cos(θ) ≥ 0, (4) Minimum height: h_height = z - z_min ≥ 0, and (5) Foot contact force bounds: h_force = F_max - F_z ≥ 0. These are formulated as HOCBFs where needed (position constraints have relative degree 2 with respect to torque inputs).

A practical consideration is warm-starting the CBF-QP: since consecutive timesteps have similar states, the previous QP solution provides an excellent warm start, reducing solve time from ~100μs to ~20μs. The authors also introduce a constraint relaxation schedule: during early training, CBF constraints are slightly relaxed (α reduced) to allow broader exploration, then tightened to full strictness as training progresses. This "safety curriculum" prevents overly conservative early policies while ensuring full safety by the end of training.

## Key Results & Numbers
- Zero safety constraint violations during training across all experiments (by construction of CBF-QP)
- Sample efficiency improvement: 25-40% fewer environment steps to reach target performance compared to unconstrained training + deployment filter
- Humanoid walking: achieves 95% of unconstrained policy speed while maintaining 100% safety, vs 85% with deployment-only filter (due to distribution mismatch)
- Training-time CBF-QP overhead: 15-20% increase in wall-clock time per step due to QP solving and gradient computation
- Policy deployment without safety filter achieves <0.5% constraint violation rate (vs 0% with filter), demonstrating near-complete internalization
- Convergence speed: 30% faster to stable walking compared to reward-penalty-based safety approaches
- QP solve time: 20-50μs per step with warm starting (compatible with 1kHz control in simulation)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
CBF-RL is directly applicable to Project A's Mini Cheetah training pipeline. Instead of adding safety constraints as reward penalties (which are soft and can be violated) or as a deployment-only filter (which creates distribution mismatch), CBF-RL embeds hard safety constraints into PPO training. For the Mini Cheetah, relevant CBF constraints include joint limits, body orientation (anti-flip), maximum ground reaction forces, and foot collision avoidance.

The 25-40% sample efficiency improvement is significant for Mini Cheetah training, which already requires millions of environment steps. The elimination of the deployment-time safety filter reduces onboard compute requirements and latency, important for Mini Cheetah's embedded controller. The safety curriculum (relaxed → strict constraints) aligns well with Project A's curriculum learning approach, where difficulty and safety strictness can co-evolve.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
CBF-RL is critically relevant to Project B's LCBF (Learned Control Barrier Function) component at the Safety level. The traditional approach for LCBF is to learn a CBF function h(x) and apply it as a deployment filter. CBF-RL suggests a more integrated approach: embed CBF constraints into the Controller-level RL training, allowing the joint-tracking policy to internalize safety constraints from the start.

This could simplify Project B's architecture: rather than maintaining a separate Safety level that filters Controller outputs, the Controller could be trained with CBF-RL to produce inherently safe actions. The HOCBF formulation for Cassie's joint limits (relative degree 2) is directly addressed in the paper. The differentiable QP gradient is compatible with the Dual Asymmetric-Context Transformer architecture—gradients can flow through the QP layer into the Transformer weights. The safety curriculum concept maps to Project B's Adversarial Curriculum, where safety strictness increases alongside environmental difficulty.

## What to Borrow / Implement
- Integrate differentiable CBF-QP into the training loop for both Mini Cheetah (Project A) and Cassie Controller level (Project B)
- Use the implicit function theorem gradient computation for end-to-end policy optimization through safety constraints
- Implement the safety curriculum: start with relaxed CBF constraints (α reduced by 50%), tighten to full strictness over training
- Define specific CBF constraints for each platform: joint limits, orientation bounds, foot clearance, contact forces
- Consider replacing Project B's separate Safety level with CBF-RL-integrated Controller training for architectural simplification

## Limitations & Open Questions
- Differentiable QP gradient computation assumes the active set doesn't change between consecutive gradient steps; rapid active set changes can cause gradient discontinuities
- CBF constraint formulation still requires domain expertise and an accurate dynamics model; learned CBFs introduce approximation errors in the gradient
- Training-time overhead of 15-20% may compound in large-scale distributed training setups
- Near-complete but not perfect safety internalization (<0.5% violation rate without filter) may be insufficient for safety-critical deployments, still requiring a backup filter
