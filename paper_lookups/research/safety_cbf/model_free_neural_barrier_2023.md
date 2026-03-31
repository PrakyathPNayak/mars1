# Model-Free Safe Reinforcement Learning through Neural Barrier Certificate

**Authors:** Yujie Yang, Yuxuan Jiang, Yichen Liu, Jianyu Chen, Shengbo Eben Li
**Year:** 2023 | **Venue:** IEEE Robotics and Automation Letters (RA-L)
**Links:** [Project Page](http://people.iiis.tsinghua.edu.cn/~jychen/publication/2023/ral2023yujie/)

---

## Abstract Summary
This paper addresses a fundamental limitation of most Control Barrier Function (CBF) approaches: the requirement for an explicit dynamics model. In real-world robotics, accurate dynamics models are often unavailable or prohibitively expensive to obtain, particularly for systems with complex contact dynamics like legged robots. The authors propose a model-free framework for learning Neural Barrier Certificates (NBCs) directly from trajectory data, without requiring knowledge of the system dynamics f(x) or g(x).

The key innovation is a multi-step invariant loss that enforces the barrier certificate's forward-invariance condition over trajectory rollouts rather than requiring analytical Lie derivatives. Given a trajectory τ = (s₀, a₀, s₁, a₁, ..., sT), the loss penalizes any transition where the barrier value decreases faster than the allowable rate: L_inv = ∑_t max(0, -(h(s_{t+1}) - h(s_t) + α·h(s_t))). This data-driven loss approximates the continuous-time CBF condition using finite differences, eliminating the need for dynamics knowledge while maintaining the safety guarantee in expectation.

The framework is applied to classic control (CartPole, pendulum), collision avoidance, and autonomous driving scenarios. The authors demonstrate that the model-free NBC achieves comparable safety to model-based CBFs while requiring only trajectory data, and significantly outperforms reward-shaping approaches to safe RL. The multi-step invariant loss also provides robustness to single-step estimation errors by enforcing consistency over multiple timesteps.

## Core Contributions
- Proposes the first fully model-free framework for learning neural barrier certificates from trajectory data, eliminating the dynamics model requirement of standard CBFs
- Introduces a multi-step invariant loss that enforces the CBF decrease condition over trajectory segments, providing robustness to single-step transition noise
- Derives theoretical guarantees showing that the model-free NBC converges to a valid CBF under mild assumptions on trajectory coverage and policy stochasticity
- Demonstrates a joint training procedure that co-optimizes the NBC and the RL policy, with the NBC providing both a safety constraint and a shaping reward
- Shows that multi-step enforcement (k=3-5 steps) significantly improves barrier certificate quality compared to single-step enforcement, reducing false negatives by 40-60%
- Validates on four environments with increasing complexity, achieving zero constraint violations in classic control and <1% violation rate in driving scenarios
- Provides computational analysis showing that model-free NBC training adds only 8-15% overhead to standard RL training

## Methodology Deep-Dive
The neural barrier certificate h_θ(s) is parameterized as a feedforward neural network mapping states to a scalar value. The safe set is defined as C = {s : h_θ(s) ≥ 0}, and the goal is to learn θ such that h_θ satisfies the discrete-time barrier condition: h_θ(s_{t+1}) - h_θ(s_t) + α·h_θ(s_t) ≥ 0 for all transitions under the policy. In the model-based setting, this requires computing ∇h · (f + gu), which needs dynamics knowledge. The model-free approach instead uses observed transitions directly.

The training loss has three components. The classification loss L_cls ensures correct sign assignment: h_θ(s) > 0 for known-safe states and h_θ(s) < 0 for known-unsafe states. Safe/unsafe labels come from environment-defined constraint functions (e.g., joint limits, collision indicators). The single-step invariant loss L_inv1 = E_{(s,a,s')~D}[max(0, -(h_θ(s') - h_θ(s) + α·h_θ(s)))] enforces the barrier condition on individual transitions from the replay buffer D. The multi-step invariant loss L_invk extends this over k consecutive steps: L_invk = E_{τ~D}[∑_{t=0}^{k-1} γ^t · max(0, -(h_θ(s_{t+1}) - h_θ(s_t) + α·h_θ(s_t)))], where γ < 1 discounts future steps. The total loss is L = L_cls + β₁·L_inv1 + β₂·L_invk.

The multi-step loss is the paper's central technical contribution. Single-step enforcement is noisy because the observed transition s → s' may not be representative of the worst-case transition from s. Multi-step enforcement provides two benefits: (1) it averages over multiple transition samples, reducing variance; (2) it enforces a trajectory-level consistency that catches violations missed by single-step checks. Intuitively, a barrier certificate that satisfies the single-step condition at each time point but accumulates small violations will be caught by the multi-step loss, which checks that the barrier value doesn't drift below zero over a k-step horizon.

The joint training procedure alternates between: (1) collecting rollouts using the current policy π_φ with an ε-greedy exploration strategy, (2) updating the NBC h_θ using the three-component loss on the collected data, and (3) updating the policy using PPO with a modified reward r'(s,a) = r(s,a) + λ·h_θ(s) that encourages staying in high-barrier-value regions. Additionally, actions that cause h_θ(s') < 0 are rejected and replaced with a safe fallback (e.g., zero action or the previous action), implementing a model-free safety filter.

The authors analyze the sample complexity of the model-free approach, showing that O(1/(α²ε²)) transitions are needed to guarantee that the learned NBC satisfies the barrier condition with probability 1-ε over the state distribution induced by the policy. This is comparable to model-based CBF learning when dynamics estimation error is accounted for. The key assumption is sufficient coverage: the trajectory data must visit states near the safe set boundary, which is ensured by the ε-greedy exploration and the shaping reward that draws the policy toward boundary regions.

For the safe action rejection mechanism, the paper proposes a computationally efficient approach: rather than solving a QP (which requires dynamics knowledge), the agent simply evaluates h_θ(s') for the predicted next state using a learned one-step predictor ŝ' = f_ψ(s, a). If h_θ(ŝ') < 0, the action is rejected. The one-step predictor is trained on the same trajectory data using a simple MSE loss, requiring no structural knowledge of the dynamics.

## Key Results & Numbers
- CartPole: 0% constraint violations with model-free NBC vs 0% for model-based CBF and 8.3% for reward-shaping baseline; comparable task performance
- Pendulum swing-up: 0.2% violation rate for model-free NBC vs 0% for model-based CBF and 6.7% for constrained RL (CPO)
- 2D collision avoidance: 0.5% violation rate (model-free NBC) vs 0% (model-based CBF) vs 4.2% (CPO) vs 11.3% (reward shaping)
- Autonomous driving (lane keeping + obstacle avoidance): 0.8% violation rate for model-free NBC, competitive with model-based CBF at 0.3%
- Multi-step invariant loss (k=5) reduces barrier condition violations by 55% compared to single-step only (k=1)
- Training overhead: model-free NBC adds 8-15% wall-clock time to standard PPO training; inference overhead is <0.5ms per step
- The learned one-step predictor achieves 97-99% accuracy on transition prediction, enabling effective model-free action rejection

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The model-free NBC approach is attractive for Mini Cheetah deployment because it doesn't require an accurate dynamics model of the real robot (only MuJoCo simulation data during training). The multi-step invariant loss is particularly useful for learning safety constraints that capture the delayed effects of actions—e.g., a torque command that doesn't immediately cause joint limit violation but leads to one 3-5 steps later. The training overhead (8-15%) is acceptable for Mini Cheetah's MuJoCo training pipeline.

For sim-to-real transfer, the model-free NBC trained in simulation may generalize better than model-based CBFs because it doesn't encode specific dynamics assumptions that might not hold on hardware. The one-step predictor used for action rejection can be retrained on a small amount of real-world data to calibrate the safety filter for the physical Mini Cheetah.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The model-free neural barrier certificate framework is directly applicable to training the LCBF in Project B's Safety level. Cassie's contact-rich dynamics make analytical Lie derivative computation impractical, and even learned dynamics models struggle with the discontinuities of foot-ground contact. The model-free approach sidesteps this entirely by learning the barrier certificate from trajectory data.

The multi-step invariant loss is especially valuable for Cassie, where safety violations (falls) typically develop over multiple timesteps: a slightly off-balance state may not be immediately dangerous but becomes critical 5-10 steps later as the center-of-mass diverges. Using k=5-10 step enforcement in the invariant loss would capture these delayed-onset safety violations. The joint training procedure (NBC + policy co-optimization) integrates naturally with Project B's hierarchical training: the Safety level's NBC can be co-optimized with the Controller level's policy, with the multi-step loss computed over trajectory segments from the Controller level's rollouts.

The model-free action rejection mechanism is a practical complement to the differentiable CBF-QP approach (BarrierNet): during early training when the CBF-QP may be poorly calibrated, the trajectory-based rejection provides a fallback safety mechanism. As training progresses and the LCBF improves, the QP-based correction becomes primary and the rejection mechanism serves as a secondary check.

## What to Borrow / Implement
- Adopt the multi-step invariant loss (k=5-10) for training Project B's LCBF, capturing delayed-onset safety violations in Cassie's dynamics
- Use the joint NBC + policy co-optimization procedure to train the Safety level and Controller level simultaneously
- Implement the model-free action rejection as a secondary safety check alongside the primary CBF-QP correction in Project B's Safety level
- Apply the trajectory-level barrier condition enforcement during domain randomization training for both projects, ensuring safety robustness across dynamics variations
- Use the shaping reward r' = r + λ·h(s) to encourage policies to maintain high safety margins even when not near constraint boundaries

## Limitations & Open Questions
- The model-free approach provides expected-case rather than worst-case safety guarantees; adversarial perturbations or distributional shift can cause violations
- The one-step predictor used for action rejection introduces model dependence through the back door, partially undermining the "model-free" claim
- Sample efficiency is lower than model-based CBFs: ~2-3x more training data needed to achieve comparable barrier quality, which is acceptable in simulation but costly for real-world data
- The multi-step loss introduces a new hyperparameter k (number of steps) that must be tuned per environment; too large k leads to overly conservative barriers, too small misses delayed violations
