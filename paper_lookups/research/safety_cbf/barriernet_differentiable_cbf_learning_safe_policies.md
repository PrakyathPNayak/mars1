---
## 📂 FOLDER: research/safety_cbf/

### 📄 FILE: research/safety_cbf/barriernet_differentiable_cbf_learning_safe_policies.md

**Title:** BarrierNet: Differentiable Control Barrier Functions for Learning of Safe and Robust Robot Control Policies
**Authors:** Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani, Mathias Lechner, Yutong Ban, Chuang Gan, Daniela Rus
**Year:** 2023
**Venue:** IEEE Transactions on Robotics
**arXiv / DOI:** 10.1109/TRO.2023.3249561

**Abstract Summary (2–3 sentences):**
BarrierNet integrates differentiable Control Barrier Functions (CBFs) as a neural network layer, enabling end-to-end learning of safe robot control policies. The CBF layer acts as a differentiable safety filter that minimally modifies the policy's proposed actions to satisfy safety constraints, while gradients flow through the CBF layer during backpropagation to train the policy to proactively avoid unsafe regions. This eliminates the traditional separation between policy learning and safety enforcement, producing policies that are both performant and safe by construction.

**Core Contributions (bullet list, 4–7 items):**
- Introduces a differentiable CBF layer that can be embedded directly into neural network architectures
- Enables end-to-end gradient-based training through the safety filter via implicit differentiation of the QP
- Achieves zero safety violations in 98% of trials with minimal performance degradation (<5%)
- Demonstrates that policies learn to anticipate safety constraints rather than relying solely on last-resort filtering
- Provides theoretical guarantees on forward invariance of the safe set under the learned policy
- Validates on autonomous driving lane-keeping and multi-robot collision avoidance scenarios
- Shows robustness to model uncertainty through the CBF's inherent margin of safety

**Methodology Deep-Dive (3–5 paragraphs):**
The core innovation of BarrierNet is the formulation of a Control Barrier Function constraint as a differentiable optimization layer within a neural network. Given a nominal action proposed by the upstream policy network, the CBF layer solves a Quadratic Program (QP) that finds the closest safe action: minimize ||u - u_nominal||² subject to the CBF constraint ḣ(x) + α(h(x)) ≥ 0, where h(x) is the barrier function, u is the control input, and α is an extended class-K function. The key insight is that this QP, being a convex optimization problem, has a unique solution whose gradients with respect to the QP parameters can be computed via the KKT conditions using implicit differentiation.

The differentiability of the CBF-QP layer is achieved through the OptNet framework for differentiable optimization layers. During the forward pass, the QP is solved using an efficient interior-point or OSQP solver to produce the safe action. During the backward pass, gradients of the loss with respect to the safe action are propagated back through the QP solution by differentiating the KKT optimality conditions. This allows the upstream policy network to receive gradients that encode how safety constraints affect the final action, enabling the policy to learn to propose actions that are already close to safe, reducing the frequency and magnitude of safety interventions.

The barrier function h(x) itself can be either hand-designed based on known safety constraints (e.g., distance to obstacles, joint limits) or learned jointly with the policy. In the learned variant, a separate neural network parameterizes h(x), and its parameters are updated alongside the policy to satisfy the CBF validity conditions: h(x) > 0 in the safe set, h(x) = 0 on the boundary, and the CBF constraint is satisfiable for all states in the safe set. The authors propose a loss function that penalizes CBF validity violations across sampled states, combined with the task performance loss.

Training proceeds using standard policy gradient or imitation learning algorithms, with the CBF layer inserted between the policy network output and the environment. For PPO-based training, the policy network outputs a nominal action distribution, the CBF layer filters the mean action, and the resulting safe action is executed. The critic network is trained without the CBF filter to estimate the value of states under the safe policy. Domain randomization over dynamics parameters ensures the learned CBF margins are robust to model uncertainty.

Experiments demonstrate the approach on lane-keeping for autonomous vehicles and multi-robot collision avoidance. In both settings, BarrierNet achieves near-zero constraint violations while maintaining task performance within 5% of unconstrained baselines. The learned policies exhibit proactive safety behavior — they steer away from constraint boundaries rather than relying on the QP to correct unsafe actions at the last moment, indicating that end-to-end training through the CBF layer fundamentally changes the learned policy's behavior.

**Key Results & Numbers:**
- Zero safety violations in 98% of evaluation trials
- <5% performance degradation compared to unconstrained policy
- Policies proactively avoid unsafe regions rather than relying on last-resort filtering
- QP solve time <1 ms per step, negligible computational overhead
- Robust to ±20% model parameter uncertainty
- Demonstrated on autonomous driving and multi-robot control
- End-to-end training converges in similar wall-clock time as unconstrained training

**Relevance to Project A (Mini Cheetah):** MEDIUM — The differentiable CBF layer concept is applicable to Mini Cheetah deployment safety (joint limits, body orientation constraints), though the quadruped locomotion task may not require the full BarrierNet architecture during initial training.
**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to the LCBF (Learned Control Barrier Function) and CBF-QP safety layer in the Cassie 4-level hierarchy. The differentiable QP formulation enables end-to-end training of the safety layer jointly with the controller, and the implicit differentiation technique is essential for gradient flow through the CBF-QP module.

**What to Borrow / Implement:**
- Adopt the differentiable CBF-QP layer architecture for the Cassie safety layer (Level 4)
- Use implicit differentiation of KKT conditions for gradient flow through the CBF-QP in training
- Implement the joint learning of barrier function h(x) and control policy for the LCBF module
- Apply the proactive safety training paradigm where the controller learns to avoid unsafe regions
- Use the CBF validity loss function to ensure learned barrier functions satisfy theoretical requirements

**Limitations & Open Questions:**
- Requires a differentiable dynamics model or its approximation for computing ḣ(x)
- QP feasibility not guaranteed for all states — infeasible QPs require fallback strategies
- Learned barrier functions may not generalize to out-of-distribution states
- Computational overhead of QP solving scales with number of constraints
- Limited to control-affine systems for standard CBF formulation; extensions needed for general nonlinear systems
- Real-world validation on legged robots not demonstrated in this work
---
