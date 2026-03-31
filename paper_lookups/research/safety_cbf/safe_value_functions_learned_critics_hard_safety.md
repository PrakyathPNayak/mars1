---
## 📂 FOLDER: research/safety_cbf/

### 📄 FILE: research/safety_cbf/safe_value_functions_learned_critics_hard_safety.md

**Title:** Safe Value Functions: Learned Critics as Hard Safety Constraints
**Authors:** Daniel C. H. Tan, Fernando Acero, Robert McCarthy, Stephen J. Roberts, Marc Peter Deisenroth
**Year:** 2023
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2306.04026

**Abstract Summary (2–3 sentences):**
This paper proposes using RL-trained value (critic) functions as Control Barrier Functions for enforcing hard safety constraints. The value function naturally encodes long-horizon safety information through its discounted cumulative cost formulation, enabling CBF-based safety filtering without the need to separately train a barrier function. The method provides formal safety certificates derived from learned value functions and demonstrates reduced conservatism compared to hand-crafted CBFs on continuous control tasks.

**Core Contributions (bullet list, 4–7 items):**
- Establishes theoretical connection between RL value functions and Control Barrier Functions
- Demonstrates that a well-trained safety critic can serve directly as a CBF for hard constraint enforcement
- Provides formal safety certificates (forward invariance guarantees) from learned value functions
- Reduces conservatism compared to hand-crafted CBFs by leveraging the value function's long-horizon reasoning
- Eliminates the need for separate barrier function design or training
- Introduces a verification procedure to certify the learned value function satisfies CBF conditions
- Validates on continuous control benchmarks with formal safety guarantees

**Methodology Deep-Dive (3–5 paragraphs):**
The key theoretical insight is that a value function trained to predict cumulative safety cost (discounted sum of constraint violation indicators) naturally satisfies the structural requirements of a Control Barrier Function. Specifically, if V_safety(s) represents the expected discounted cumulative safety cost from state s under the current policy, then the superlevel set {s : V_safety(s) ≤ threshold} defines a safe set. The value function's Bellman recursion implies a decrease condition analogous to the CBF condition: if the value function decreases along trajectories (indicating the system moves toward safer states), then the safe set is forward invariant. The authors formalize this connection and derive conditions under which V_safety qualifies as a valid CBF.

The practical implementation trains two separate critics using standard RL (e.g., SAC or PPO): a task critic V_task(s) that estimates task performance returns, and a safety critic V_safety(s) that estimates cumulative constraint violation costs. The safety critic is trained on a binary cost signal: c(s) = 1 if the state violates a safety constraint, c(s) = 0 otherwise. Once trained, V_safety is used as the barrier function h(s) = threshold - V_safety(s) in a CBF-QP safety filter. At each timestep, the nominal action from the task policy is filtered through a QP: minimize ||u - u_nominal||² subject to ∂h/∂s · f(s,u) + α(h(s)) ≥ 0, where the dynamics gradients are estimated from the learned model or finite differences.

A critical challenge is verifying that the learned value function actually satisfies CBF validity conditions across the state space, not just on the training distribution. The authors propose a verification procedure that samples states from the safe set boundary and checks: (1) the CBF decrease condition is satisfiable (there exists a feasible control input), and (2) the gradient of V_safety is non-zero at the boundary (ensuring the barrier is well-defined). States that fail verification are added to a buffer for additional training, iteratively improving the value function's reliability as a CBF.

The advantage of using learned value functions over hand-crafted CBFs is that value functions inherently capture long-horizon safety reasoning. A hand-crafted CBF typically encodes instantaneous safety (e.g., distance to obstacle), but the value function accounts for the system's dynamics and future trajectory, resulting in less conservative safety interventions. For example, the value function may allow the system to approach an obstacle more closely if the current dynamics indicate the system can safely decelerate, whereas a distance-based CBF would intervene earlier regardless of velocity.

Experiments are conducted on continuous control benchmarks including cart-pole with safety constraints, point-mass navigation with obstacles, and a simulated manipulator with joint limit constraints. Results show that the value-function-based CBF achieves comparable safety to hand-crafted CBFs while permitting higher task performance due to reduced conservatism. The formal verification procedure confirms CBF validity on 95%+ of sampled boundary states after iterative refinement.

**Key Results & Numbers:**
- Formal safety certificates derived from learned value functions
- Reduced conservatism vs hand-crafted CBFs (10–25% higher task reward at equivalent safety levels)
- 95%+ CBF validity verification rate after iterative refinement
- Zero constraint violations in verified safe set regions
- Compatible with any RL algorithm that trains a critic (SAC, PPO, TD3)
- Verified on continuous control tasks with state-space dimensions up to 20+
- Safety filtering overhead comparable to standard CBF-QP (~1 ms per step)

**Relevance to Project A (Mini Cheetah):** MEDIUM — The value-function-as-CBF concept is applicable to deployment safety for Mini Cheetah, particularly for enforcing joint limits and orientation constraints using the already-trained PPO critic.
**Relevance to Project B (Cassie HRL):** HIGH — Directly informs the LCBF (Learned Control Barrier Function) design in the Cassie safety layer. The idea of using the learned safety critic as the barrier function eliminates separate CBF training and leverages the hierarchy's existing value function infrastructure. The long-horizon safety reasoning is especially valuable for bipedal balance.

**What to Borrow / Implement:**
- Use the safety critic from Cassie's RL training directly as the barrier function h(s) in the CBF-QP layer
- Implement the dual-critic architecture (task + safety) within the Cassie controller level
- Apply the iterative verification procedure to certify the learned barrier function's validity
- Leverage the reduced conservatism for Cassie's dynamic locomotion where overly conservative safety filters would impair agility
- Adapt the threshold-based safe set definition for bipedal balance constraints

**Limitations & Open Questions:**
- Value function approximation errors can lead to invalid barrier certificates in some regions
- Verification procedure is sampling-based and cannot guarantee completeness
- Requires well-trained safety critic — poor critic training leads to unsafe filtering
- Dynamics model or gradient estimation needed for the CBF-QP, adding complexity
- Limited to settings where safety can be expressed as a cumulative cost function
- Scaling to high-dimensional state spaces (e.g., full humanoid) may challenge the verification procedure
---
