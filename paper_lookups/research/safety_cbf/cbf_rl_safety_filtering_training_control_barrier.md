---
## 📂 FOLDER: research/safety_cbf/

### 📄 FILE: research/safety_cbf/cbf_rl_safety_filtering_training_control_barrier.md

**Title:** CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions
**Authors:** Yang Luo, Dong Xu, Yitong Sun, et al.
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv (2025)

**Abstract Summary (2–3 sentences):**
CBF-RL integrates Control Barrier Function safety filtering during RL training, not just at deployment time, enabling safe exploration throughout the learning process. The method is demonstrated on a Unitree G1 humanoid robot, achieving zero falls during training and 99% safety rate during real-world deployment. The CBF filter minimally perturbs RL actions only when safety constraints are at risk, preserving 90% of the speed achieved by unconstrained RL.

**Core Contributions (bullet list, 4–7 items):**
- Introduces CBF safety filtering during RL training to enable safe exploration
- Demonstrates zero falls during the entire training process on a humanoid robot
- Achieves 99% safety rate in real-world deployment on Unitree G1 humanoid
- Maintains 90% of unconstrained RL walking speed despite safety constraints
- Proposes a minimal intervention principle — CBF only modifies actions when necessary
- Shows that safe training produces more robust policies than unsafe training + deployment filtering
- Validates both sim-to-real transfer and direct real-world training with safety guarantees

**Methodology Deep-Dive (3–5 paragraphs):**
CBF-RL modifies the standard RL training loop by inserting a CBF-based safety filter between the policy output and the environment. At each training timestep, the policy network proposes an action u_nominal, and a CBF-QP is solved to produce the minimally modified safe action: u_safe = argmin ||u - u_nominal||² subject to Lf·h(x) + Lg·h(x)·u + α(h(x)) ≥ 0, where h(x) is the control barrier function encoding safety constraints (e.g., center-of-mass height, body orientation limits, foot contact constraints for the humanoid). The safe action u_safe is then executed in the environment, and the resulting transition (s, u_safe, r, s') is stored in the replay buffer for policy updates.

A critical design choice is how the CBF intervention affects the policy gradient computation. The authors explore two variants: (1) the policy is updated using the nominal action u_nominal in the gradient computation (ignoring the CBF's modification), and (2) the policy is updated using the actual executed action u_safe. Variant (1) allows the policy to explore the full action space in gradient space while remaining safe in execution, effectively decoupling exploration from safety. Variant (2) trains the policy to produce actions that are already safe, similar to BarrierNet but without requiring differentiability of the QP. The paper finds that variant (1) produces more agile policies while variant (2) produces more conservative but reliably safe policies.

The barrier function h(x) for the humanoid is constructed from multiple safety constraints combined using smooth minimum functions. The individual constraints include: minimum center-of-mass height (prevents falling), maximum body tilt angle (prevents toppling), minimum foot-ground clearance during swing phase (prevents tripping), and maximum joint velocity (prevents hardware damage). Each constraint is formulated as h_i(x) ≥ 0 for safety, and the composite barrier h(x) = smooth_min(h_1(x), ..., h_n(x)) ensures all constraints are simultaneously satisfied. The dynamics model required for the CBF (Lf·h and Lg·h) is obtained from the robot's URDF-based rigid body dynamics, computed using Pinocchio or similar libraries.

The minimal intervention principle is central to the approach: the QP objective minimizes the distance between the safe and nominal actions, ensuring that the CBF only modifies the policy's output when a safety violation is imminent. This contrasts with reward-shaping approaches that penalize unsafe behavior but provide no hard guarantees. The authors show that minimal intervention preserves the policy's exploration efficiency — the CBF intervenes in only 5–15% of training timesteps on average, and the magnitude of intervention decreases as the policy learns to avoid unsafe regions.

Real-world experiments on the Unitree G1 humanoid validate the approach. The robot is trained in simulation with the CBF filter, transferred to hardware, and deployed with the same CBF filter active. During training in simulation, zero constraint violations occur across 100 million timesteps. In real-world deployment, the robot achieves a 99% safety rate (defined as no falls or joint limit violations per episode) while walking at 90% of the speed achieved by unconstrained RL. The 1% failure rate is attributed to extreme perturbations (pushes) that exceed the CBF's margin of safety.

**Key Results & Numbers:**
- Zero falls during the entire RL training process (~100M timesteps)
- 99% safety rate in real-world deployment on Unitree G1 humanoid
- 90% speed retention compared to unconstrained RL
- CBF intervenes in only 5–15% of training timesteps
- Intervention magnitude decreases over training as policy learns safety
- Sim-to-real transfer successful with same CBF filter active
- Composite barrier function from 4+ individual safety constraints

**Relevance to Project A (Mini Cheetah):** MEDIUM — Safe training is applicable to Mini Cheetah RL training, especially for preventing joint damage or falls during sim training. The minimal intervention CBF could be adapted for the quadruped's safety constraints.
**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to the safety layer (Level 4) design of the Cassie hierarchy. Demonstrates CBF integration during both training and deployment for a bipedal robot, validating the approach for the CBF-QP safety module. The composite barrier function construction and minimal intervention principle are directly applicable.

**What to Borrow / Implement:**
- Implement CBF safety filtering during Cassie RL training at the controller level (Level 3)
- Adopt the composite barrier function construction (smooth minimum of individual constraints) for Cassie
- Use the minimal intervention QP formulation for the CBF-QP safety layer
- Apply variant (1) — nominal action gradients with safe execution — for maximum agility in Cassie
- Construct Cassie-specific barriers for COM height, body tilt, foot clearance, and joint velocity limits
- Use rigid body dynamics (from Cassie URDF) for CBF Lie derivative computation

**Limitations & Open Questions:**
- Requires accurate dynamics model for CBF Lie derivatives — model errors can compromise safety
- Composite barrier function (smooth min) can be overly conservative at constraint intersections
- CBF margin of safety must be tuned — too large reduces performance, too small risks violations
- Does not handle state estimation errors, which are significant on real hardware
- Fixed barrier function — no online adaptation to changing conditions
- Scaling to higher-dimensional action spaces increases QP solve time
---
