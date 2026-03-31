---
## 📂 FOLDER: research/skill_discovery/

### 📄 FILE: research/skill_discovery/dads_dynamics_aware_unsupervised_skill_discovery.md

**Title:** Dynamics-Aware Discovery of Skills (DADS)
**Authors:** Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, Karol Hausman
**Year:** 2020
**Venue:** ICLR 2020
**arXiv / DOI:** arXiv:1907.01657

**Abstract Summary (2–3 sentences):**
DADS learns skills whose dynamics are predictable by maximizing the conditional mutual information I(z; s'|s) between the skill code z and the next state s' given the current state s. Unlike DIAYN which discovers skills distinguished by the states they visit, DADS discovers skills with predictable state transitions, enabling a learned dynamics model to be used for model-based planning over skills. Demonstrated on a quadruped robot, DADS achieves zero-shot transfer to real hardware and enables hierarchical planning by composing skills through model-predictive control.

**Core Contributions (bullet list, 4–7 items):**
- Proposes a dynamics-aware skill discovery objective: maximize I(z; s'|s) to learn skills with predictable transition dynamics
- Learns a skill-conditioned dynamics model p(s'|s, z) simultaneously with skill discovery, enabling model-based planning over skills
- Enables planning via model-predictive control (MPC) in skill space: compose skills to reach arbitrary goals using the learned dynamics model
- Demonstrates zero-shot sim-to-real transfer on a real quadruped robot (D'Kitty)
- Shows that dynamics-aware skills are superior to DIAYN skills for goal-reaching and navigation tasks
- Provides a continuous skill space (unlike DIAYN's discrete skills), enabling smooth interpolation between behaviors
- Bridges unsupervised skill discovery with model-based RL, combining the benefits of both paradigms

**Methodology Deep-Dive (3–5 paragraphs):**
DADS reformulates the skill discovery objective compared to DIAYN. While DIAYN maximizes I(z; s) — the mutual information between skill and state — DADS maximizes I(z; s'|s) — the conditional mutual information between skill and next state given current state. This distinction is critical: I(z; s) encourages skills to visit different regions of state space (positional diversity), while I(z; s'|s) encourages skills to produce different state transitions from the same starting state (dynamical diversity). A skill in DADS is defined not by where it goes, but by how it moves. This makes DADS skills inherently more suitable for planning, because a dynamics model trained on these skills will be accurate — the skills are optimized to be dynamically predictable.

The objective I(z; s'|s) is maximized through a variational lower bound using a learned dynamics model q_φ(s'|s, z) that approximates the true skill-conditioned transition dynamics p(s'|s, z). The intrinsic reward for the policy is: r(s, a, s', z) = log q_φ(s'|s, z) - log p(s'), where p(s') is approximated by a mixture model. The first term rewards the agent for producing transitions that are predictable by the dynamics model, and the second term encourages visiting diverse next states (not collapsing all skills to the same transition). The policy π_θ(a|s, z) is trained with SAC to maximize this intrinsic reward, while the dynamics model q_φ is trained via maximum likelihood to predict next states given current states and skills.

A key innovation in DADS is the use of a continuous skill space z ∈ R^d (typically d = 2 or 3), sampled from a unit Gaussian or uniform prior, rather than DIAYN's discrete categorical skills. The continuous skill space enables smooth interpolation between behaviors — intermediate skill values produce intermediate locomotion patterns (e.g., interpolating between forward walking and lateral walking produces diagonal walking). This continuity is essential for gradient-based planning: the planner can optimize over the continuous skill space using gradient descent through the learned dynamics model.

For planning and task execution, DADS uses model-predictive control (MPC) in the skill space. Given a desired goal state s_goal, the planner optimizes a sequence of skill codes (z_1, z_2, ..., z_H) over a planning horizon H by simulating trajectories through the learned dynamics model q_φ and minimizing the distance to the goal. At each timestep, the first skill in the planned sequence is executed by the low-level policy π(a|s, z_1), the state is updated, and replanning occurs. This hierarchical planning (MPC over skills, with each skill executed by a neural network policy) provides long-horizon goal-reaching capability that pure model-free RL struggles with. The key advantage is that the dynamics model in skill space is much simpler and more accurate than a raw action-level dynamics model, because skills produce predictable transitions by construction.

The experimental evaluation demonstrates DADS on the D'Kitty quadruped robot (12 DoF) in both simulation and real hardware. In simulation, DADS discovers a variety of locomotion skills (moving in different directions and speeds) and the MPC planner successfully composes them for navigation tasks. Compared to DIAYN, DADS skills enable significantly better goal-reaching performance because the learned dynamics model can accurately predict skill outcomes. For sim-to-real transfer, the policy and dynamics model trained in simulation are deployed directly on the real D'Kitty with no fine-tuning, achieving successful locomotion and goal-reaching. The authors attribute this to the dynamics-aware objective: because skills are optimized for predictable transitions, they tend to be smoother and more conservative, which transfers better to real hardware.

**Key Results & Numbers:**
- Superior goal-reaching vs DIAYN: ~2× higher success rate on navigation tasks
- Zero-shot sim-to-real transfer on D'Kitty quadruped robot (12 DoF)
- Continuous skill space enables smooth interpolation between locomotion behaviors
- MPC planning in skill space achieves long-horizon navigation (10+ meters) that flat RL cannot
- Dynamics model prediction error: ~10% of state magnitude for 1-step prediction
- Demonstrated planning horizon of 10–20 skill steps (each skill executes for 50–100 environment steps)
- Skills are more conservative and stable than DIAYN skills, contributing to sim-to-real transfer success

**Relevance to Project A (Mini Cheetah):** MEDIUM — DADS could discover dynamics-aware locomotion primitives for Mini Cheetah, enabling MPC-based planning over learned skills for navigation tasks. The zero-shot sim-to-real transfer result on a similar quadruped (D'Kitty) is directly encouraging. However, Mini Cheetah's current PPO-based approach directly trains velocity-tracking policies, making the skill discovery layer potentially unnecessary for the primary task.

**Relevance to Project B (Cassie HRL):** HIGH — DADS is directly relevant to the Primitives level of Cassie's 4-level HRL. The dynamics-aware skills produce predictable transitions that the Planner can model and plan over — this is exactly the Planner→Primitives interface in the Cassie system. The continuous skill space enables smooth gait transitions. The MPC-in-skill-space concept maps to the Planner's role: it selects skill parameters based on terrain and task requirements, trusting that the Primitives will produce predictable outcomes. The learned dynamics model q_φ(s'|s, z) could serve as the Planner's world model.

**What to Borrow / Implement:**
- Dynamics-aware skill discovery for the Primitives level: train skills to be dynamically predictable so the Planner can accurately plan over them
- Continuous skill space z ∈ R^d for smooth interpolation between gaits — map skill dimensions to locomotion parameters (speed, direction, gait type)
- Learned skill-conditioned dynamics model as the Planner's internal model for selecting primitives
- MPC-in-skill-space as the Planner's decision-making mechanism: optimize skill sequences to reach terrain-conditioned goals
- The intrinsic reward r = log q(s'|s,z) - log p(s') as the training signal for Cassie's Primitives, replacing hand-crafted per-gait rewards
- The zero-shot transfer result motivates using DADS-style training for sim-to-real transfer of the full HRL system

**Limitations & Open Questions:**
- The learned dynamics model may be inaccurate for long planning horizons, leading to compounding prediction errors
- Continuous skill space may require more training data than discrete skills to cover the behavior space adequately
- Skills are defined over fixed-length execution horizons, which may not align with natural gait cycle durations
- The approach assumes that skill diversity and dynamical predictability are sufficient for downstream tasks — some tasks may require skills with specific properties not captured by the objective
- Integration with safety constraints (LCBF) during both skill discovery and planning is not addressed
- How to warm-start DADS training with prior knowledge about desired locomotion modes (e.g., walking, running) is unexplored
---
