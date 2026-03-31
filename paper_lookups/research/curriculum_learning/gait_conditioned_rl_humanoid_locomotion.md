---
## 📂 FOLDER: research/curriculum_learning/

### 📄 FILE: research/curriculum_learning/gait_conditioned_rl_humanoid_locomotion.md

**Title:** Gait-Conditioned Reinforcement Learning for Humanoid Locomotion
**Authors:** Zhi-Hao Lin, Paul Makoviichuk, et al.
**Year:** 2024
**Venue:** arXiv / ICRA 2025
**arXiv / DOI:** arXiv (2024)

**Abstract Summary (2–3 sentences):**
This paper presents a gait-conditioned RL framework where gait phase, frequency, and contact schedule are explicitly provided as conditioning inputs to the locomotion policy. By conditioning the policy on gait parameters, a single network can produce multiple gait modes (walk, trot, run) with smooth transitions between them. The approach uses a curriculum over gait parameters during training and is validated on both simulated and real humanoid robots.

**Core Contributions (bullet list, 4–7 items):**
- Introduces explicit gait conditioning (phase, frequency, contact schedule) as policy inputs
- Enables multi-gait locomotion from a single unified policy network
- Demonstrates smooth gait transitions (walk→trot→run) through continuous gait parameter modulation
- Achieves improved energy efficiency through gait-appropriate locomotion patterns
- Uses curriculum-based training over gait parameters for stable multi-gait learning
- Validates on simulation and real-world humanoid robot deployment
- Shows that gait conditioning acts as a strong inductive bias, accelerating training convergence

**Methodology Deep-Dive (3–5 paragraphs):**
The core idea is to augment the locomotion policy's observation space with explicit gait parameters that define the desired locomotion pattern. The gait conditioning vector includes: (1) gait phase θ ∈ [0, 2π) — the current position in the gait cycle, advanced at each timestep by Δθ = 2π·f·dt where f is the gait frequency; (2) gait frequency f — the number of gait cycles per second, controlling speed; (3) contact schedule — a binary vector indicating which feet should be in contact with the ground at the current phase (e.g., [1,0] for single support, [1,1] for double support); and (4) duty factor — the fraction of the gait cycle spent in stance phase, distinguishing walking (high duty factor) from running (low duty factor).

The policy network is a standard MLP that takes as input the concatenation of proprioceptive observations (joint positions, velocities, body orientation, angular velocity) and the gait conditioning vector. The output is joint position targets for PD controllers. Crucially, the gait conditioning is not just a label — it provides time-varying phase information that the policy can use to generate periodic motions. The contact schedule serves as a soft constraint that is reinforced through a contact reward: r_contact = Σ_i (c_i · s_i + (1-c_i) · (1-s_i)), where c_i is the actual foot contact and s_i is the scheduled contact for foot i at the current phase.

Training uses PPO with a curriculum over gait parameters. The curriculum starts with a single gait mode (e.g., slow walking with f=1 Hz and high duty factor) and progressively expands the range of gait parameters. Phase 1 trains walking gaits (f ∈ [0.8, 1.2] Hz, duty factor ∈ [0.6, 0.7]). Phase 2 introduces trotting (f ∈ [1.2, 2.0] Hz, duty factor ∈ [0.4, 0.6]). Phase 3 adds running (f ∈ [2.0, 3.0] Hz, duty factor ∈ [0.2, 0.4]). At each phase, the gait parameters are uniformly sampled from the expanded range, and training continues until the policy achieves a success rate threshold before advancing. This curriculum is critical — training on all gait modes simultaneously from the start leads to mode collapse where the policy only learns one gait.

Gait transitions are achieved by smoothly interpolating the gait conditioning vector between source and target gait parameters. For example, transitioning from walk to run involves linearly interpolating the frequency from 1.0 to 2.5 Hz and the duty factor from 0.65 to 0.3 over a transition period of 1–2 seconds. Because the policy is conditioned on these parameters, it smoothly adapts its behavior to the changing gait specification. The phase signal θ is continuously updated throughout, ensuring the legs maintain proper coordination during transitions.

Experiments demonstrate that gait conditioning significantly improves locomotion quality compared to unconditioned policies. The conditioned policy achieves 15–25% better energy efficiency (cost of transport) because it can match the gait pattern to the commanded speed, rather than using a single compromise gait for all speeds. Real-world deployment on a humanoid robot confirms that gait transitions are smooth and stable, with the robot switching between walking, trotting, and running based on speed commands.

**Key Results & Numbers:**
- Smooth transitions between walk, trot, and run from a single policy
- 15–25% improved energy efficiency through gait-appropriate locomotion
- Multi-gait policy trained with gait parameter curriculum
- Curriculum prevents mode collapse during multi-gait training
- Contact schedule reward improves foot timing accuracy by ~30%
- Real-world humanoid deployment validates smooth gait transitions
- Training converges 2x faster with gait conditioning vs unconditioned baseline

**Relevance to Project A (Mini Cheetah):** HIGH — Gait conditioning is directly applicable to multi-gait Mini Cheetah locomotion. The phase, frequency, and contact schedule conditioning can enable walk/trot/gallop from a single policy, improving the versatility of the 12 DoF PPO-trained controller.
**Relevance to Project B (Cassie HRL):** HIGH — Gait phase conditioning is directly relevant to the Neural ODE Gait Phase module in the Cassie hierarchy. The explicit phase signal θ and contact schedule align with how the Neural ODE generates phase-varying reference trajectories for the controller level.

**What to Borrow / Implement:**
- Implement gait conditioning (phase, frequency, contact schedule) as inputs to Mini Cheetah policy
- Use the gait parameter curriculum (walk→trot→run) for multi-gait Mini Cheetah training
- Adopt the contact schedule reward for improving foot timing accuracy
- Map the gait phase signal to the Neural ODE Gait Phase module's phase variable in Cassie
- Use the duty factor parameter to distinguish walking/running modes in both projects
- Apply the smooth gait transition mechanism for the Cassie primitives→controller interface

**Limitations & Open Questions:**
- Gait parameters must be externally specified — no automatic gait selection based on terrain
- Contact schedule assumes flat ground; adaptation to uneven terrain requires modification
- Phase tracking can drift under perturbations, requiring phase reset mechanisms
- Limited to periodic gaits — aperiodic motions (jumping, recovery) not well-captured
- Curriculum over gait parameters requires careful stage design and threshold tuning
- Energy efficiency improvements are gait-dependent and may not generalize to all robot morphologies
---
