---
## 📂 FOLDER: research/curriculum_learning/

### 📄 FILE: research/curriculum_learning/curricular_hindsight_rl_agile_locomotion.md

**Title:** Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning
**Authors:** Sicen Li, Gang Wang, Peng Zhang, Ruiqi Yu, Lei Han, Yuzhen Liu, Rui Zhao
**Year:** 2024
**Venue:** Scientific Reports (Nature)
**arXiv / DOI:** arXiv:2310.15583

**Abstract Summary (2–3 sentences):**
CHRL (Curricular Hindsight Reinforcement Learning) integrates automatic curriculum learning with Hindsight Experience Replay to train end-to-end legged robot locomotion controllers. The framework automatically adjusts training difficulty based on the robot's current performance, eliminating the need for hand-tuned curriculum schedules. The method achieves autonomous fall recovery, running at 3.45 m/s, and rapid turning on real quadruped hardware.

**Core Contributions (bullet list, 4–7 items):**
- Proposes CHRL, a novel combination of automatic curriculum learning with Hindsight Experience Replay for locomotion
- Eliminates manual curriculum schedule design through performance-based automatic difficulty adjustment
- Demonstrates autonomous fall recovery from arbitrary initial configurations
- Achieves 3.45 m/s outdoor running speed on real quadruped hardware
- Introduces a hindsight relabeling mechanism that repurposes failed experiences as successful ones for easier goals
- Shows rapid turning and agile maneuver capabilities with a single unified policy
- Validates sim-to-real transfer without fine-tuning on physical hardware

**Methodology Deep-Dive (3–5 paragraphs):**
The CHRL framework builds on the Proximal Policy Optimization (PPO) algorithm by wrapping it with two key mechanisms: an automatic curriculum scheduler and a hindsight experience replay (HER) module. The automatic curriculum maintains a distribution over task difficulties (e.g., commanded velocity ranges, terrain complexity) and adaptively shifts this distribution based on the agent's success rate. When the agent consistently succeeds at the current difficulty level, the curriculum advances; when it struggles, the difficulty is reduced. This eliminates the brittle hand-tuned curriculum schedules common in legged locomotion training.

The hindsight experience replay component addresses the sparse reward problem inherent in agile locomotion tasks. When the robot fails to achieve a commanded velocity or maneuver, the trajectory is relabeled with a goal that was actually achieved during that episode. This converts failed episodes into successful training signal, dramatically improving sample efficiency. The relabeling is integrated directly into the PPO training loop by modifying the goal conditioning of the policy and value networks, allowing the agent to learn from every trajectory regardless of whether the original goal was met.

The policy architecture uses an asymmetric actor-critic design where the critic has access to privileged simulation information (ground truth terrain, contact states) while the actor only observes proprioceptive data available on real hardware. The observation space includes joint positions, joint velocities, body orientation, angular velocity, and the commanded velocity goal. Actions are target joint positions sent to PD controllers at 50 Hz. Domain randomization over dynamics parameters (friction, mass, motor strength) is applied throughout training to enable sim-to-real transfer.

Training proceeds in three phases managed by the automatic curriculum: (1) basic standing and walking at low speeds, (2) progressive speed increase up to maximum velocity with directional commands, and (3) recovery scenarios where the robot starts from fallen or perturbed configurations. The curriculum transitions are triggered by a moving-average success metric that must exceed a threshold before advancing. The HER mechanism is particularly critical in phase 3, where fall recovery tasks have extremely sparse rewards under standard training.

Real-world deployment uses a Unitree quadruped robot with 12 degrees of freedom. The learned policy runs directly on the onboard computer at 50 Hz with no additional adaptation or fine-tuning. The authors demonstrate outdoor running at 3.45 m/s, rapid 180-degree turns, and recovery from being pushed or placed upside down, all using the single trained policy.

**Key Results & Numbers:**
- 3.45 m/s maximum outdoor running speed on real hardware
- Autonomous fall recovery from arbitrary initial poses
- Automatic difficulty scaling eliminates manual curriculum tuning
- Single unified policy handles walking, running, turning, and recovery
- Sim-to-real transfer without fine-tuning
- Training converges ~30% faster than standard PPO with manual curriculum
- HER component improves recovery task success rate from ~15% to ~85%

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable curriculum approach for Mini Cheetah training. The automatic difficulty adjustment and HER integration can accelerate PPO training for agile locomotion. The fall recovery capability is especially relevant for robust deployment.
**Relevance to Project B (Cassie HRL):** HIGH — Curriculum strategy is relevant to the adversarial curriculum component of the hierarchical system. The automatic difficulty scheduling could be applied at each level of the 4-level hierarchy to progressively increase task complexity.

**What to Borrow / Implement:**
- Adopt the automatic curriculum scheduler for Mini Cheetah PPO training to eliminate manual schedule tuning
- Integrate HER with PPO for sparse reward tasks like recovery and extreme maneuvers
- Use the performance-based difficulty metric as a template for curriculum progression in both projects
- Apply the asymmetric actor-critic architecture for sim-to-real transfer
- Implement the three-phase curriculum progression (basic→speed→recovery) for Mini Cheetah

**Limitations & Open Questions:**
- HER relabeling assumes goal-conditioned policies; adapting to non-goal-conditioned settings requires modification
- Curriculum thresholds (success rate for advancement) still require some tuning
- Limited to flat/mildly uneven terrain; no structured terrain curriculum
- Real-world experiments limited to a single quadruped platform
- No explicit safety constraints during training or deployment
- Long-term stability over extended operation periods not evaluated
---
