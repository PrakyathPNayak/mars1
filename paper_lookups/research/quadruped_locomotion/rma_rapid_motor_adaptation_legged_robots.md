---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/rma_rapid_motor_adaptation_legged_robots.md

**Title:** RMA: Rapid Motor Adaptation for Legged Robots
**Authors:** Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
**Year:** 2021
**Venue:** RSS 2021 (Robotics: Science and Systems)
**arXiv / DOI:** arXiv:2107.04034

**Abstract Summary (2–3 sentences):**
RMA introduces a two-component architecture for real-time online adaptation of quadruped robots to unseen environments. A base policy trained via model-free RL in simulation is paired with an adaptation module that infers latent extrinsic environmental parameters from recent proprioceptive history, enabling sub-second adaptation to novel terrain, payloads, and dynamics without any real-world fine-tuning.

**Core Contributions (bullet list, 4–7 items):**
- Two-phase architecture: base policy (RL) + adaptation module (supervised learning) for online environmental adaptation
- Real-time inference of extrinsic parameters (friction, mass, terrain) from proprioceptive history
- Asynchronous execution: adaptation module at 10 Hz, base policy at 100 Hz for computational efficiency
- Zero-shot sim-to-real transfer on Unitree A1 without any real-world data or calibration
- Robust locomotion over sand, mud, grass, gravel, stairs, and carrying unexpected payloads
- No hand-coded reference trajectories or specialized recovery behaviors needed
- Sub-second adaptation to abrupt environmental changes

**Methodology Deep-Dive (3–5 paragraphs):**
RMA's training proceeds in two phases. In Phase 1, the base policy is trained using PPO in simulation with access to privileged environment parameters (termed "extrinsics") such as ground friction, restitution, terrain slope, and added mass. These extrinsics are encoded into a compact latent vector via a learned encoder. The base policy takes as input the robot's proprioceptive state concatenated with this latent extrinsic vector and outputs joint position targets.

In Phase 2, an adaptation module is trained via supervised learning to predict the extrinsic latent vector from a window of recent proprioceptive observations and actions (typically the last 50 timesteps). This module replaces the privileged encoder at deployment time, enabling the robot to infer environmental conditions from interaction history alone. The key insight is that terrain and dynamics properties leave observable traces in the proprioceptive signal over time.

The simulation environment uses a procedural terrain generator with randomized ground properties, slopes, stairs, and obstacles. Domain randomization extends to motor parameters, sensor noise, latency, and body mass. The reward function encourages velocity tracking, penalizes energy consumption, and discourages unstable behaviors like large body rotations and foot slippage.

At deployment, the adaptation module runs asynchronously at a lower frequency (10 Hz) than the base policy (100 Hz), making it feasible to run on the limited onboard compute of affordable robots. The base policy continuously uses the most recent latent vector from the adaptation module, allowing smooth real-time adaptation without interrupting locomotion.

**Key Results & Numbers:**
- Successfully deployed on Unitree A1 across 8+ real-world terrain types with zero-shot transfer
- Adapted to carrying payloads up to 100% of robot body weight within ~0.2 seconds
- Achieved stable locomotion at speeds up to 1.0 m/s on deformable terrain (sand, mud)
- Outperformed prior state-of-the-art (including domain randomization-only baselines) by 30–50% in success rate on challenging terrains
- Adaptation latency: ~0.2 seconds to converge to new environmental parameters

**Relevance to Project A (Mini Cheetah):** HIGH — The RMA architecture is a foundational reference for the Mini Cheetah project's adaptive locomotion pipeline and sim-to-real transfer strategy.
**Relevance to Project B (Cassie HRL):** HIGH — The adaptation module concept directly maps to the controller level's need for real-time environmental adaptation in the hierarchical architecture.

**What to Borrow / Implement:**
- Implement the two-phase training pipeline (privileged RL + adaptation module distillation) for both projects
- Use the asynchronous execution scheme to balance compute between high-level adaptation and low-level control
- Adopt the extrinsic parameter encoding as input to the terrain-adaptive components of the Cassie HRL system

**Limitations & Open Questions:**
- Adaptation module accuracy degrades for environmental parameters that have weak proprioceptive signatures (e.g., distant terrain features)
- The system assumes quasi-static environmental changes; rapid terrain transitions can cause transient instability
- Limited to proprioceptive adaptation — does not leverage vision or other exteroceptive modalities
---
