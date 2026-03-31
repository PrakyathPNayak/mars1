---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/learning_quadrupedal_locomotion_challenging_terrain.md

**Title:** Learning Quadrupedal Locomotion over Challenging Terrain
**Authors:** Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter
**Year:** 2020
**Venue:** Science Robotics, Vol. 5, Issue 47
**arXiv / DOI:** arXiv:2010.11251 / DOI: 10.1126/scirobotics.abc5986

**Abstract Summary (2–3 sentences):**
This work from ETH Zurich demonstrates that a quadruped robot can learn to locomote robustly over highly challenging, unstructured real-world terrains (mud, snow, rubble, water, mountain trails) using only proprioceptive sensing. The method introduces a teacher-student training paradigm with a terrain-adaptive curriculum that achieves remarkable zero-shot sim-to-real transfer on the ANYmal robot.

**Core Contributions (bullet list, 4–7 items):**
- Teacher-student (privileged learning) training paradigm for bridging sim-to-real gap
- Temporal Convolutional Network (TCN) for processing proprioceptive history sequences
- Adaptive terrain curriculum with particle-filter-based difficulty scheduling
- Purely proprioceptive "blind" locomotion achieving robust performance on extreme terrains
- Zero-shot transfer to ANYmal on snow, mud, rubble, water, and mountain trails
- Implicit contact and slippage detection through learned temporal features
- Comprehensive real-world deployment across multiple outdoor environments

**Methodology Deep-Dive (3–5 paragraphs):**
The training pipeline uses a two-stage approach. First, a privileged "teacher" policy is trained in simulation with access to ground-truth terrain information (height map under and around the robot's feet). This teacher policy learns to navigate diverse terrains effectively by exploiting this information. The teacher is trained using PPO with a reward function that combines velocity tracking, energy minimization, and stability terms.

Second, a "student" policy is trained to imitate the teacher using only proprioceptive signals available on the real robot. The student uses a Temporal Convolutional Network (TCN) that processes a sliding window of proprioceptive observations (joint positions, velocities, torques, body orientation, and angular velocity). The TCN architecture enables the student to extract temporal patterns that implicitly encode terrain information — for example, detecting foot slippage from joint velocity anomalies or inferring terrain softness from torque feedback patterns.

The terrain curriculum is a key innovation. Instead of randomly sampling terrain difficulty, the system uses a particle-filter-based approach to adaptively adjust terrain complexity. Each "particle" represents a terrain difficulty level, and particles are resampled based on the robot's traversability (success rate) at each difficulty. This creates a natural curriculum that pushes the robot to progressively harder terrains while maintaining sufficient successful episodes for learning.

Domain randomization covers body mass, friction coefficients, restitution, motor strength, observation noise, and action latency. The simulation uses randomized terrain profiles including stairs, slopes, rough terrain, and gaps of varying difficulty levels.

**Key Results & Numbers:**
- Deployed on ANYmal across 6+ real-world terrain types with zero-shot transfer
- Successfully traversed rubble fields, flowing water, thick mud, deep snow, and steep mountain trails
- Maintained stable locomotion at speeds up to 1.2 m/s on unstructured outdoor terrain
- Teacher policy achieved >95% success rate on the most difficult simulation terrains
- Student policy retained ~90% of teacher performance using proprioceptive signals alone
- Demonstrated autonomous outdoor navigation for distances exceeding 1 km

**Relevance to Project A (Mini Cheetah):** HIGH — The teacher-student paradigm and terrain curriculum are directly applicable to the Mini Cheetah's training pipeline for robust outdoor locomotion.
**Relevance to Project B (Cassie HRL):** HIGH — The privileged learning framework maps directly to the asymmetric-context approach in the Cassie system, and the terrain curriculum aligns with the adversarial curriculum component.

**What to Borrow / Implement:**
- Implement the teacher-student distillation pipeline with privileged terrain information for both projects
- Adopt the particle-filter-based terrain curriculum for progressive difficulty scaling
- Use the TCN architecture for proprioceptive history encoding in the controller module

**Limitations & Open Questions:**
- Blind locomotion fails on terrain requiring anticipatory foot placement (large gaps, stepping stones)
- The TCN window length creates a fundamental trade-off between adaptation speed and stability
- No recovery mechanism for catastrophic failures (falls from large height differences)
---
