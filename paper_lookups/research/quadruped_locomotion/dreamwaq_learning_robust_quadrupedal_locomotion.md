---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/dreamwaq_learning_robust_quadrupedal_locomotion.md

**Title:** DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning
**Authors:** I Made Aswin Nahrendra, Byeongho Yu, Hyun Myung
**Year:** 2023
**Venue:** ICRA 2023 (IEEE International Conference on Robotics and Automation)
**arXiv / DOI:** arXiv:2301.10602 / DOI: 10.1109/ICRA48891.2023.10161144

**Abstract Summary (2–3 sentences):**
DreamWaQ proposes a deep RL-based locomotion framework that enables quadruped robots to implicitly "imagine" terrain features using only proprioceptive sensing, eliminating reliance on exteroceptive sensors like cameras or LiDAR. The method achieves robust and adaptive locomotion over diverse, unstructured real-world terrains within a single continuous run, validated through long-distance outdoor experiments.

**Core Contributions (bullet list, 4–7 items):**
- Implicit terrain imagination module that infers terrain properties from proprioceptive history alone
- Eliminates the need for exteroceptive sensors (cameras, LiDAR) for rough terrain traversal
- Single-policy approach that handles diverse terrains without explicit terrain classification
- Real-world validation with long-distance outdoor traversal experiments on challenging surfaces
- Open-source implementation built on Isaac Gym, Legged Gym, and RSL-RL frameworks
- Demonstrates robustness to sensor degradation scenarios

**Methodology Deep-Dive (3–5 paragraphs):**
DreamWaQ addresses a fundamental limitation of proprioceptive-only locomotion policies: without explicit terrain sensing, the robot cannot anticipate upcoming terrain changes. The key insight is that terrain properties leave detectable signatures in the robot's proprioceptive history (joint positions, velocities, torques, and IMU readings). The framework learns to extract implicit terrain representations from a sliding window of proprioceptive observations.

The architecture consists of a terrain imagination module that processes temporal sequences of proprioceptive data to produce a latent terrain embedding. This embedding is concatenated with the current proprioceptive state and velocity commands to form the input to a locomotion policy network. The entire system is trained end-to-end using PPO in the Isaac Gym simulator with extensive domain randomization over terrain types, friction coefficients, and robot dynamics.

Training employs a teacher-student paradigm where a privileged teacher policy has access to ground-truth terrain information during training in simulation. The student policy (deployed on the real robot) learns to reconstruct the teacher's terrain-conditioned behavior using only proprioceptive signals. This distillation process enables the student to develop implicit terrain understanding.

The reward function balances velocity tracking, energy efficiency, smoothness of motion, and stability metrics. Domain randomization is applied to masses, friction coefficients, motor strengths, sensor noise, and terrain geometry to ensure robust sim-to-real transfer.

**Key Results & Numbers:**
- Successfully traversed grass, gravel, slopes, stairs, and uneven outdoor terrain in continuous runs exceeding 500 meters
- Maintained stable locomotion at commanded velocities up to 1.5 m/s across mixed terrain
- Outperformed baseline proprioceptive policies (without terrain imagination) on rough terrain success rates by ~25%
- Zero-shot sim-to-real transfer with no fine-tuning required on real hardware

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable as a proprioceptive locomotion framework for sim-to-real quadruped control with implicit terrain adaptation.
**Relevance to Project B (Cassie HRL):** MEDIUM — The terrain imagination concept could be adapted as a terrain encoder within the hierarchical controller's planner level.

**What to Borrow / Implement:**
- Adopt the implicit terrain imagination architecture as a baseline for proprioceptive-only locomotion
- Use the teacher-student distillation pipeline for bridging privileged simulation information to deployable policies
- Integrate the terrain embedding as an input feature for the adaptive controller module

**Limitations & Open Questions:**
- Performance degrades significantly on highly discontinuous terrain (e.g., large gaps) where anticipation requires exteroception
- The proprioceptive history window length is a critical hyperparameter that requires careful tuning
- No explicit mechanism for failure detection or recovery from falls
---
