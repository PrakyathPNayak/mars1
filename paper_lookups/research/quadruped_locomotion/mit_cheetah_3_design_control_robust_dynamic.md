---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/mit_cheetah_3_design_control_robust_dynamic.md

**Title:** MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot
**Authors:** Gerardo Bledt, Matthew J. Powell, Benjamin Katz, Jared Di Carlo, Patrick M. Wensing, Sangbae Kim
**Year:** 2018
**Venue:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018)
**arXiv / DOI:** DOI: 10.1109/IROS.2018.8593885

**Abstract Summary (2–3 sentences):**
This paper presents the design and control architecture of the MIT Cheetah 3, a robust 12-DOF quadruped robot featuring high-bandwidth proprioceptive actuators. The robot demonstrates robust blind locomotion including stair climbing without external sensors, achieving dynamic gaits (trot, bound, gallop, pronk) with a remarkably low cost of transport through a modular control architecture based on model-predictive control and whole-body impulse control.

**Core Contributions (bullet list, 4–7 items):**
- Proprioceptive actuator design with high bandwidth and force transparency for dynamic locomotion
- Expanded range of motion in hip abduction/adduction for improved lateral stability
- Modular control architecture enabling reactive gait modification without exteroceptive sensing
- Cost of Transport (CoT) as low as 0.45 during trotting — exceptional energy efficiency
- Blind stair climbing using proprioceptive feedback alone
- Multiple dynamic gaits: trot, bound, gallop, pronk, pace, and flying trot
- Robust disturbance rejection and balance recovery under external perturbations

**Methodology Deep-Dive (3–5 paragraphs):**
The MIT Cheetah 3 represents a hardware-software co-design philosophy where the mechanical platform is specifically optimized for learning-compatible control. The robot has 12 degrees of freedom (3 per leg: hip abduction/adduction, hip flexion/extension, knee flexion/extension) driven by custom proprioceptive actuators. These actuators use high-torque-density motors with low-ratio gearing, providing excellent force transparency and bandwidth — critical for dynamic locomotion and impedance control.

The control architecture is hierarchical: a high-level gait scheduler and state machine determines the desired contact schedule and body trajectory, while a mid-level convex model-predictive controller (MPC) computes optimal ground reaction forces over a planning horizon. A low-level whole-body impulse controller then maps these desired forces to individual joint torques while respecting friction cone constraints and joint limits.

The MPC formulation uses a simplified single rigid body dynamics model for computational efficiency, solving a convex optimization at 30 Hz. This allows real-time re-planning of ground reaction forces in response to unexpected terrain or external disturbances. The whole-body controller operates at 1 kHz, providing fast reactive control at the joint level.

Blind locomotion capability is achieved through the robot's proprioceptive sensing (joint encoders and IMU) combined with reactive controllers. When the robot encounters unexpected terrain changes (e.g., a step), the contact detection from proprioceptive signals triggers adaptive behavior without requiring a terrain map or prior knowledge. This reactive approach is complemented by robust mechanical design that provides passive stability margins.

**Key Results & Numbers:**
- Cost of Transport: 0.45 (trotting at optimal speed)
- Maximum speed: ~3.7 m/s (galloping)
- Blind stair climbing: successful on standard indoor stairs (17 cm rise, 28 cm run)
- Disturbance rejection: maintained balance under lateral pushes up to 200 N
- Control loop: MPC at 30 Hz, joint-level control at 1 kHz (500 Hz PD)
- Weight: ~45 kg with full electronics and battery

**Relevance to Project A (Mini Cheetah):** HIGH — Direct predecessor platform; the control architecture, actuator design philosophy, and PD control at 500 Hz are directly inherited by the Mini Cheetah project.
**Relevance to Project B (Cassie HRL):** LOW — Quadruped-specific hardware design, though the MPC + whole-body control hierarchy provides relevant architectural inspiration.

**What to Borrow / Implement:**
- Use the Cheetah 3's control frequency specification (500 Hz PD) as the baseline for the Mini Cheetah RL policy
- The reactive blind locomotion philosophy can inform the design of proprioceptive-only locomotion modes
- The MPC formulation provides a useful comparison baseline against RL-based controllers

**Limitations & Open Questions:**
- Relies on hand-designed control architecture rather than end-to-end learning
- Blind locomotion fails on highly irregular terrain requiring foot placement planning
- No learning-based adaptation — performance is limited by the fixed MPC model assumptions
---
