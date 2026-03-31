---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/learning_agile_dynamic_motor_skills_legged_robots.md

**Title:** Learning Agile and Dynamic Motor Skills for Legged Robots
**Authors:** Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, Marco Hutter
**Year:** 2019
**Venue:** Science Robotics, Vol. 4, Issue 26
**arXiv / DOI:** arXiv:1901.08652 / DOI: 10.1126/scirobotics.aau5872

**Abstract Summary (2–3 sentences):**
This pioneering work demonstrates that neural network locomotion policies trained entirely in simulation via reinforcement learning can be successfully transferred to the physical ANYmal quadruped robot. The learned controller achieves faster, more precise, and more energy-efficient locomotion than prior model-based controllers, including autonomous fall recovery capabilities.

**Core Contributions (bullet list, 4–7 items):**
- End-to-end RL-trained locomotion policy for ANYmal deployed via sim-to-real transfer
- Novel actuator network that models series-elastic actuator dynamics with high fidelity
- Domain randomization techniques specifically designed for closing the sim-to-real gap
- Demonstrated faster locomotion than state-of-the-art model-based controllers
- Autonomous fall recovery from arbitrary configurations without explicit recovery logic
- Improved energy efficiency compared to hand-designed controllers
- Comprehensive real-world validation on the ANYmal platform

**Methodology Deep-Dive (3–5 paragraphs):**
The core innovation is training a neural network policy in simulation and transferring it directly to the real ANYmal robot. The policy maps proprioceptive observations (joint positions, velocities, body orientation, and velocity commands) to desired joint positions through a multi-layer perceptron. Training uses PPO with a carefully designed reward function that balances velocity tracking, energy minimization, and motion smoothness.

A critical technical contribution is the actuator network — a learned model of the series-elastic actuator (SEA) dynamics present in ANYmal. Instead of using a simplified analytical model, the authors train a separate neural network to predict actuator torques given desired and actual joint states. This actuator network is integrated into the simulation environment, dramatically improving simulation fidelity and enabling more accurate sim-to-real transfer.

Domain randomization is applied to physical parameters including body mass, center of mass position, joint friction, ground friction, and restitution coefficients. Additionally, observation noise is injected to simulate sensor imperfections. The combination of the accurate actuator model and domain randomization allows the policy to bridge the reality gap effectively.

The training process uses thousands of parallel environments on a single GPU, enabling rapid iteration. The reward function includes terms for linear velocity tracking, angular velocity tracking, torque minimization, joint acceleration penalties, and action smoothness. A curriculum gradually increases terrain difficulty and command velocity ranges during training.

**Key Results & Numbers:**
- Maximum forward velocity of 1.6 m/s (significantly faster than the 0.65 m/s of prior MPC controllers)
- Energy cost of transport reduced by ~50% compared to hand-tuned controllers
- Successful fall recovery from arbitrary poses in under 3 seconds
- Zero-shot sim-to-real transfer with no real-world fine-tuning
- Policy inference at 200 Hz on ANYmal's onboard compute

**Relevance to Project A (Mini Cheetah):** HIGH — Foundational reference for the project's sim-to-real pipeline; the actuator modeling approach is directly applicable to the Mini Cheetah's PD control scheme.
**Relevance to Project B (Cassie HRL):** MEDIUM — The actuator network concept could improve simulation fidelity for Cassie's SEA dynamics, and the sim-to-real methodology is broadly applicable.

**What to Borrow / Implement:**
- Adopt the actuator network approach to model Mini Cheetah's motor dynamics for improved simulation fidelity
- Use the reward function structure (velocity tracking + energy + smoothness) as a starting template
- Apply the domain randomization protocol to both projects for robust sim-to-real transfer

**Limitations & Open Questions:**
- Limited to flat and mildly rough terrain; no perceptive locomotion over highly irregular surfaces
- Actuator network requires real-world actuator data for training, adding a data collection step
- Single-gait policy without explicit gait switching or multi-modal locomotion capabilities
---
