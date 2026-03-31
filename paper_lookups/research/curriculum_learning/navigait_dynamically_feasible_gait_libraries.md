---
## 📂 FOLDER: research/curriculum_learning/

### 📄 FILE: research/curriculum_learning/navigait_dynamically_feasible_gait_libraries.md

**Title:** NaviGait: Navigating Dynamically Feasible Gait Libraries using Deep Reinforcement Learning
**Authors:** Jun Wang, et al.
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2510.11542

**Abstract Summary (2–3 sentences):**
NaviGait combines an offline-optimized gait library, generated using Hybrid Zero Dynamics (HZD), with a hierarchical deep RL agent that learns to navigate and interpolate through the library online. The RL agent selects and blends pre-computed dynamically feasible gaits in real-time, providing robust and interpretable locomotion with formal dynamic feasibility guarantees. The approach achieves faster adaptation than end-to-end RL while maintaining the interpretability and theoretical guarantees of model-based gait planning.

**Core Contributions (bullet list, 4–7 items):**
- Combines offline model-based gait optimization (HZD) with online RL-based gait selection
- Maintains dynamic feasibility guarantees from the pre-computed gait library
- Achieves faster online adaptation than pure end-to-end RL approaches
- Provides interpretable gait selection — each library entry corresponds to a named gait pattern
- Demonstrates robust locomotion on bipedal robots with smooth gait transitions
- Introduces a gait interpolation mechanism for continuous navigation through discrete library entries
- Bridges the gap between model-based and learning-based locomotion approaches

**Methodology Deep-Dive (3–5 paragraphs):**
The offline gait library is constructed using Hybrid Zero Dynamics (HZD), a formal framework for bipedal locomotion planning. HZD defines gaits as periodic orbits of the hybrid dynamical system (continuous dynamics + discrete impact events at foot strike), parameterized by gait parameters such as step length, step height, walking speed, and body pitch angle. For each point in a discretized gait parameter space, a trajectory optimization problem is solved to find Bézier polynomial coefficients that define dynamically feasible joint trajectories. The resulting library contains hundreds to thousands of gait entries, each with guaranteed stability (all eigenvalues of the Poincaré return map inside the unit circle) and feasibility (respecting actuator torque limits and friction cone constraints).

The RL agent operates at a higher level than the gait library, acting as a navigator through the gait parameter space. Its observation space includes the robot's current state (body pose, velocity, joint positions/velocities), the commanded task (desired velocity, heading), and the current position in gait parameter space. The action space is a continuous vector in gait parameter space — the agent outputs a desired gait parameter vector, which is then mapped to the nearest library entries for interpolation. The RL agent is trained using PPO to maximize a reward that combines velocity tracking, energy efficiency, and smoothness of gait parameter transitions.

Gait interpolation is critical for smooth locomotion, as the discrete library entries would cause jerky transitions if switched abruptly. NaviGait uses a weighted blending scheme: given the RL agent's desired gait parameter vector, the K nearest library entries (typically K=4) are identified, and their joint trajectories are blended using distance-weighted interpolation in gait parameter space. This produces a smooth composite trajectory that inherits approximate dynamic feasibility from its constituent library entries. The blending weights are updated at each gait cycle boundary (foot strike event), ensuring phase-consistent transitions.

The hierarchical structure naturally separates concerns: the offline library handles the complex dynamics optimization (which is computationally expensive but done once), while the online RL agent handles the reactive decision-making (which must be fast but operates in a simpler action space). This separation means the RL agent trains significantly faster than end-to-end approaches because it does not need to discover dynamically feasible gaits from scratch — it only needs to learn which pre-computed gaits are appropriate for the current situation. Training convergence is typically 5–10x faster than end-to-end RL for equivalent locomotion tasks.

Experiments on bipedal robot models (Cassie-like and RABBIT) demonstrate that NaviGait achieves comparable or superior locomotion performance to end-to-end RL while maintaining interpretability (the gait selection can be visualized and understood) and formal guarantees (each library entry is provably stable). The approach is particularly strong in scenarios requiring diverse gaits (walking, running, stair climbing) because the library explicitly covers these modes, whereas end-to-end RL may fail to discover all gait modes during training.

**Key Results & Numbers:**
- 5–10x faster training convergence than end-to-end RL
- Interpretable gait selection with named gait patterns
- Dynamic feasibility maintained through library guarantees
- Smooth transitions via K-nearest neighbor gait interpolation
- Demonstrated on Cassie-like and RABBIT bipedal models
- Library contains 500–2000 gait entries covering diverse locomotion modes
- Real-time execution at >100 Hz for gait selection and interpolation

**Relevance to Project A (Mini Cheetah):** MEDIUM — The gait library concept is relevant for reference-based quadruped locomotion, though Mini Cheetah's end-to-end RL approach may not require a pre-computed library. The HZD framework could provide reference trajectories for reward shaping.
**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to the primitives level (Level 2) of the Cassie hierarchy. The gait library navigation concept maps to how the planner selects among primitive behaviors, and the HZD-based gait entries provide dynamically feasible references for the Neural ODE Gait Phase module.

**What to Borrow / Implement:**
- Use HZD-optimized gait library as reference trajectories for the Cassie primitives level
- Adopt the gait parameter space navigation concept for the planner→primitives interface
- Implement K-nearest neighbor gait interpolation for smooth primitive transitions
- Leverage dynamic feasibility guarantees from the library for safety verification
- Use the gait library to initialize or warm-start the Neural ODE Gait Phase module
- Apply the hierarchical separation (offline optimization + online RL) to Cassie's architecture

**Limitations & Open Questions:**
- Gait library construction requires significant offline computational effort
- Library coverage is finite — novel terrains/conditions outside the library may not be handled
- Interpolation between library entries only approximately preserves dynamic feasibility
- HZD framework assumes known robot dynamics — model errors degrade guarantees
- Scaling to 3D bipedal robots with high DOF increases library dimensionality exponentially
- No mechanism for online library expansion or refinement based on real-world experience
---
