# Learning-Based Legged Locomotion: State of the Art and Future Perspectives

**Authors:** (arXiv 2024 Survey)
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2406.01152

---

## Abstract Summary
This comprehensive survey covers the rapidly evolving field of learning-based legged locomotion, systematically reviewing over 200 papers spanning simulation environments, reward design, domain randomization, teacher-student training, world models, and sim-to-real transfer. The survey provides a unified taxonomy that organizes approaches by their key design decisions: observation space, action space, reward structure, training methodology, and deployment strategy. It covers quadrupeds (ANYmal, Mini Cheetah, Go1, A1), bipeds (Cassie, Digit, Atlas), and hexapods.

The survey identifies several major trends that have converged to produce the current state of the art: (1) GPU-accelerated simulation (Isaac Gym, MuJoCo MJX) enabling billions of environment steps in hours, (2) teacher-student training pipelines where a privileged teacher policy is distilled into a deployable student, (3) domain randomization as the primary sim-to-real transfer mechanism, (4) the shift from hand-crafted rewards to automated curriculum learning, and (5) the emerging role of world models and model-based RL for sample efficiency.

The authors recommend hybrid approaches that combine learning-based adaptability with model-based priors (physics constraints, kinematic feasibility, safety bounds) as the most promising direction. They identify key open challenges including: long-horizon planning for navigation, multi-modal locomotion (walk-run-jump transitions), whole-body manipulation during locomotion, and safety-certified deployment.

## Core Contributions
- **Comprehensive taxonomy:** Organized 200+ papers into a structured taxonomy covering observation design, action representation, reward engineering, training pipelines, and deployment strategies
- **Trend analysis:** Identified and documented the five major trends driving progress: GPU simulation, teacher-student training, domain randomization, automated curricula, and world models
- **Cross-platform comparison:** Provided the first systematic comparison across quadruped, biped, and hexapod platforms with consistent evaluation criteria
- **Sim-to-real analysis:** Detailed analysis of sim-to-real transfer techniques with success/failure patterns across different robot platforms
- **Hybrid approach recommendation:** Made a well-supported case for combining learning with model-based priors as the optimal strategy
- **Open challenges roadmap:** Clearly articulated the unsolved problems and promising research directions
- **Reproducibility assessment:** Evaluated which approaches have been independently reproduced and which remain unvalidated

## Methodology Deep-Dive
The survey structures its analysis around the RL pipeline for legged locomotion, examining each component in depth.

**Observation Space Design:** The survey categorizes observations into proprioceptive (joint angles, velocities, torques, IMU), exteroceptive (depth images, LiDAR, heightmaps), and privileged (ground-truth terrain, exact contact states, body velocity). A key finding is that proprioceptive-only policies transfer best to real hardware but have limited terrain awareness, while exteroceptive policies offer better terrain handling but face significant sim-to-real challenges in perception. The teacher-student paradigm resolves this by training a privileged teacher and distilling to a proprioceptive student that implicitly infers terrain from proprioceptive history.

**Action Space Design:** The survey identifies three dominant action representations: (1) target joint angles (most common, used by >60% of reviewed papers), (2) joint angle deltas (incremental offsets from current position), and (3) joint torques (direct force control). Target joint angles with PD control at the actuator level are recommended as the most stable for sim-to-real transfer because PD control provides a regularizing effect that smooths the sim-to-real gap. The survey notes that torque control achieves higher performance in simulation but transfers poorly due to actuator modeling errors.

**Reward Design:** The survey documents the evolution from hand-crafted multi-term rewards (velocity tracking + energy + smoothness + alive bonus) toward automated reward design through curricula and reward shaping. Key reward terms analyzed include: forward velocity tracking (present in >90% of papers), energy minimization (>70%), contact-based rewards (foot clearance, contact timing, ~40%), and style rewards (reference motion tracking, ~20%). The survey finds that the specific reward weights are often more important than the reward terms themselves, and recommends using curriculum learning to anneal reward weights during training.

**Domain Randomization:** The survey provides a comprehensive catalog of randomized parameters across all reviewed papers. The most commonly randomized parameters are: ground friction (95% of papers), link masses (85%), motor strength (75%), observation noise (70%), external pushes (65%), and terrain type (60%). The survey finds that excessive randomization leads to overly conservative policies, while insufficient randomization leads to sim-to-real failure. Adaptive domain randomization (adjusting randomization ranges based on policy performance) is identified as a promising solution.

**Training Pipelines:** The survey categorizes training approaches into: (1) single-stage PPO (most basic, works for simple locomotion), (2) teacher-student distillation (recommended for deployment), (3) curriculum learning (progressive task difficulty), (4) multi-phase training (separate phases for different skills), and (5) world model approaches (learning a dynamics model for planning). Isaac Gym and MuJoCo MJX are identified as the dominant simulation platforms, with Isaac Gym's GPU parallelism (4096+ environments) enabling training runs of 1-2 billion steps in 2-4 hours.

## Key Results & Numbers
- Papers surveyed: 200+ spanning 2018-2024, with 65% published in 2023-2024 indicating rapid acceleration
- Most common platform: ANYmal (35% of papers), followed by A1/Go1 (25%), Mini Cheetah (15%), Cassie (10%), others (15%)
- Dominant RL algorithm: PPO (used in >85% of papers), followed by SAC (8%), TD3 (4%), other (3%)
- Training scale: Typical training uses 2000-4096 parallel environments, 0.5-2 billion steps, 2-12 hours on 1 GPU
- Sim-to-real success rate: ~70% of papers claiming sim-to-real transfer provide video evidence, ~40% provide quantitative real-world metrics
- Teacher-student gap: Student policies typically achieve 85-95% of teacher performance after distillation
- Domain randomization parameters: Average paper randomizes 5-8 parameters; top-performing papers randomize 12-15
- Reported maximum speeds: Quadrupeds ~5 m/s (Mini Cheetah), bipeds ~3.5 m/s (Cassie), hexapods ~1.5 m/s

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This survey is a critical reference for the Mini Cheetah project, covering virtually every aspect of the RL pipeline. Specific high-value sections include: (1) the observation space analysis helps design the Mini Cheetah's state representation — proprioceptive with optional exteroceptive augmentation; (2) the action space comparison validates target joint angles with PD control as the recommended choice for sim-to-real; (3) the domain randomization catalog provides a comprehensive checklist of parameters to randomize, with recommended ranges; (4) the reward design analysis offers validated reward term combinations specific to quadruped locomotion; (5) the training pipeline comparison supports the teacher-student approach for Mini Cheetah deployment; (6) the MuJoCo-specific analysis (Section on simulation environments) provides best practices for MuJoCo terrain generation, contact modeling, and actuator simulation. The survey's recommendation of curriculum learning over fixed reward weights directly supports the Mini Cheetah project's use of progressive training difficulty.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The survey covers all major components relevant to Project B's Cassie pipeline. Specific connections include: (1) the hierarchical RL section reviews approaches directly applicable to the 4-level hierarchy (Planner→Primitives→Controller→Safety), identifying that 2-3 levels are most common with 4-level hierarchies being rare but promising; (2) the Cassie-specific papers reviewed provide baselines and design patterns for the bipedal platform; (3) the teacher-student training section directly informs the Dual Asymmetric-Context Transformer design, where the privileged teacher corresponds to the "training context" and the deployable student to the "deployment context"; (4) the safety section covers Lyapunov-based approaches (LCBF) and constrained RL, providing context for the Safety level; (5) the domain randomization analysis identifies bipedal-specific parameters that need randomization (ankle compliance, hip width, leg length asymmetry); (6) the sim-to-real section documents Cassie-specific challenges including spring dynamics, gear backlash, and communication delays. The survey's open challenges section identifies multi-modal gait transitions and safety certification as key unsolved problems — both of which Project B directly addresses.

## What to Borrow / Implement
- **Domain randomization checklist:** Use the survey's comprehensive parameter list as a checklist for both Project A (Mini Cheetah) and Project B (Cassie), ensuring no critical randomization parameters are missed
- **Teacher-student pipeline design:** Follow the survey's recommended teacher-student architecture: privileged teacher trained with full state → distillation to proprioceptive student using DAgger or behavior cloning
- **Reward term library:** Adopt the survey's categorized reward terms as a starting library, including the recommended weight ranges for velocity tracking, energy, smoothness, and contact rewards
- **Training hyperparameter guidelines:** Use the survey's aggregated PPO hyperparameter ranges (lr: 1e-4 to 3e-4, clip: 0.1-0.2, epochs: 5-10, minibatch: 4-8) as validated starting points
- **Evaluation protocol:** Adopt the survey's recommended evaluation metrics (velocity tracking RMSE, energy efficiency, disturbance recovery rate, sim-to-real gap) for standardized benchmarking

## Limitations & Open Questions
- The survey has a publication bias toward successful approaches; failed approaches and negative results are underrepresented, potentially giving an overly optimistic view of the field
- Quantitative cross-paper comparisons are limited because papers use different simulation setups, reward functions, and evaluation protocols — the survey acknowledges this but cannot fully resolve it
- The survey coverage ends in early 2024; the rapidly evolving field means some recent developments (e.g., foundation models for robotics, diffusion policies for locomotion) are not covered
- The hierarchical RL section is relatively brief compared to single-level approaches, reflecting the field's current focus on flat policies rather than hierarchical control — this limits the survey's direct utility for Project B's 4-level architecture design
