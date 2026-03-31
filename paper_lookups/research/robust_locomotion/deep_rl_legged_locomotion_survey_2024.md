# A Systematic Review of Deep Reinforcement Learning for Legged Robot Locomotion

**Authors:** (MDPI 2024)
**Year:** 2024 | **Venue:** MDPI
**Links:** https://www.mdpi.com/2410-390X/10/1/8/pdf

---

## Abstract Summary
This paper presents a comprehensive systematic review of deep reinforcement learning (DRL) methods applied to legged robot locomotion, covering quadrupeds, bipeds, and humanoids. The survey organizes the rapidly growing literature along multiple axes: RL algorithms (PPO, SAC, TD3, DDPG), simulation platforms (MuJoCo, Isaac Gym, PyBullet, Gazebo), training methodologies (domain randomization, teacher-student, curriculum learning, reward shaping), sim-to-real transfer techniques, and safety considerations. The review encompasses over 150 papers published between 2018 and 2024, providing a structured overview of the field's evolution.

A key contribution is the identification of dominant trends and emerging directions. PPO has become the de facto algorithm for sim-to-real locomotion, largely due to its stability and compatibility with massively parallel simulation (Isaac Gym). Teacher-student distillation is increasingly adopted for bridging privileged simulation information to deployable policies. Curriculum learning has emerged as essential for training on challenging terrains. The survey also highlights the growing importance of safety in RL-based locomotion, including safe exploration, constraint satisfaction, and robustness verification.

The review identifies several open challenges: bridging the persistent sim-to-real gap for dynamic locomotion, scaling to truly diverse and unstructured environments, integrating perception (vision, lidar) with locomotion policies, and ensuring formal safety guarantees. The paper concludes with recommendations for standardized evaluation protocols and reproducibility practices that would accelerate progress in the field.

## Core Contributions
- Provides a comprehensive taxonomy of DRL methods for legged locomotion, organized by algorithm, simulation platform, training methodology, and robot type
- Reviews over 150 papers spanning 2018-2024, covering quadrupeds (ANYmal, Mini Cheetah, A1, Go1, Solo12), bipeds (Cassie, ATLAS, Digit), and humanoids (H1, Atlas)
- Identifies PPO + Isaac Gym + teacher-student + curriculum learning as the dominant paradigm for sim-to-real locomotion
- Provides detailed comparison of RL algorithms (PPO vs. SAC vs. TD3) for locomotion, including sample efficiency, stability, and real-world performance
- Surveys safety in RL-based locomotion: safe exploration, constraint satisfaction, Control Barrier Functions, and robustness verification
- Identifies open challenges: sim-to-real gap, perception integration, formal safety guarantees, and standardized evaluation
- Provides recommendations for reproducibility, benchmarking, and future research directions

## Methodology Deep-Dive
The survey follows the PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) methodology. Papers are collected from IEEE, ACM, Springer, arXiv, and Google Scholar using keywords: "reinforcement learning" AND ("legged locomotion" OR "quadruped" OR "bipedal" OR "humanoid walking"). Inclusion criteria: peer-reviewed or notable preprints, DRL-based control (not classical), real or sim-validated results. The final corpus is 157 papers.

**Algorithm Analysis:** PPO dominates with 68% of surveyed papers using it as the primary algorithm. SAC accounts for 15%, TD3 8%, and other algorithms (DDPG, A2C, MPO) 9%. PPO's dominance is attributed to: (1) compatibility with massively parallel simulation (Isaac Gym supports 4096+ environments), enabling 10-100x speedups; (2) stable training through clipped surrogate objective, reducing sensitivity to hyperparameters; (3) on-policy nature matching the episodic structure of locomotion tasks. SAC is favored for sample-efficient settings (real-world training) and when exploration is critical (diverse gait discovery). TD3 appears primarily in continuous control benchmarks rather than real-robot deployments.

**Simulation Platforms:** Isaac Gym (now Isaac Lab) has emerged as the dominant platform, used in 42% of 2023-2024 papers. Its GPU-accelerated physics enables 10,000+ parallel environments on a single GPU, reducing training time from days (MuJoCo/CPU) to hours. MuJoCo remains popular (35%) for its accuracy and research heritage. PyBullet (12%) serves as an accessible open-source alternative. The survey notes a trend toward differentiable simulators (Brax, DiffTaichi) that enable gradient-based optimization alongside RL.

**Training Methodologies:** The survey identifies four key training techniques. (1) Domain Randomization (used in 78% of sim-to-real papers): randomizes physical parameters (mass, friction, motor properties), observation noise, and terrain. The survey catalogs typical randomization ranges across papers. (2) Teacher-Student Distillation (45% of 2023-2024 papers): a privileged teacher policy (with access to ground-truth terrain, contact forces) is distilled into a deployable student policy using only onboard sensors. This bridges the information gap between simulation and reality. (3) Curriculum Learning (52% of papers): progressively increases terrain difficulty, command complexity, or perturbation magnitude. Automated curriculum (based on agent performance) outperforms hand-designed schedules. (4) Reward Shaping (100% of papers): all surveyed papers use multi-component reward functions; the survey tabulates common reward terms and their typical weights.

**Safety in Locomotion RL:** The survey dedicates a section to safety, covering: (a) Safe Exploration: methods that constrain the policy during training to avoid dangerous states (joint limits, falls), including constrained optimization (Lagrangian methods), Control Barrier Functions (CBFs), and safety critics. (b) Robust Training: techniques ensuring policies remain safe under distribution shift (adversarial training, worst-case domain randomization). (c) Formal Verification: emerging work on verifying neural network policies for safety properties (Lyapunov stability, reachability analysis). The survey notes that formal safety guarantees for DRL locomotion remain an open challenge, with most work relying on empirical robustness testing.

**Bipedal-Specific Challenges:** The survey highlights that bipedal locomotion presents unique challenges compared to quadrupedal: (1) Smaller stability margin requiring more precise balance control; (2) Underactuation in many biped designs (passive ankles, compliant springs); (3) Hybrid dynamics with discrete contact mode switches; (4) Higher sensitivity to model inaccuracies during sim-to-real transfer. Successful bipedal RL papers (Cassie, Digit) typically employ additional techniques: reference motion tracking, phase-based reward design, and more conservative domain randomization.

## Key Results & Numbers
- Algorithm distribution: PPO 68%, SAC 15%, TD3 8%, Other 9% across 157 papers
- Simulation platform trend: Isaac Gym usage grew from 5% (2021) to 42% (2024)
- Sim-to-real success rate: papers report 60-95% of simulation performance transfers to real hardware, with median 78%
- Training time reduction: Isaac Gym reduces training from 24-72 hours (MuJoCo/CPU) to 1-4 hours (single GPU)
- Teacher-student performance: typically achieves 85-95% of teacher performance with deployable observation space
- Domain randomization: reduces sim-to-real velocity tracking error by 30-60% across surveyed papers
- Curriculum learning: enables terrain difficulty scaling from flat ground to 20cm stairs, 30° slopes in single training run
- Safety: <5% of papers provide formal safety guarantees; most rely on empirical testing
- Quadruped success: >90% of quadruped papers demonstrate real-world walking; bipedal success rate is ~65%
- Common reward components: velocity tracking (100% of papers), orientation penalty (95%), energy penalty (85%), smoothness (78%), foot clearance (65%)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This survey is an essential reference covering the entire Mini Cheetah RL locomotion pipeline. It provides empirical evidence for algorithm selection (PPO for parallelized training, SAC for sample-efficient iteration), simulation platform choice (Isaac Gym for speed, MuJoCo for accuracy), and training methodology (domain randomization + teacher-student + curriculum). The reward function comparison across dozens of quadruped papers provides a comprehensive starting point for designing Mini Cheetah's reward. The sim-to-real transfer analysis directly informs the deployment strategy.

The survey's cataloging of domain randomization ranges (mass ±10-30%, friction 0.3-1.5, motor strength ±10-20%) across papers provides validated parameter ranges for Mini Cheetah training. The curriculum learning strategies documented across papers offer tested progressions from flat terrain to challenging environments. This survey should be the first reference consulted for any design decision in the Mini Cheetah RL pipeline.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The survey covers all technologies relevant to Cassie's project: hierarchical RL, bipedal-specific challenges, safety in locomotion, teacher-student distillation, and curriculum learning. The bipedal-specific section identifies the exact challenges Cassie faces (underactuation, narrow stability margin, hybrid dynamics) and catalogs approaches used in successful Cassie papers. The safety section surveys Control Barrier Functions and constrained RL approaches directly relevant to Cassie's Safety level.

The HRL subsection reviews hierarchical approaches for locomotion, including option frameworks, feudal networks, and sub-goal conditioned policies. These map to Cassie's 4-level architecture (Planner, Safety, Primitives, Controller). The teacher-student distillation paradigm could inform the training strategy for Cassie's DACT transformer: train a privileged teacher across all hierarchy levels, then distill to a deployable student. The survey's identification of open challenges (formal safety guarantees, perception integration) directly maps to Cassie's research agenda.

## What to Borrow / Implement
- Use the survey's algorithm comparison to justify PPO (with Isaac Gym) as the primary training algorithm for both projects, with SAC as a secondary option for fine-tuning
- Adopt the documented domain randomization ranges as starting points for both Mini Cheetah and Cassie training
- Implement the teacher-student distillation pipeline identified as best practice for sim-to-real transfer
- Follow the curriculum learning progressions documented across papers for terrain difficulty scaling
- Reference the reward function taxonomy to design comprehensive, multi-component rewards for both robots

## Limitations & Open Questions
- The survey inherently reflects publication bias—negative results and failed approaches are underrepresented, potentially overstating the maturity of DRL locomotion
- Comparison across papers is difficult due to lack of standardized evaluation protocols; reported success rates are not directly comparable
- The survey's coverage of safety is limited to existing work, which the authors acknowledge is insufficient for real-world deployment—formal guarantees remain elusive
- Rapidly evolving field means the survey may miss very recent developments (late 2024 and beyond), particularly in foundation models for locomotion, diffusion policies, and world models
