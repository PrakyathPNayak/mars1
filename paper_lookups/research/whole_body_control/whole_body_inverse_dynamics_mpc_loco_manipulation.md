# Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation

**Authors:** Lukas Molnar et al. (ETH Zurich)
**Year:** 2024 | **Venue:** arXiv 2024
**Links:** https://arxiv.org/abs/2511.19709

---

## Abstract Summary
This paper presents a unified motion and force planning framework that optimizes joint torques via full-order inverse dynamics within MPC for legged robots. It enables simultaneous locomotion and manipulation on quadruped robots (Unitree B2), running at 80 Hz real-time using advanced solvers (Fatrop, Pinocchio, CasADi). The approach achieves sim-to-real transfer for loco-manipulation tasks.

## Core Contributions
- Unifies locomotion and manipulation planning in a single whole-body inverse dynamics MPC formulation
- Optimizes joint torques directly through full-order inverse dynamics rather than simplified models
- Achieves 80 Hz real-time MPC using efficient solver stack (Fatrop for NLP, Pinocchio for dynamics, CasADi for autodiff)
- Demonstrates loco-manipulation tasks including pushing, pulling, and wiping on real hardware (Unitree B2)
- Provides open-source implementation enabling community reproducibility
- Shows that full-order dynamics MPC outperforms reduced-order approaches for contact-rich tasks
- Handles simultaneous locomotion constraints (balance, gait) and manipulation objectives (end-effector tracking, force control)

## Methodology Deep-Dive
The core contribution is formulating the MPC problem using full-order inverse dynamics rather than the commonly used centroidal or single-rigid-body approximations. Full-order inverse dynamics means the optimization directly reasons about individual joint torques, joint accelerations, and contact forces for the entire multi-body system. This provides higher fidelity planning compared to reduced models that abstract away limb dynamics.

The optimization problem is structured as a nonlinear program (NLP) with the following decision variables at each timestep: joint positions, velocities, accelerations, and torques, plus contact forces at each active contact point. The inverse dynamics equations serve as equality constraints, ensuring physical consistency. Task objectives (desired base trajectory, end-effector poses for manipulation, gait timing) are encoded in the cost function.

The solver stack is carefully chosen for real-time performance. Fatrop is a structure-exploiting NLP solver designed for optimal control problems, leveraging the banded structure of the dynamics constraints for efficient factorization. Pinocchio provides highly optimized rigid-body dynamics algorithms (RNEA, ABA, contact Jacobians) that feed into the NLP. CasADi provides automatic differentiation of the cost function and constraints, enabling gradient-based optimization without manual derivative computation.

For loco-manipulation, the framework treats the robot's limbs as dual-purpose: legs for locomotion support and, when not in the stance phase, arms/legs for manipulation. The MPC naturally coordinates these roles by optimizing over the full prediction horizon, finding contact sequences that maintain balance while executing manipulation tasks. Contact transitions (lifting a leg for manipulation, placing it back for support) are handled within the optimization rather than by an external scheduler.

Sim-to-real transfer is achieved through careful system identification of the Unitree B2 platform and robustness to model uncertainty through cost function regularization. The 80 Hz control rate provides sufficient bandwidth for reactive corrections to disturbances and model errors. The open-source release includes the complete solver pipeline, robot models, and task specifications.

## Key Results & Numbers
- 80 Hz real-time MPC execution on standard computing hardware
- Successful loco-manipulation tasks on Unitree B2: pushing objects, pulling doors, wiping surfaces
- Sim-to-real transfer validated on real hardware
- Full-order inverse dynamics outperforms centroidal approximations for manipulation accuracy
- Open-source implementation with solver stack and robot models
- Handles simultaneous locomotion and manipulation without task-specific engineering

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
While Mini Cheetah's primary focus is locomotion rather than manipulation, the inverse dynamics MPC provides a strong analytical control baseline for comparison with RL approaches. The full-order dynamics optimization through MuJoCo-compatible tools (Pinocchio, CasADi) can be set up alongside the RL pipeline. The 80 Hz solve rate demonstrates the feasibility of model-based approaches for real-time quadruped control. The framework could be particularly useful for generating expert demonstrations to bootstrap RL training.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Full-body inverse dynamics MPC is highly relevant to Cassie's low-level Controller design. The direct joint torque optimization aligns with the need for precise torque commands at Cassie's actuators. The combined motion and force optimization is relevant for contact-rich bipedal tasks where the controller must simultaneously track desired trajectories and manage ground reaction forces. The solver stack (Fatrop + Pinocchio + CasADi) could be integrated as the model-based component in a hybrid RL-MPC controller. The loco-manipulation capability is relevant if Cassie is extended to handle objects.

## What to Borrow / Implement
- Set up the Fatrop + Pinocchio + CasADi solver stack for model-based baselines on both platforms
- Use full-order inverse dynamics MPC trajectories as expert demonstrations for RL policy pretraining
- Integrate the inverse dynamics formulation with Cassie's low-level Controller for hybrid RL-MPC control
- Apply the contact scheduling approach for coordinating stance/swing transitions in bipedal walking
- Benchmark RL policies against the MPC baseline on identical locomotion tasks

## Limitations & Open Questions
- 80 Hz may be insufficient for highly dynamic maneuvers requiring faster replanning
- Full-order inverse dynamics is computationally more expensive than centroidal approaches, limiting prediction horizon length
- NLP may converge to local optima, especially for complex contact sequences
- Requires accurate dynamics model (inertias, friction, actuator models) which may be difficult to obtain precisely
- Manipulation tasks demonstrated are relatively simple; complex manipulation with tight tolerances may require higher control rates
