# Real-Time Whole-Body Control of Legged Robots

**Authors:** Various
**Year:** 2024 | **Venue:** arXiv 2024
**Links:** https://arxiv.org/abs/2409.10469

---

## Abstract Summary
This paper presents a real-time whole-body controller combining task-space objectives with joint-level optimization for legged robots. It integrates contact scheduling, momentum optimization, and joint torque computation into a unified QP-based framework running at control frequency. The approach handles multi-contact scenarios and has been tested on real hardware.

## Core Contributions
- Unifies task-space control, contact scheduling, and joint torque optimization into a single QP formulation
- Achieves real-time QP solution at over 200 Hz for whole-body control
- Handles multi-contact scenarios including transitions between different support configurations
- Integrates momentum-based objectives for balance and stability during dynamic motions
- Validates on real legged robot hardware with dynamic locomotion tasks
- Provides a principled framework for incorporating inequality constraints (joint limits, torque limits, friction cones)
- Demonstrates compatibility with both predefined gait schedules and reactive contact planning

## Methodology Deep-Dive
The controller formulates whole-body control as a Quadratic Program (QP) solved at each control timestep. The QP decision variables include joint accelerations, contact forces, and task-space accelerations. The objective function minimizes tracking errors for desired task-space trajectories (body orientation, CoM position, swing foot placement) while regularizing joint torques for energy efficiency.

The QP constraints encode the physics of the system: the equations of motion (Newton-Euler dynamics) appear as equality constraints linking joint accelerations, contact forces, and the robot's inertia matrix. Friction cone constraints ensure contact forces are physically realizable (no sliding, no pulling). Joint position, velocity, and torque limits are encoded as inequality constraints. This formulation guarantees that the computed torques respect all physical and actuator limitations.

Contact scheduling is handled through a combination of predefined gait patterns and reactive adjustments. The gait schedule determines which feet are in contact at each timestep, and the QP optimizes forces only for active contacts. When unexpected contacts occur (e.g., early touchdown), the controller can reactively add contact constraints. When contacts are lost (e.g., slipping), the corresponding force variables are removed from the QP. This flexibility enables robust locomotion across uneven terrain.

Momentum optimization is a critical component for bipedal and dynamic quadrupedal locomotion. The controller tracks desired linear and angular momentum rates, which are derived from higher-level planners or heuristics. For walking, the desired angular momentum is typically zero (upright balance), while for dynamic maneuvers, specific momentum profiles enable agile motions. The QP naturally resolves the redundancy in the system—distributing forces and accelerations across all joints to achieve the desired momentum while satisfying constraints.

The computational efficiency of the QP formulation relies on exploiting the sparse structure of the dynamics equations and using efficient active-set or interior-point solvers. The warm-starting capability (initializing each QP solve from the previous solution) further reduces computation time, enabling consistent real-time performance even for high-dimensional systems.

## Key Results & Numbers
- Real-time QP solve at greater than 200 Hz on standard embedded computing hardware
- Handles multi-contact scenarios including three-point and four-point support transitions
- Tested on real legged robot hardware with dynamic walking and trotting gaits
- Respects joint limits, torque limits, and friction cone constraints at all times
- Compatible with both predefined gait schedules and reactive contact management
- Momentum tracking errors within 5% of desired values during steady-state locomotion

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The QP-based whole-body controller is directly applicable to Mini Cheetah for torque-level command generation. The 200+ Hz solve rate is compatible with the 500 Hz PD control loop (QP runs at 200 Hz, PD interpolates at 500 Hz). The constraint enforcement (joint limits, torque limits, friction cones) provides safety guarantees that pure RL policies lack. This controller could serve as a lower-level safety filter beneath the RL policy, converting desired body-level commands from PPO into safe, physically realizable joint torques.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The QP framework is directly relevant to Project B's LCBF (Learned Control Barrier Function) safety layer, which uses QP for CBF constraint enforcement. The whole-body QP formulation provides the mathematical foundation for integrating CBF constraints alongside tracking objectives. The momentum optimization component aligns with Cassie's need for balance control during bipedal locomotion. The Differentiable Capture Point can be incorporated as an additional constraint or objective within this QP framework. The reactive contact management handles the stance/swing transitions critical for bipedal walking.

## What to Borrow / Implement
- Implement QP-based whole-body control as a safety layer beneath the RL policy for Mini Cheetah
- Extend the QP formulation with CBF constraints for Cassie's LCBF safety layer
- Use the momentum optimization framework for balance control in Cassie's bipedal locomotion
- Integrate the Differentiable Capture Point as a QP constraint for push recovery
- Adopt the reactive contact management for handling unexpected contact events during dynamic gaits

## Limitations & Open Questions
- QP-based approach requires accurate dynamics model (inertia matrix, contact Jacobians) which may have calibration errors
- Hard constraints may lead to infeasible QPs during extreme perturbations; soft constraints or relaxation strategies needed
- Does not learn from experience—fixed optimization structure may be suboptimal compared to learned controllers
- Friction cone approximation (polyhedral) introduces conservatism; exact conic constraints require SOCP solvers
- Interaction between the QP controller and upstream RL policy needs careful interface design to avoid conflicts
