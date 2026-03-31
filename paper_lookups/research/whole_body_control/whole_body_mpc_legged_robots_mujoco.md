# Whole-Body Model-Predictive Control of Legged Robots with MuJoCo

**Authors:** Various (CMU / Google DeepMind)
**Year:** 2025 | **Venue:** arXiv 2025
**Links:** https://arxiv.org/abs/2503.04613

---

## Abstract Summary
This paper demonstrates real-time whole-body MPC for legged robots using MuJoCo's analytical derivatives and iterative Linear Quadratic Regulator (iLQR). It achieves hardware-capable control rates on both quadruped and humanoid platforms with successful sim-to-real transfer. The open-source implementation provides community baselines for benchmarking model-based control approaches.

## Core Contributions
- Achieves real-time whole-body MPC at control frequency using MuJoCo's analytical contact derivatives
- Leverages iLQR with MuJoCo's efficient dynamics computation for trajectory optimization
- Demonstrates successful sim-to-real transfer for both quadruped and humanoid platforms
- Provides open-source baselines for community benchmarking of model-based legged locomotion
- Shows that model-based control can match RL-based approaches in robustness while providing interpretability
- Unifies the simulation environment and the control model, eliminating model mismatch
- Handles contact-rich dynamics including multi-legged support and transitions

## Methodology Deep-Dive
The key innovation is using MuJoCo not just as a simulation environment but as the dynamics model within the MPC optimization loop. MuJoCo provides analytical derivatives of the contact-inclusive dynamics, enabling efficient gradient computation for trajectory optimization. This eliminates the need for simplified dynamics models (e.g., single rigid body or centroidal dynamics) that introduce model mismatch.

The MPC formulation uses iterative Linear Quadratic Regulator (iLQR) as the optimization backbone. At each control step, the controller solves a finite-horizon optimal control problem: given the current state, find the sequence of joint torques that minimizes a cost function over a prediction horizon while respecting the full rigid-body dynamics with contacts. MuJoCo's analytical Jacobians and Hessians of the dynamics enable efficient iLQR iterations within the real-time budget.

The cost function encodes task objectives (desired velocity, body orientation, foot placement targets) alongside regularization terms (joint torque penalties, smoothness). The contact-inclusive dynamics model means the optimizer naturally discovers appropriate contact sequences and force distributions without requiring a separate contact planner or predefined gait schedule. This is a significant advantage over approaches that decouple contact planning from motion optimization.

Sim-to-real transfer is facilitated by the fact that the same MuJoCo model used for MPC online is also used during offline parameter identification. Physical parameters (masses, inertias, friction coefficients, actuator models) are calibrated to match the real hardware, and domain randomization over remaining uncertainties is applied during offline validation. The unified model reduces the sim-to-real gap compared to approaches using different models for simulation and control.

The open-source release includes the full MPC pipeline, robot models, and deployment infrastructure for community use. This enables direct comparison between model-based (MPC) and learning-based (RL) approaches on identical hardware and tasks.

## Key Results & Numbers
- Real-time MPC execution at the control frequency of the robot (200+ Hz)
- Successful sim-to-real transfer on both quadruped and humanoid platforms
- Open-source implementation with robot models and deployment code
- Handles multi-contact scenarios and contact transitions without separate gait planning
- Competitive with RL-based approaches in tracking and disturbance rejection
- Uses MuJoCo's analytical derivatives for efficient iLQR optimization

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to the Mini Cheetah MuJoCo simulation pipeline. The whole-body MPC provides a strong model-based baseline to compare against the PPO-trained RL policy. Using MuJoCo as both the simulation environment and the MPC dynamics model eliminates model mismatch—a key advantage for the Mini Cheetah setup. The open-source implementation can be adapted to Mini Cheetah's 12 DoF model for direct comparison. The MPC approach also provides interpretable behaviors that can inform reward shaping for RL training.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Whole-body MPC offers an alternative or complement to the RL-based low-level Controller in Cassie's hierarchy. The MPC could serve as the Controller level, receiving desired trajectory commands from the Primitives level and computing joint torques. The QP-based nature of iLQR aligns with the LCBF safety layer's QP formulation, potentially enabling joint optimization of tracking and safety constraints. For humanoid platforms in particular, the demonstrated sim-to-real validates the MuJoCo-based approach for bipedal control.

## What to Borrow / Implement
- Set up the MuJoCo-based whole-body MPC as a baseline controller for Mini Cheetah to benchmark against RL policies
- Use MPC trajectories as expert demonstrations for initializing PPO training (imitation + RL)
- Integrate MPC as an optional low-level controller in Cassie's hierarchy for comparison studies
- Leverage MuJoCo's analytical derivatives for model-based components in the hybrid RL-MPC pipeline
- Use the open-source codebase to establish reproducible benchmarks for both projects

## Limitations & Open Questions
- Real-time MPC is computationally expensive, potentially challenging on embedded hardware with limited compute
- iLQR may get stuck in local optima, especially for contact-rich scenarios with multiple possible contact sequences
- Requires accurate dynamics model calibration; performance degrades with unmodeled dynamics or parameter errors
- Prediction horizon length trades off between planning quality and computational cost
- Does not naturally incorporate learning from experience—performance is bounded by the model accuracy
