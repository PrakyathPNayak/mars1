# Maximizing Quadruped Velocity by Minimizing Energy (EIPO)

**Authors:** Srinath et al.
**Year:** 2024 | **Venue:** ICRA
**Links:** https://srinathm1359.github.io/eipo-locomotion/

---

## Abstract Summary
This paper introduces Extrinsic-Intrinsic Policy Optimization (EIPO), a constrained reinforcement learning framework designed to simultaneously maximize locomotion task performance (velocity tracking, stability) while minimizing energy consumption. Unlike standard RL approaches that combine task and energy objectives into a single weighted reward, EIPO treats energy minimization as a hard constraint, guaranteeing that the policy satisfies an energy budget while optimizing task performance. This formulation avoids the brittle reward-weight tuning that plagues multi-objective locomotion RL.

The key insight is separating the optimization into an extrinsic objective (task performance) and an intrinsic objective (energy efficiency), then solving the resulting constrained optimization problem using a Lagrangian relaxation approach integrated with PPO. The Lagrange multiplier is automatically adjusted during training, dynamically balancing task performance against energy cost without manual hyperparameter sweeps.

EIPO is validated both in simulation and on real quadruped hardware, demonstrating that it achieves near-optimal velocity tracking while consuming significantly less energy than unconstrained baselines. At high speeds, where the energy-speed trade-off is most pronounced, EIPO finds locomotion strategies that are both fast and efficient—discovering gaits that naturally minimize unnecessary motion and ground impact forces.

## Core Contributions
- Formulation of energy-efficient locomotion as a constrained RL problem with task performance as the objective and energy as a constraint, avoiding reward weight tuning
- EIPO algorithm integrating Lagrangian relaxation with PPO, featuring automatic Lagrange multiplier adaptation for the energy constraint
- Theoretical analysis showing EIPO converges to a locally optimal policy on the Pareto frontier of the speed-energy trade-off
- Demonstration that constrained formulation discovers qualitatively different (and more efficient) gaits compared to reward-weighted approaches
- Real-world validation showing energy savings of 20–30% at high speeds without sacrificing velocity tracking accuracy
- Comprehensive comparison against reward-weighted baselines, multi-objective RL, and prior constrained RL methods

## Methodology Deep-Dive
EIPO formulates the locomotion problem as a Constrained Markov Decision Process (CMDP): maximize E[Σ r_task(s,a)] subject to E[Σ c_energy(s,a)] ≤ ε, where r_task captures velocity tracking, orientation stability, and smoothness, while c_energy measures instantaneous mechanical power |τ · ω| summed across joints. The constraint threshold ε is a single interpretable hyperparameter representing the maximum allowable average energy consumption.

The constrained problem is converted to an unconstrained one via the Lagrangian: L(π, λ) = E[Σ r_task] - λ · (E[Σ c_energy] - ε). The policy π and Lagrange multiplier λ are updated alternately: π is updated via PPO to maximize L, and λ is updated via gradient ascent on the constraint violation. This min-max optimization ensures that if the policy violates the energy constraint, λ increases (making energy more expensive), and if the policy is well within budget, λ decreases (allowing the policy to focus on task performance).

The extrinsic reward r_task comprises: velocity tracking (exponential kernel on velocity error), body height maintenance, orientation penalty (roll and pitch), action smoothness (penalizing action differences between timesteps), and foot slip penalty. The intrinsic energy cost c_energy is the sum of absolute mechanical power across all 12 joints: Σ|τ_i · ω_i|.

A critical implementation detail is the use of a separate value network for the energy cost function. While the task value function V_task(s) estimates future task rewards, a second network V_energy(s) estimates future energy costs. Both are trained with GAE (λ=0.95) but with independent advantage estimates. This dual-critic architecture prevents interference between the task and energy estimation, improving training stability.

Training is conducted in Isaac Gym with 4096 parallel environments. The policy network is a 3-layer MLP (512×256×128) with ELU activations. The Lagrange multiplier λ is initialized to 1.0 and updated with learning rate 5×10⁻³. Key domain randomization includes friction (0.3–1.5), payload (±2 kg), motor strength (±10%), and terrain height noise (±2 cm). Training converges in approximately 4000 iterations.

## Key Results & Numbers
- 20–30% energy reduction at speeds above 1.0 m/s compared to reward-weighted PPO baselines on real hardware
- Velocity tracking RMSE < 0.05 m/s across 0–2.0 m/s command range, matching unconstrained baseline accuracy
- EIPO discovers distinct gait strategies: shorter stride lengths with higher frequency at medium speeds, reducing ground reaction forces
- Pareto-optimal solutions: EIPO policies consistently lie on or near the Pareto frontier of the speed-energy trade-off
- The automatic Lagrange multiplier adaptation eliminates the need for reward weight sweeps, reducing hyperparameter tuning from ~50 runs to a single energy threshold selection
- Sim-to-real transfer success rate >95% across tested speed commands
- Training time approximately 10 hours on a single NVIDIA A100 GPU

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
EIPO provides an elegant solution to the speed-efficiency trade-off that is central to Mini Cheetah's locomotion objectives. The constrained optimization formulation is directly applicable—rather than manually tuning the weight between velocity tracking and energy penalty in Mini Cheetah's PPO reward, EIPO's Lagrangian approach automatically finds the optimal balance. This is especially valuable for Mini Cheetah's multi-speed operation, where the appropriate energy budget varies with commanded velocity.

The dual-critic architecture (separate value networks for task and energy) can be integrated into Mini Cheetah's existing PPO implementation with moderate effort. The energy constraint threshold ε provides a single, interpretable knob for tuning the speed-efficiency trade-off, replacing multiple interacting reward weights. Given Mini Cheetah's real-world deployment goals, the demonstrated 20–30% energy savings at high speeds translate directly to extended battery life and reduced actuator wear.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The EIPO framework concepts are transferable to Cassie's energy-constrained locomotion, particularly at the Controller level where PPO optimizes joint-level actions. The constrained optimization formulation could replace reward-weighted energy terms in the Controller's reward function, providing more principled energy management. However, integration with Cassie's 4-level hierarchy adds complexity—the energy constraint may need to be decomposed across hierarchy levels.

The dual-critic architecture concept could extend to Cassie's hierarchical setup, with energy critics at each level of the hierarchy estimating energy budgets for their respective action spaces.

## What to Borrow / Implement
- Implement the CMDP formulation with Lagrangian relaxation for Mini Cheetah's PPO training, replacing manual reward weight tuning for energy terms
- Adopt the dual-critic architecture with separate value networks for task performance and energy estimation
- Use the automatic Lagrange multiplier adaptation to dynamically balance speed and energy objectives during training
- Apply the energy constraint threshold as a deployment-time parameter for Mini Cheetah, allowing operators to trade speed for battery life
- Borrow the Pareto analysis methodology to evaluate Mini Cheetah policies along the speed-energy frontier

## Limitations & Open Questions
- The energy constraint threshold ε is fixed during training; adaptive thresholds that change with terrain difficulty or battery state are not explored
- EIPO's Lagrangian optimization can exhibit oscillatory behavior in λ during early training, requiring careful learning rate tuning for the multiplier update
- The method assumes energy cost is well-captured by mechanical power; thermal losses, actuator inefficiencies, and computational energy are not modeled
- Extension to hierarchical architectures (relevant for Cassie) where energy constraints must be distributed across levels is not addressed
