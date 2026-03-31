# Adaptive Energy Regularization for Autonomous Gait Transition and Energetically Economical Quadruped Locomotion

**Authors:** (arXiv 2024)
**Year:** 2024 | **Venue:** arXiv (ICRA 2025)
**Links:** https://arxiv.org/abs/2403.20001

---

## Abstract Summary
This paper introduces a velocity-adaptive energy regularization scheme integrated into the reinforcement learning reward function for quadruped locomotion. The central idea is that energy penalties should not be static but should scale with the commanded velocity—low-speed locomotion demands stricter energy minimization (favoring walking gaits), while high-speed locomotion permits greater energy expenditure (enabling trotting or bounding). This adaptive regularization naturally facilitates energy-optimal gait selection without explicit gait labels or transition logic.

The approach is validated on two real quadruped platforms: ANYmal-C and Unitree Go1. On both platforms, the velocity-adaptive energy regularization yields substantial reductions in Cost of Transport (CoT) compared to fixed-regularization baselines and prior state-of-the-art energy-aware locomotion methods. The policies demonstrate smooth, autonomous gait transitions that emerge purely from the energy optimization landscape.

The method is remarkably simple to implement—requiring only a modification to the reward function's energy penalty coefficient—yet produces significant improvements in locomotion economy. This simplicity makes it highly attractive for integration into existing PPO-based locomotion training pipelines.

## Core Contributions
- A velocity-adaptive energy regularization term where the energy penalty coefficient is a function of the commanded velocity magnitude, providing speed-appropriate energy constraints
- Demonstration that a single scalar function controlling energy regularization is sufficient to induce autonomous gait transitions without gait labels, contact schedules, or state machines
- Real-world deployment on ANYmal-C and Unitree Go1 showing substantial CoT reductions (up to 40% improvement over fixed-regularization baselines)
- Comprehensive ablation study showing the importance of the adaptive schedule versus fixed or linearly-scaled alternatives
- Analysis of emergent gait patterns showing walking at low speeds, trotting at medium speeds, and pronking/bounding at high speeds—mirroring biological energy optimization
- Sim-to-real transfer without additional fine-tuning, demonstrating robustness of the learned energy-efficient behaviors

## Methodology Deep-Dive
The reward function follows the standard form: r = r_task + α(v_cmd) · r_energy, where r_task includes velocity tracking, orientation maintenance, and stability terms, while r_energy penalizes mechanical power consumption (sum of |τ · ω| across all joints). The key innovation is that α(v_cmd) is not a fixed hyperparameter but a function of the commanded velocity magnitude. At low velocities, α is large (strong energy penalty, encouraging economical walking), and at high velocities, α decreases (relaxed energy penalty, allowing the robot to expend energy for speed).

The adaptive schedule α(v_cmd) is parameterized as a piecewise linear function with learned or hand-tuned breakpoints. The authors explore several functional forms: linear decay, exponential decay, and a step function. Empirical results favor the piecewise linear form with two breakpoints, roughly corresponding to walk-trot and trot-gallop transition speeds. The breakpoints are set based on preliminary experiments analyzing CoT curves from fixed-α policies.

The observation space consists of proprioceptive measurements: joint angles (12), joint velocities (12), base angular velocity (3), projected gravity vector (3), and commanded velocity (3). No exteroceptive sensing is used. The action space outputs target joint positions at 50 Hz, which are tracked by PD controllers at 1 kHz on the real robot.

Training uses PPO with generalized advantage estimation (GAE, λ=0.95) in Isaac Gym with 4096 parallel environments. Domain randomization covers friction (μ ∈ [0.2, 1.5]), added payload (0–3 kg for Go1, 0–5 kg for ANYmal-C), motor strength (±10%), PD gain variation (±15%), and observation noise (Gaussian, σ=0.02). Training converges in approximately 2000 iterations (~8 hours on an RTX 3090).

The sim-to-real transfer leverages the domain randomization without additional real-world fine-tuning. On hardware, the policies are evaluated on flat indoor surfaces and gentle outdoor slopes, measuring actual CoT via motor current and velocity measurements.

## Key Results & Numbers
- CoT reduction of 30–40% on ANYmal-C compared to fixed-regularization baselines across the 0.3–1.5 m/s speed range
- CoT reduction of 25–35% on Unitree Go1 across 0.2–1.2 m/s, with the largest gains at intermediate speeds near gait transitions
- Emergent gait transitions: walking below ~0.5 m/s, trotting between 0.5–1.0 m/s, and bounding above 1.0 m/s on Go1
- Velocity tracking RMSE maintained below 0.08 m/s across all speed commands, comparable to energy-agnostic baselines
- Successful zero-shot sim-to-real transfer on both platforms without policy fine-tuning
- Training wall-clock time of ~8 hours on a single GPU with 4096 parallel environments
- Robustness to 2 kg payload variation on Go1 without significant CoT degradation

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is critically relevant to the Mini Cheetah project. The velocity-adaptive energy regularization is directly implementable in the existing PPO training pipeline. Since Mini Cheetah uses MuJoCo simulation with domain randomization, the reward modification requires minimal infrastructure changes—only adding the adaptive α(v_cmd) coefficient to the energy penalty term. The demonstrated 30–40% CoT reduction on real hardware (ANYmal-C, Go1) strongly suggests similar gains are achievable on Mini Cheetah.

The method's simplicity is its greatest strength for Project A: it does not require additional network architectures, auxiliary losses, or gait planners. The piecewise linear schedule for α can be tuned based on Mini Cheetah's specific motor characteristics and target speed range. Given Mini Cheetah's 12-DoF actuated joints and similar scale to Go1, the breakpoint speeds and regularization magnitudes provide a solid starting point for hyperparameter initialization.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The adaptive energy regularization concept transfers well to Cassie's Controller level reward design. While Cassie's hierarchical architecture adds complexity (the energy regularization would primarily live in the low-level PPO controller), the principle of velocity-dependent energy penalties is directly applicable. For bipedal locomotion, the adaptive schedule would encode the walking-to-running transition rather than quadruped-specific gaits.

The reward design philosophy—simple, adaptive coefficients rather than complex reward engineering—aligns well with Cassie's multi-level reward specification challenge. Each level of the hierarchy could employ its own adaptive regularization schedule tuned to its respective action space and objectives.

## What to Borrow / Implement
- Implement the velocity-adaptive energy regularization α(v_cmd) in Mini Cheetah's PPO reward function with piecewise linear schedule
- Use the paper's ablation methodology to tune breakpoint speeds for Mini Cheetah's specific motor and mass characteristics
- Adopt the CoT measurement protocol for real-world Mini Cheetah evaluation using motor current sensing
- Apply the adaptive energy penalty concept to Cassie's low-level controller reward, adapting breakpoints for bipedal walking/running transitions
- Replicate the domain randomization ranges (friction, payload, motor strength, PD gains) as a starting configuration for Mini Cheetah training

## Limitations & Open Questions
- The adaptive schedule α(v_cmd) requires manual tuning of breakpoints; automatic discovery of optimal schedules via meta-learning or Bayesian optimization is not explored
- Evaluation is limited to flat and gently sloped terrain; interaction between energy regularization and rough terrain negotiation (where energy expenditure may be necessary for stability) is not studied
- The method does not account for terrain-adaptive energy budgets—e.g., uphill locomotion may require relaxing energy penalties regardless of speed
- Long-term hardware effects (motor heating, battery depletion profiles) are not modeled in the energy regularization framework
