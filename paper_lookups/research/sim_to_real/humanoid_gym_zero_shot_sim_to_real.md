# Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim-to-Real Transfer

**Authors:** Xinyang Gu, Yen-Jen Wang, Jianyu Chen
**Year:** 2024 | **Venue:** arXiv 2024
**Links:** https://arxiv.org/abs/2404.05695

---

## Abstract Summary
Built on NVIDIA Isaac Gym, Humanoid-Gym provides a complete RL framework for humanoid locomotion with zero-shot sim-to-real transfer. It integrates a sim-to-sim pipeline (Isaac Gym → MuJoCo) for cross-simulator policy verification before real-world deployment. The framework has been validated on RobotEra's XBot-S and XBot-L humanoids across diverse terrains and tasks.

## Core Contributions
- End-to-end RL training framework for humanoid locomotion built on NVIDIA Isaac Gym with massively parallel simulation
- Sim-to-sim verification pipeline transferring policies from Isaac Gym to MuJoCo before real-world deployment, catching simulator-specific artifacts early
- Zero-shot sim-to-real transfer demonstrated on real humanoid hardware (XBot-S and XBot-L) without any fine-tuning
- Open-source release of the full training and deployment codebase, lowering the barrier for humanoid RL research
- Terrain curriculum spanning flat ground, slopes, stairs, and rough terrain for progressive skill acquisition
- Modular reward design enabling rapid iteration on locomotion objectives and task specification

## Methodology Deep-Dive
Humanoid-Gym leverages NVIDIA Isaac Gym's GPU-accelerated parallel simulation to train locomotion policies at scale. Thousands of humanoid instances are simulated simultaneously, each experiencing different terrain configurations and perturbations. The policy network takes proprioceptive observations (joint positions, velocities, body orientation, angular velocity) and outputs target joint positions that are tracked by PD controllers at the actuator level.

The sim-to-sim transfer pipeline is a key innovation. After training in Isaac Gym, policies are deployed in MuJoCo — a physically distinct simulator with different contact models, integration schemes, and dynamics. If a policy performs well in both simulators, it is far more likely to transfer to real hardware. This cross-validation catches overfitting to simulator-specific quirks (e.g., penetration handling, friction cone approximations) that would otherwise manifest as failures on real robots.

Domain randomization is applied to physical parameters including mass, center of mass offsets, friction coefficients, motor strength, and joint damping. This ensures the policy learns behaviors robust to the inevitable parameter mismatches between simulation and reality. The randomization ranges are calibrated against real robot measurements where possible.

The terrain curriculum progressively increases difficulty, starting from flat ground and advancing through slopes, rough terrain, and stairs. An automatic curriculum mechanism promotes agents to harder terrains only when they demonstrate competence at the current level, preventing premature exposure to terrain that would destabilize learning.

Reward shaping balances multiple objectives: forward velocity tracking, energy minimization, body orientation stability, foot clearance, and contact pattern regularity. The reward weights are tuned to produce natural-looking gaits that are also robust, avoiding the pathological behaviors (e.g., excessive crouching, shuffling) that often emerge from naive reward specifications.

## Key Results & Numbers
- Zero-shot transfer to real XBot-S and XBot-L humanoids without any real-world fine-tuning
- Successful locomotion across flat ground, slopes, stairs, and rough terrain in both simulation and reality
- Cross-simulator verification (Isaac Gym → MuJoCo) catches policy failures before hardware deployment
- Open-source framework enables reproducible results and community adoption
- Training converges within hours on a single GPU using Isaac Gym's parallel simulation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The Humanoid-Gym framework and training pipeline are transferable to quadruped settings. The Isaac Gym parallel training infrastructure, terrain curriculum design, and domain randomization strategies can be directly adapted for Mini Cheetah. The sim-to-sim verification concept (Isaac Gym → MuJoCo) is particularly valuable since Project A uses MuJoCo as its primary simulator — this paper's pipeline could validate policies trained in alternative simulators before deployment. The PD control architecture at the actuator level mirrors Project A's 500 Hz PD control scheme. However, the humanoid-specific aspects (bipedal balance, arm coordination) are less directly applicable.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Humanoid zero-shot sim-to-real transfer is directly relevant to Cassie deployment challenges. The sim-to-sim verification pipeline could be adapted to validate Cassie policies across simulators before real-world transfer, reducing the risk of catastrophic failures. The Isaac Gym training infrastructure supports the large-scale parallel training needed for Project B's multi-level hierarchy. The terrain curriculum approach aligns with Project B's adversarial curriculum design, though Humanoid-Gym uses a simpler progression mechanism. The domain randomization strategy for humanoid dynamics (mass, CoM, friction, motor parameters) maps well to Cassie's parameter uncertainty, especially around its series elastic actuators. The open-source nature provides a reference implementation for humanoid RL training that could accelerate Project B's development.

## What to Borrow / Implement
- Adopt the sim-to-sim verification pipeline (Isaac Gym → MuJoCo) as a pre-deployment validation step for both projects
- Use the terrain curriculum progression mechanism as a baseline for Project B's adversarial curriculum
- Leverage the open-source codebase as a reference for Isaac Gym humanoid training setup
- Apply the domain randomization parameter ranges and calibration methodology to Cassie's actuator models
- Consider the modular reward design pattern for structuring multi-objective rewards in Project A

## Limitations & Open Questions
- Zero-shot transfer demonstrated only on RobotEra's specific humanoids; generalization to other platforms (like Cassie) is unverified
- Sim-to-sim pipeline adds engineering complexity and requires maintaining two simulator environments
- Limited to proprioceptive observations; no vision or exteroceptive sensing integration
- Terrain curriculum is relatively simple compared to adversarial or procedural generation approaches
- No explicit safety constraints or recovery behaviors in the framework
- How well does the sim-to-sim verification correlate with actual sim-to-real success rates across diverse robot morphologies?
