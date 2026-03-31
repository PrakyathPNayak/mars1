# HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robots via Wasserstein Adversarial Imitation

**Authors:** Various
**Year:** 2024 | **Venue:** ICRA 2024
**Links:** https://arxiv.org/abs/2309.14225

---

## Abstract Summary
HumanMimic applies Wasserstein Adversarial Imitation Learning (WAIL) to humanoid robots, enabling them to naturally mimic human locomotion and smoothly transition between gait styles. The approach resolves human-to-robot morphological differences using a primitive-skeleton motion retargeting pipeline combined with adversarial and reinforcement learning. The result is naturalistic, robust humanoid behaviors validated on real hardware.

## Core Contributions
- Introduces Wasserstein Adversarial Imitation Learning (WAIL) for humanoid locomotion
- Develops primitive-skeleton motion retargeting to bridge human-robot morphological gaps
- Achieves smooth, natural transitions between walking, running, and turning gaits
- Demonstrates training stability improvements over standard GAIL via Wasserstein distance
- Validates on real humanoid hardware with natural-looking locomotion
- Combines adversarial imitation with task-specific RL rewards for goal-directed behavior
- Shows robustness to perturbations and varied terrain conditions

## Methodology Deep-Dive
The primitive-skeleton retargeting pipeline is the first stage. Human motion capture data is first mapped to a simplified "primitive skeleton" that captures essential kinematic features (pelvis height, foot placement, limb orientations) while abstracting away morphological details. This primitive skeleton serves as an intermediate representation between human and robot kinematics, making the retargeting more robust than direct joint-angle mapping. The robot-specific retargeting then maps from primitive skeleton to the target robot's joint space using inverse kinematics with joint-limit constraints.

The Wasserstein Adversarial Imitation Learning (WAIL) framework replaces the standard Jensen-Shannon divergence used in GAIL with the Wasserstein-1 distance (Earth Mover's Distance). This provides several advantages: smoother gradients even when the discriminator is well-trained, meaningful distance metrics for measuring imitation quality, and more stable training dynamics. The discriminator (critic) is trained with a gradient penalty to enforce the Lipschitz constraint required by the Wasserstein formulation.

The policy receives observations including proprioception (joint angles, velocities, IMU), a phase variable encoding gait timing, and a command signal specifying desired velocity and gait type. The adversarial reward from the Wasserstein critic is combined with task rewards (velocity tracking, stability) via a weighted sum. This allows the policy to simultaneously mimic natural human gaits and achieve specific locomotion goals.

Gait transitions are handled by conditioning the policy on a gait type indicator and the phase variable. During training, the reference motion dataset includes transitions between gaits (walk-to-run, forward-to-turn), and the adversarial objective encourages the policy to reproduce these transition dynamics. The phase variable provides temporal grounding, preventing the policy from producing temporally inconsistent motions during transitions.

Real-hardware deployment uses a standard sim-to-real pipeline with domain randomization over dynamics parameters. The Wasserstein adversarial training in simulation produces policies that transfer well because the naturalistic gait priors constrain the policy to physically plausible behaviors, reducing the space of sim-to-real discrepancies.

## Key Results & Numbers
- Natural human-like walking and running gaits achieved on real humanoid hardware
- Smooth gait transitions (walk↔run, forward↔turning) with <0.5s transition time
- Wasserstein training shows 30% fewer training failures compared to standard GAIL
- Velocity tracking error <0.15 m/s for walking, <0.25 m/s for running
- Robust to external pushes up to 50N on hardware
- Motion naturalness rated higher by human evaluators compared to pure RL baselines
- Training converges in ~50M environment steps in simulation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The Wasserstein adversarial imitation framework could replace or augment standard PPO reward shaping for the Mini Cheetah. While the humanoid focus doesn't directly apply to quadruped morphology, the primitive-skeleton retargeting concept could be adapted for animal-to-robot retargeting. The training stability benefits of Wasserstein over standard GAIL are universally applicable. The gait transition methodology is relevant for smooth velocity-dependent gait switching (walk→trot→gallop) in the Mini Cheetah's curriculum learning pipeline.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This paper is directly applicable to Cassie's bipedal locomotion learning. The Wasserstein adversarial training can serve as the adversarial curriculum component, providing natural gait priors from human motion data. The primitive-skeleton retargeting handles the Cassie-to-human morphological differences (Cassie's unique leg kinematics). The gait transition framework maps to the Primitives level, where smooth switching between walking, turning, and stair climbing is needed. The phase variable aligns with the Neural ODE Gait Phase component. The combined adversarial + task reward approach is directly implementable in the PPO training pipeline.

## What to Borrow / Implement
- Implement Wasserstein adversarial reward as the adversarial curriculum component for Cassie
- Adopt primitive-skeleton retargeting for human-to-Cassie motion mapping
- Use the phase variable approach to augment the Neural ODE Gait Phase module
- Apply gait transition training methodology at the Primitives level
- Combine WAIL reward with task rewards using the weighted-sum approach in PPO training
- Leverage the training stability benefits of Wasserstein distance over GAIL for both projects

## Limitations & Open Questions
- Wasserstein distance computation adds overhead; may affect 500 Hz control loop timing
- Primitive-skeleton abstraction may lose fine-grained motion details important for dynamic locomotion
- Gait transition dataset must include transition examples—not all transitions may be available
- Limited evaluation on highly uneven terrain or stairs
- Open question: How does WAIL interact with domain randomization for sim-to-real transfer?
- Scaling to more diverse gait types (lateral walking, backward) not demonstrated
