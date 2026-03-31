# Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions

**Authors:** Alejandro Escontrela, Xue Bin Peng, Wenhao Yu, Tingnan Zhang, Atil Iscen, Ken Goldberg, Pieter Abbeel
**Year:** 2022 | **Venue:** IROS 2022
**Links:** https://arxiv.org/abs/2203.15103

---

## Abstract Summary
AMP (Adversarial Motion Priors) replaces hand-engineered locomotion reward functions with a learned style reward from motion capture data. A GAN discriminator distinguishes between reference motions and policy-generated motions, providing a style reward that encourages natural, energy-efficient gaits. Demonstrated on quadruped robots with successful sim-to-real transfer, eliminating the need for complex reward shaping.

## Core Contributions
- Replaced complex hand-engineered reward functions with a learned adversarial style reward from unstructured motion data
- Used a GAN discriminator to distinguish between reference (motion capture) and policy-generated state transitions, providing a dense style reward signal
- Demonstrated natural gait emergence without explicit gait pattern specification or phase variables
- Achieved energy-efficient locomotion as an emergent property of imitating natural motion data
- Showed successful sim-to-real transfer of AMP-trained policies on quadruped hardware
- Eliminated the need for reward engineering iteration — the same framework works across different locomotion styles by simply swapping reference data
- Proved that unstructured, non-paired motion data is sufficient — no frame-by-frame correspondence between reference and policy is needed

## Methodology Deep-Dive
Traditional locomotion reward functions are complex compositions of many terms: forward velocity, energy penalty, smoothness, foot clearance, body height, orientation, joint limits, contact patterns, and more. Each term requires careful weighting, and the interaction between terms creates unintuitive optimization landscapes. AMP sidesteps this entirely by using a discriminator network trained to distinguish "natural" motion (from reference data) from "unnatural" motion (from the policy).

The discriminator D(s, s') takes a state transition (current state s, next state s') as input and outputs the probability that this transition came from the reference dataset rather than the policy. The policy receives a reward r_style = -log(1 - D(s, s')) for each transition, incentivizing it to produce transitions that the discriminator cannot distinguish from reference motions. This is the standard GAN training objective, applied to state transitions rather than images. The discriminator and policy are trained alternately, with the discriminator seeing mini-batches from both the reference data and the policy's rollout buffer.

The reference data consists of motion capture recordings or kinematic demonstrations of the desired locomotion style. Crucially, this data does not need to be paired with the policy's observations or aligned in time — the discriminator only judges individual state transitions, not entire trajectories. This means reference data can come from different robots, animals, or even hand-crafted animations, as long as the state representation is compatible. For quadruped locomotion, the state includes body orientation, joint angles, joint velocities, and body velocity.

The total reward combines the style reward from the discriminator with a task reward (e.g., forward velocity target): r = w_task * r_task + w_style * r_style. The task reward provides the objective (move forward), while the style reward ensures the policy achieves the objective in a natural manner. The weight balance is important: too much task reward leads to unnatural but effective gaits, while too much style reward leads to beautiful but aimless motion.

Sim-to-real transfer is facilitated by the natural gaits that AMP produces. Because the policy imitates physically realistic motions rather than exploiting simulation artifacts, the resulting gaits are more likely to transfer successfully to real hardware. The energy efficiency emerges naturally — animals evolved energy-efficient gaits, and motion capture data inherits this property. The policy, by imitating these motions, also becomes energy-efficient without explicit energy penalty terms.

## Key Results & Numbers
- Eliminated need for complex reward shaping (replaced 8+ reward terms with a single discriminator)
- Natural gait emergence: trotting, bounding, and pacing gaits learned by imitating reference data
- Energy-efficient locomotion: 30-40% less energy than reward-engineered baselines achieving same speed
- Successful sim-to-real deployment on quadruped hardware (Unitree A1)
- Style reward converges faster than hand-engineered reward (fewer training iterations to natural gaits)
- Works with as few as 10 seconds of reference motion data
- No phase variables or gait clocks needed — gait timing emerges from the adversarial objective
- Discriminator accuracy stabilizes at ~55% (near chance), indicating the policy successfully fools it

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
AMP could replace the manual reward tuning currently required for Mini Cheetah locomotion training. Instead of carefully balancing velocity tracking, energy penalty, smoothness, and contact pattern rewards, a single discriminator trained on motion capture data from the MIT Mini Cheetah (or similar quadrupeds) would provide a holistic style reward. This promotes natural gaits without reward engineering, and the energy efficiency is a valuable bonus for hardware deployment. The successful sim-to-real demonstration on a similar quadruped (Unitree A1) provides strong evidence of transferability. Reference data could come from the Mini Cheetah's existing trotting demonstrations or even from animal locomotion datasets.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The adversarial curriculum in Project B draws from AMP principles. A discriminator-based style reward for natural bipedal gaits would complement the hierarchical architecture: the planner selects locomotion primitives, and AMP ensures each primitive produces natural-looking motion. This is especially valuable for Cassie, where unnatural gaits lead to hardware damage and energy waste. The AMP framework could provide style rewards at the primitives level, using human or Cassie motion capture as reference data. The adversarial training principle also connects to the DIAYN/DADS diversity objectives — both use discriminator-based rewards, suggesting potential architectural unification.

## What to Borrow / Implement
- Implement a GAN discriminator for style reward in both projects, replacing or augmenting hand-engineered reward terms
- Collect or source reference motion data for quadruped (Mini Cheetah) and bipedal (Cassie) locomotion
- Use the state transition (s, s') input format for the discriminator rather than single-state inputs
- Combine style reward with task reward using tunable weights: r = w_task * r_task + w_style * r_style
- Train discriminator on unstructured motion clips (no alignment needed) for ease of data collection
- Evaluate energy efficiency of AMP-trained policies versus current reward-engineered policies
- Consider using AMP at the primitives level in Project B's hierarchy to ensure natural gait for each primitive

## Limitations & Open Questions
- Discriminator training can be unstable (mode collapse, vanishing gradients) — requires careful GAN training practices
- Reference motion data may not be available for all desired locomotion behaviors (jumping, climbing)
- The style reward may conflict with the task reward, leading to policies that look natural but underperform on the task
- Unclear how AMP handles multi-modal locomotion (switching between different gaits) without separate discriminators
- The quality of learned locomotion is upper-bounded by the quality of reference data
- No explicit safety constraints — the discriminator rewards naturalness, not safety
- Discriminator may reward superficial similarity to reference motions rather than deep physical fidelity
