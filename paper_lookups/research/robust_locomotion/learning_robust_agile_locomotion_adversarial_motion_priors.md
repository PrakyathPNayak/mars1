# Learning Robust and Agile Legged Locomotion Using Adversarial Motion Priors

**Authors:** Various
**Year:** 2023 | **Venue:** IEEE Robotics and Automation Letters (RA-L)
**Links:** https://ieeexplore.ieee.org/document/10167753

---

## Abstract Summary
This work extends AMP to achieve blind (proprioceptive-only) agile locomotion on the Unitree Go1 quadruped. AMP-trained policies show zero-shot generalization to novel terrains despite being trained only on flat-ground motion data. The paper demonstrates that adversarial motion priors encode robustness priors beyond just style — they implicitly regularize the policy toward dynamically stable behaviors that transfer across environments.

## Core Contributions
- Extended AMP to blind (proprioceptive-only) quadruped locomotion, demonstrating that style rewards are sufficient without visual terrain information
- Achieved zero-shot terrain generalization: policies trained on flat ground successfully navigate unseen rough terrain, slopes, and obstacles
- Demonstrated that AMP implicitly encodes robustness and stability priors, not just visual style — the learned gaits are inherently more robust than reward-engineered alternatives
- Showed that AMP-trained policies outperform hand-crafted reward baselines on push recovery and terrain traversal
- Validated on real Unitree Go1 hardware with zero-shot sim-to-real transfer
- Provided evidence that natural gaits are inherently more robust than optimized-but-unnatural gaits, connecting biomechanics insights to RL

## Methodology Deep-Dive
This paper takes the AMP framework from Escontrela et al. (2022) and pushes it toward robust, agile locomotion using only proprioceptive observations. The key insight is that AMP doesn't just make gaits look natural — it makes them inherently more robust. Natural animal gaits evolved under selection pressure for energy efficiency, stability, and robustness to perturbations. By imitating these gaits via adversarial learning, the policy inherits these robustness properties.

The observation space is purely proprioceptive: joint angles, joint velocities, body angular velocity, gravity vector in the body frame, and the previous action. Notably, no terrain information, no heightmap, no vision is provided. The action space is target joint angles for PD controllers. The discriminator takes state transitions (proprioceptive state at t, proprioceptive state at t+1) and classifies them as reference or policy-generated. The reference data consists of motion capture from flat-ground trotting at various speeds.

Training occurs in simulation with domain randomization over terrain roughness, friction, external forces, and robot dynamics (mass, motor strength). However, the reference data remains flat-ground trotting throughout — the discriminator only knows what flat-ground natural locomotion looks like. The remarkable finding is that the policy generalizes to rough terrain zero-shot: the AMP objective forces the policy to maintain natural gait patterns even when the terrain changes, and these natural patterns happen to be robust.

The paper provides extensive ablation studies comparing AMP-trained policies against hand-crafted reward baselines on several metrics: forward velocity tracking, energy consumption, push recovery magnitude, terrain traversal success rate, and sim-to-real transfer success. AMP consistently outperforms or matches the baselines, with the largest improvements on robustness metrics (push recovery, terrain traversal). The authors hypothesize that hand-crafted rewards can be overfit by unnatural but locally optimal behaviors that fail under perturbation, while AMP constrains the policy to the manifold of natural motions that are inherently more robust.

For agile locomotion, the policy is trained to track high-speed velocity commands (up to 3 m/s) and aggressive turning. The AMP objective ensures that even at high speeds, the gait remains natural and stable. This is particularly impressive because high-speed quadruped locomotion is challenging and typically requires careful reward engineering for foot clearance, body stability, and stride frequency.

Real-world deployment on the Unitree Go1 requires no fine-tuning: the sim-trained policy transfers zero-shot. The robot successfully walks on grass, gravel, slopes, and over small obstacles despite never encountering these terrains during training. External push recovery is demonstrated with moderate-force perturbations, with the robot maintaining balance and resuming normal locomotion within 1-2 steps.

## Key Results & Numbers
- Zero-shot terrain generalization: flat-ground-trained policy navigates rough terrain, 15° slopes, gravel, and grass
- Blind locomotion: no vision or heightmap needed, proprioception-only
- Push recovery: withstands 2-3x larger perturbations than hand-crafted reward baselines
- Terrain traversal success rate: 92% vs. 78% for reward-engineered baseline on random rough terrain
- Energy efficiency: 25% lower cost of transport than reward-engineered baseline at matched speed
- Sim-to-real transfer: zero-shot deployment on Unitree Go1 without fine-tuning
- Speed range: robust locomotion from 0.3 m/s to 3.0 m/s
- Reference data: only 30 seconds of flat-ground trotting motion capture required
- Training: ~4 hours on a single GPU with IsaacGym

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Zero-shot terrain generalization is critical for Mini Cheetah deployment in unstructured environments. This paper demonstrates that a blind, AMP-trained policy can handle diverse terrains without any terrain-specific training — exactly the capability needed for a robust baseline locomotion policy. The proprioceptive-only approach serves as an excellent fallback when visual sensing fails or is unavailable. Training only requires flat-ground reference data (easily obtainable from Mini Cheetah demos), and domain randomization handles the sim-to-real gap. The 4-hour training time in IsaacGym is very practical, and the zero-shot sim-to-real transfer eliminates fine-tuning on expensive hardware.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
AMP for bipedal locomotion style transfer is relevant to the low-level controller in Project B's hierarchy. The finding that AMP encodes robustness priors (not just style) is valuable — it suggests that the controller level could use AMP-based training to produce inherently robust bipedal gaits. However, bipedal locomotion is significantly harder than quadruped, and the zero-shot terrain generalization may not transfer directly. The adversarial training principle connects to Project B's adversarial curriculum, and the discriminator architecture could be shared or adapted. The proprioceptive-only approach could serve as a robust fallback mode when Cassie's terrain perception is degraded.

## What to Borrow / Implement
- Implement blind AMP locomotion as a robust baseline policy for Mini Cheetah before adding terrain-aware components
- Use flat-ground reference data only — the zero-shot terrain generalization eliminates the need for diverse terrain demonstrations
- Adopt the proprioceptive observation space: joint angles, velocities, body angular velocity, gravity vector, previous action
- Train with domain randomization (terrain, friction, mass, motor strength) in IsaacGym for practical training times
- Use the push recovery evaluation protocol to benchmark policy robustness across training approaches
- For Project B, adapt the AMP discriminator for bipedal gaits at the controller level of the hierarchy
- Evaluate energy efficiency of AMP vs. current reward-engineered approaches on both platforms

## Limitations & Open Questions
- Zero-shot terrain generalization tested on moderate terrain — extreme obstacles (large gaps, steep stairs) likely require terrain awareness
- Bipedal locomotion may not benefit from the same implicit robustness — bipeds have a smaller stability margin than quadrupeds
- Flat-ground reference data may bias the policy against necessary terrain adaptations (e.g., crouching, high-stepping)
- The discriminator may prevent exploration of novel behaviors needed for extreme agility (parkour, jumping)
- Only trotting gait demonstrated — unclear if AMP produces robust gaits for other patterns (bounding, galloping)
- Push recovery is bounded by the physical limits of the gait pattern — larger perturbations require gait transitions
- No mechanism for the policy to explicitly reason about terrain, limiting performance in highly structured environments
