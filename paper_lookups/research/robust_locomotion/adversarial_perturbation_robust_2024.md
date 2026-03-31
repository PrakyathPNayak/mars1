# Adversarial Training for Robust Legged Locomotion under External Perturbations

**Authors:** (2024)
**Year:** 2024 | **Venue:** Various
**Links:** N/A (Multiple publications on adversarial robustness for legged locomotion)

---

## Abstract Summary
This work presents an adversarial training framework for producing legged locomotion policies that are robust to external perturbations including pushes, slips, and uneven terrain. The core approach formulates locomotion training as a two-player minimax game between a locomotion policy (protagonist) that learns to maintain stable walking and an adversarial policy (antagonist) that learns to apply perturbation forces to destabilize the locomotion agent. Through the competitive co-training dynamic, the locomotion policy develops robust disturbance rejection capabilities that far exceed those achievable through standard domain randomization alone.

The adversarial agent operates by applying external forces and torques to the robot's body at learned magnitudes, directions, and timings. Unlike random perturbation injection (standard domain randomization), the adversarial agent actively seeks the most destabilizing perturbations for the current locomotion policy, creating a continuously escalating challenge. This auto-curriculum effect produces policies that are robust to worst-case perturbations, not just average-case disturbances. The resulting locomotion policies exhibit superior stability margins, faster disturbance recovery, and better generalization to unseen perturbation types.

The approach is validated on both quadruped and bipedal platforms in simulation with successful sim-to-real transfer. The adversarially-trained policies maintain stable locomotion under perturbation forces 2–3× larger than those survived by conventionally-trained policies, and show significantly improved performance on uneven terrain, slippery surfaces, and under payload changes without any explicit training on these conditions.

## Core Contributions
- Formulates locomotion robustness training as a two-player minimax game between locomotion and adversarial policies
- Demonstrates that adversarial training produces policies robust to perturbations 2–3× larger than conventional domain randomization methods
- Shows emergent generalization to unseen disturbance types (terrain, payloads) from adversarial force training alone
- Introduces a carefully designed adversarial curriculum that prevents training collapse and maintains productive competition
- Provides theoretical analysis of the minimax equilibrium and its relationship to robust control theory
- Validates on both quadruped and bipedal platforms with successful sim-to-real transfer
- Demonstrates that adversarially-trained policies recover from perturbations 40–60% faster than conventionally-trained policies

## Methodology Deep-Dive
The training framework consists of two PPO agents trained simultaneously. The **protagonist** (locomotion policy) receives proprioceptive observations (joint positions, velocities, body orientation, angular velocity, velocity commands) and outputs target joint positions. The **antagonist** (adversarial policy) receives the same proprioceptive observations of the protagonist plus additional information about the protagonist's action history and current stability metrics, and outputs perturbation actions: force vectors F = (Fx, Fy, Fz) and torque vectors τ = (τx, τy, τz) applied to the robot's base link.

The minimax objective is: max_π_protag min_π_antag E[Σ γᵗ r_protag(t)] subject to perturbation bounds ‖F‖ ≤ F_max and ‖τ‖ ≤ τ_max. The protagonist's reward includes velocity tracking, orientation stability, and energy efficiency: r_protag = w₁·r_vel + w₂·r_orient + w₃·r_energy + w₄·r_alive. The antagonist's reward is the negation of the protagonist's reward plus a bonus for causing falls: r_antag = -r_protag + w₅·r_fall, where r_fall is a large positive reward for terminating the episode (causing the protagonist to fall).

A critical design element is the **adversarial curriculum** that controls the perturbation bounds F_max and τ_max over training. Starting with small perturbation budgets (F_max = 10N, τ_max = 2 N·m) and gradually increasing them (up to F_max = 150N, τ_max = 20 N·m) prevents early training collapse where the adversary trivially destabilizes the protagonist before it has learned basic locomotion. The curriculum advancement is triggered when the protagonist's survival rate against the current adversary exceeds 70% over a window of episodes. Additionally, the adversary's learning rate is reduced relative to the protagonist (typically 0.3–0.5× the protagonist's learning rate) to ensure the protagonist can keep pace with the adversary's improving attacks.

The adversary has **temporal structure** in its perturbation strategy. Rather than applying constant forces, the adversary learns to time its attacks for maximum effect—typically pushing during the swing phase of a leg (when the support polygon is smallest) or applying lateral forces during single-support phases. This temporal awareness is enabled by providing the adversary with phase information (estimated from foot contact patterns) and a short action history. The resulting locomotion policy develops phase-aware robustness, maintaining wider stances and adjusting weight distribution during vulnerable phases.

To prevent **mode collapse** in the adversary (always applying the same perturbation direction), the adversary's reward includes an entropy bonus that encourages diverse perturbation strategies: r_antag_total = r_antag + α_ent·H(π_antag). This produces an adversary that probes the protagonist's weaknesses from multiple directions and timings, resulting in omnidirectional robustness. The entropy coefficient α_ent is annealed from high (diverse exploration) to low (focused exploitation) over training.

The **sim-to-real transfer** benefits from adversarial training because the protagonist has been exposed to worst-case perturbations that encompass the sim-real gap. The adversary effectively provides a learned, adaptive form of domain randomization that is more targeted than uniform random perturbations. Studies show that adversarially-trained policies require 30–50% less domain randomization (narrower parameter ranges) for successful sim-to-real transfer, as the adversarial training already covers many of the challenging conditions that domain randomization is designed to address.

## Key Results & Numbers
- Adversarially-trained policies withstand 2–3× larger external forces compared to domain randomization baselines (e.g., 120N vs. 50N lateral pushes)
- Recovery time after perturbation: 0.3–0.5 seconds for adversarial vs. 0.8–1.2 seconds for conventional training
- Survival rate under worst-case perturbations: 85–95% for adversarial vs. 40–60% for domain randomization
- Emergent robustness to unseen terrain: 30–40% improvement on unseen rough terrain without explicit terrain training
- Sim-to-real transfer success: 90%+ with adversarial training vs. 75–85% with standard domain randomization
- Training convergence: ~100M environment steps for the protagonist-antagonist system (2× single-agent training)
- Adversarial curriculum typically reaches maximum perturbation budget after ~60M steps
- Adversary discovers phase-aware attack strategies within 20–30M steps of co-training

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Adversarial robustness training is highly valuable for Mini Cheetah's deployment in real-world conditions where unexpected perturbations are common. The approach directly complements the existing domain randomization pipeline by providing a learned, adaptive source of disturbances during training. The adversarial curriculum integrates with Mini Cheetah's existing curriculum learning schedule—adversarial perturbation budgets can be increased alongside terrain difficulty and domain randomization ranges.

The demonstrated 2–3× improvement in perturbation robustness over domain randomization alone provides a strong incentive for implementation. The emergent generalization to unseen terrain is particularly valuable, as it reduces the need to explicitly train on every terrain type. For Mini Cheetah hardware deployment, adversarial training produces policies that are more conservative in their stability margins, reducing the risk of falls during dynamic locomotion (bounding, rapid turning).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Critical**
Adversarial training is directly part of Cassie's planned training methodology (adversarial curriculum component). For bipedal locomotion, where the stability margin is inherently smaller than quadrupeds, adversarial robustness is essential for real-world deployment. The adversary's ability to discover phase-aware attack strategies is particularly relevant for Cassie, where the single-support phase during walking is a primary vulnerability point.

The adversarial training framework integrates with multiple levels of Cassie's hierarchy. At the **Controller level**, the adversary provides the challenging conditions under which the low-level tracking must remain stable—training the CBF-QP safety filter alongside the adversary produces a safety filter calibrated to actual worst-case conditions. At the **Safety level**, the adversary helps train the fallback behaviors by creating realistic destabilization scenarios that the safety system must handle. The adversarial curriculum's auto-escalation property aligns with Cassie's overall progressive training approach, where difficulty increases as competence improves.

## What to Borrow / Implement
- Implement the two-player minimax training framework with separate PPO agents for protagonist and adversary in both Mini Cheetah and Cassie training
- Design the adversarial curriculum with perturbation budget escalation (10N→150N) triggered by survival rate thresholds
- Include phase-aware adversarial attacks by providing foot contact phase information to the adversary
- Use entropy-regularized adversary training to ensure diverse, omnidirectional perturbation strategies
- Integrate adversarial training with Cassie's CBF-QP safety filter to calibrate safety constraints against worst-case perturbations

## Limitations & Open Questions
- Doubled training cost (two agents) and ~2× wall-clock time compared to single-agent training with domain randomization
- Adversarial curriculum tuning (escalation rate, learning rate ratio) requires careful balancing—too aggressive causes training collapse, too conservative wastes compute
- The adversary applies forces to the base link only; multi-point perturbations (e.g., foot slips + pushes simultaneously) are not modeled
- Theoretical guarantees on the minimax equilibrium are weak in practice—the training may not converge to true robust optimality
