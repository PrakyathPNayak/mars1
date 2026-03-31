# Multimodal Bipedal Locomotion Generation with Passive Dynamics via Deep Reinforcement Learning

**Authors:** Koseki et al.
**Year:** 2023 | **Venue:** Frontiers in Neurorobotics
**Links:** https://www.frontiersin.org/articles/10.3389/fnbot.2022.1054239

---

## Abstract Summary
This paper leverages the robot's passive dynamics and morphology to realize human-like gait transitions (walking, running, skipping) with minimal command changes using deep RL. By designing rewards that exploit natural dynamics rather than fight them, the policy discovers energy-efficient and naturalistic bipedal locomotion modes. The approach shows that a single policy can produce multiple gait types by varying speed commands, with smooth transitions emerging naturally.

## Core Contributions
- Single RL policy that produces walking, running, and skipping gaits by exploiting passive dynamics
- Human-like gait transitions that emerge naturally from speed command changes rather than explicit mode switching
- Energy-efficient locomotion achieved by leveraging the robot's natural dynamics and morphological properties
- Reward design philosophy that encourages exploitation of passive dynamics rather than full active control
- Analysis of how compliant actuators and link inertia properties create natural resonant frequencies corresponding to different gaits
- Demonstration that bipedal gait transitions share characteristics with biological locomotion (walk-run transition speed)
- Comparison against fully active control showing 20-40% energy reduction from passive dynamics exploitation

## Methodology Deep-Dive
The central thesis is that bipedal robots, like biological systems, have natural dynamic modes determined by their mechanical properties — link lengths, masses, joint stiffness, and damping. Walking corresponds to pendular dynamics (inverted pendulum swing), running exploits spring-mass bouncing dynamics, and skipping combines elements of both. Rather than fighting these natural modes with fully active control, the paper proposes leveraging them through careful reward design.

The RL framework uses a standard actor-critic architecture with PPO. The observation space includes joint positions, velocities, body orientation, angular velocity, and a target forward velocity command. The action space is joint torques (rather than position targets), which is critical for passive dynamics exploitation — the policy must learn when to apply torques and when to let natural dynamics take over. Position targets would impose rigid tracking that overrides passive dynamics.

The reward function is the key innovation. Instead of heavily penalizing deviation from reference trajectories, the reward emphasizes task-level objectives (velocity tracking, upright posture, survival) while minimizing control effort. The energy minimization term is weighted heavily, creating a strong incentive to find locomotion strategies that use gravity, inertia, and elasticity rather than continuous motor effort. Specific terms include: (1) forward velocity tracking; (2) alive bonus (strong incentive to not fall); (3) energy minimization (sum of squared torques, weighted 2-3x higher than typical locomotion rewards); (4) smoothness penalty (penalizing torque rate of change); and (5) symmetry encouragement (optional, to bias toward symmetric gaits).

The key finding is that gait transitions emerge from the interaction between the speed command and the robot's natural dynamics. At low speeds, the policy discovers pendular walking because it's the most energy-efficient mode for slow locomotion. As speed increases, the policy transitions to running because the energy cost of walking increases nonlinearly while running's spring-mass dynamics become more efficient. The transition speed matches predictions from biomechanical models (Froude number ≈ 0.5), suggesting the RL policy discovers the same principles that govern biological gait transitions.

The role of compliance is analyzed through ablation studies. Robots with stiffer joints tend to produce only walking and running, while robots with appropriately tuned compliance can also produce skipping and galloping gaits. This is because compliance stores and releases energy, enabling the bouncing dynamics needed for running and the asymmetric dynamics of skipping. The authors vary joint stiffness systematically and map out which gaits emerge at each compliance level.

Training uses a single continuous curriculum that gradually increases the commanded speed range. Initially, only low speeds are commanded, and the policy learns walking. As the speed range expands, the policy must discover new gaits to satisfy high-speed commands efficiently. The curriculum naturally produces a multi-gait policy without explicit gait labels or mode switching.

## Key Results & Numbers
- Single policy produces walking (0-1.2 m/s), running (1.2-2.5 m/s), and skipping (intermittent) gaits
- Walk-to-run transition occurs at Froude number ≈ 0.5, matching biological predictions
- 20-40% energy reduction compared to fully active control baselines across all gaits
- Smooth gait transitions with no discrete switching or mode changes
- Compliant actuators enable 2x more gait modes compared to rigid actuators
- Energy-optimal walking speed matches the theoretical prediction from inverted pendulum models
- Policy training converges in ~10M timesteps with standard PPO on a single GPU

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Low**
The focus on bipedal-specific dynamics (inverted pendulum walking, spring-mass running) limits direct applicability to quadruped locomotion. However, the energy minimization reward design and the philosophy of exploiting natural dynamics could inform Mini Cheetah's gait optimization. Quadrupeds also exhibit gait transitions (walk → trot → gallop) that may benefit from similar passive dynamics exploitation. The finding that compliance enables more gait modes is relevant to Mini Cheetah's actuator design choices.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Passive dynamics exploitation is directly relevant to Cassie's compliant actuator design. Cassie's series elastic actuators (SEAs) introduce compliance at each joint, creating exactly the kind of spring-mass dynamics this paper exploits for running and skipping gaits. The gait transition findings inform Project B's primitives level — rather than explicitly programming walk, run, and skip primitives, the system could discover these modes through energy-minimizing reward design. The speed-dependent gait transitions align with Project B's Neural ODE Gait Phase module, which must produce appropriate phase signals for different locomotion modes. The connection between Froude number and gait transitions provides a principled way to set speed thresholds in the primitives selection policy. The energy efficiency improvements from passive dynamics are important for Cassie's battery-limited real-world deployment.

## What to Borrow / Implement
- Incorporate heavy energy minimization weighting in Project B's controller-level reward to exploit Cassie's SEA compliance
- Use torque-space actions (instead of position targets) at the controller level to enable passive dynamics exploitation
- Analyze Cassie's natural dynamics (pendular frequency, spring-mass frequency) to predict expected gait transition speeds
- Design the primitives level to recognize and leverage natural gait transitions rather than imposing them
- Use the Froude number framework to set principled speed ranges for different primitives in Project B
- Validate that Project B's Neural ODE Gait Phase produces phase patterns consistent with the paper's emergent gaits
- Apply the compliance ablation methodology to tune Cassie's SEA stiffness for optimal multi-gait performance

## Limitations & Open Questions
- Demonstrated on simplified bipedal models; transfer to full Cassie dynamics with 20 DoF is non-trivial
- Skipping and galloping gaits are less reliable than walking and running; more training may be needed
- Energy minimization may conflict with robustness — highly efficient gaits can be fragile to perturbations
- Does not address terrain variation; passive dynamics exploitation may be less effective on uneven or compliant terrain
- How does the interaction between robot compliance (SEAs) and terrain compliance affect passive dynamics exploitation?
- Can the discovered gaits be used as reference motions for adversarial imitation learning in Project B?
- What is the trade-off between passive dynamics exploitation and perturbation recovery capability?
- Limited perturbation testing; the robustness of energy-efficient gaits to real-world disturbances needs evaluation
