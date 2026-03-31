# Dynamic Fall Recovery Control for Legged Robots via Reinforcement Learning

**Authors:** (MDPI Biomimetics Contributors)
**Year:** 2023 | **Venue:** MDPI Biomimetics
**Links:** [DOI: 10.3390/biomimetics9040193](https://www.mdpi.com/2313-7673/9/4/193)

---

## Abstract Summary
This paper presents a reinforcement learning framework for training legged robots to both resist disturbances and recover from falls using only proprioceptive sensor data. The approach trains a unified policy that handles the full continuum of perturbations—from mild pushes that require balance corrections to severe impacts that cause full falls and require stand-up recovery. Unlike prior methods that use separate controllers for locomotion and recovery, the unified policy seamlessly transitions between walking, stumbling, and getting back up without explicit mode switching.

The key technical contribution is a training methodology that combines adversarial perturbation injection with a phased curriculum. During training, the environment applies random external forces of increasing magnitude to the robot, forcing it to learn increasingly aggressive recovery behaviors. The curriculum begins with small perturbations (balance corrections during walking) and progressively increases to full knockdowns requiring multi-second recovery sequences. The policy receives only proprioceptive inputs: joint angles, joint velocities, body orientation (from IMU), and a short history of recent observations.

The approach is demonstrated on simulated quadruped and bipedal robots, showing successful recovery from falls in diverse configurations (on the side, on the back, face-down) across indoor and outdoor terrains. The proprioceptive-only design is deliberate—it avoids dependence on cameras or LiDAR that may be occluded or damaged during a fall, making the system more robust for real-world deployment.

## Core Contributions
- Unified RL policy handling the full disturbance spectrum from balance corrections to full fall recovery without mode switching
- Adversarial perturbation curriculum that progressively trains robustness from mild pushes to severe knockdowns
- Proprioceptive-only input design eliminating dependence on exteroceptive sensors (cameras, LiDAR) that may fail during falls
- Observation history mechanism providing temporal context for recovery state estimation (velocity estimation, contact pattern recognition)
- Demonstration on both quadruped and bipedal morphologies, showing generalization of the training methodology
- Analysis of learned recovery strategies showing emergence of physically plausible motions (rolling, pushing up, leg spreading for stability)
- Terrain generalization testing across flat ground, slopes, and uneven surfaces without terrain-specific training

## Methodology Deep-Dive
The training framework uses PPO with a carefully designed observation space and reward function within a MuJoCo-based simulation. The observation vector at each time step includes: (1) joint angles for all actuated joints (12 for quadruped, 10 for bipedal), (2) joint velocities, (3) body orientation represented as a rotation matrix or quaternion, (4) body angular velocity, (5) previous action (for action smoothness), and (6) a binary contact indicator for each foot. Critically, no ground-truth body position or linear velocity is provided, as these would not be available from proprioceptive sensors alone.

To compensate for the missing velocity information, the authors use an observation history buffer that concatenates the last 10 observations into the policy input. This temporal context allows the policy network (an MLP with 3 hidden layers of 256 units each) to implicitly estimate velocity through finite differences and recognize contact patterns. The history buffer also provides information about the current phase of a recovery maneuver—whether the robot is still falling, has just landed, or is in the process of standing up.

The reward function consists of several components with phase-dependent weighting: (1) an alive bonus (+1 per step when the torso height exceeds a threshold), (2) a forward velocity reward (tracking a commanded velocity), (3) an upright orientation reward (penalizing deviation of the body z-axis from vertical), (4) an energy penalty (sum of squared joint torques), (5) an action smoothness penalty (penalizing changes between consecutive actions), and (6) a recovery bonus (large positive reward when transitioning from a fallen state to upright within a time window). The recovery bonus is critical for incentivizing get-up behaviors that would otherwise yield low cumulative reward due to the prolonged non-upright period.

The adversarial perturbation curriculum applies external forces to the robot's base link at random intervals. Forces are sampled from a uniform distribution whose upper bound increases with training progress: starting at 20 N (mild push during walking), increasing to 100 N (stumble-inducing), and ultimately reaching 500+ N (full knockdown). Force directions are randomized in 3D, and application duration varies from a single time step (impulse) to 0.5 seconds (sustained push). This curriculum ensures the policy encounters increasingly difficult scenarios as its recovery capability grows.

Additionally, the authors implement a state initialization randomization scheme where, with increasing probability during training, the robot is initialized in non-standard configurations—lying on its side, upside down, or in mid-air. This bypasses the slow curriculum progression for fall recovery by directly training from fallen states, significantly accelerating the learning of stand-up behaviors.

Domain randomization is applied to mass (±15%), friction (±30%), motor strength (±10%), and sensor noise (Gaussian with σ = 0.02) to improve robustness. The total training uses 4096 parallel environments for approximately 500M time steps, completing in roughly 12 hours on a single GPU workstation.

## Key Results & Numbers
- Recovery success rate from full knockdown: 87% on flat ground, 72% on mildly uneven terrain
- Balance correction success (mild perturbations): 98% across all terrains
- Average recovery time from side-lying position: 2.3 seconds (quadruped), 3.8 seconds (bipedal)
- Zero-shot generalization to 10-degree slopes without slope-specific training
- Proprioceptive-only policy achieves 94% of the performance of a policy with full state access
- Training time: ~12 hours on single GPU (4096 parallel environments, 500M steps)
- Observation history length ablation: 10-step history optimal; 5-step reduces recovery success by 15%; 20-step adds no benefit
- Energy consumption during recovery: 40% lower than a hand-designed finite state machine recovery controller

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Fall recovery is a critical capability for Mini Cheetah, especially when operating on uneven terrain or under external disturbances. This paper's unified policy approach is directly applicable—rather than designing separate controllers for locomotion and recovery, a single PPO-trained policy can handle both. The adversarial perturbation curriculum integrates naturally into Mini Cheetah's existing training pipeline, and the proprioceptive-only design matches Mini Cheetah's typical sensor suite (joint encoders + IMU).

The observation history mechanism is particularly relevant for Mini Cheetah, which lacks ground-truth velocity estimation on hardware. Using a history of proprioceptive observations for implicit velocity estimation has been shown to work well on similar quadruped platforms. The 87% recovery success rate from full knockdown, while not perfect, represents a significant improvement over the complete inability to recover that most standard locomotion policies exhibit.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Fall recovery is directly relevant to the Safety level of Cassie's 4-level hierarchy, which is responsible for maintaining operational safety through LCBF (Learned Control Barrier Function) constraints. The unified policy approach demonstrated here could serve as the Safety level's recovery module, activated when the LCBF detects imminent violation of stability constraints. The proprioceptive-only design aligns with Project B's sensing approach, where Cassie relies primarily on joint encoders and IMU.

The adversarial perturbation curriculum is conceptually aligned with the Adversarial Curriculum at the Primitives level. The perturbation forces can be viewed as a simplified form of adversarial environment generation, where the difficulty of disturbances is progressively increased. The observation history mechanism is directly applicable to the Dual Asymmetric-Context Transformer—the history buffer could feed into the asymmetric context, providing temporal information about disturbance patterns. The bipedal recovery results (3.8s recovery time, 72% success on uneven terrain) provide benchmark targets for Cassie's Safety module.

## What to Borrow / Implement
- Integrate adversarial perturbation injection into both Mini Cheetah and Cassie training pipelines with progressive force magnitude curriculum
- Use the observation history buffer (10-step) for proprioceptive velocity estimation in both projects
- Implement state initialization randomization for training recovery from fallen states without waiting for the curriculum to produce falls naturally
- Adopt the recovery bonus reward component for incentivizing stand-up behaviors in addition to standard locomotion rewards
- Deploy the unified policy approach at Cassie's Safety level, replacing separate locomotion and recovery controllers

## Limitations & Open Questions
- Recovery success rate on uneven terrain (72%) leaves room for improvement, especially for real-world deployment where terrain is always uneven
- Bipedal recovery is significantly slower and less reliable than quadruped recovery (3.8s vs. 2.3s), suggesting bipedal-specific challenges require additional research
- The proprioceptive-only approach cannot anticipate disturbances before contact, limiting proactive bracing behavior that vision could enable
- Sim-to-real transfer is not demonstrated; the gap between simulated and real contact dynamics during recovery maneuvers may be substantial
