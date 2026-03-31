# Reinforcement Learning for Blind Stair Climbing with Legged and Wheeled-Legged Robots

**Authors:** Various (ETH Zurich / École Polytechnique de Montréal)
**Year:** 2024 | **Venue:** arXiv 2024
**Links:** https://arxiv.org/abs/2402.06143

---

## Abstract Summary
Develops a versatile RL controller for stair climbing applicable to quadrupeds, bipeds, and wheeled-legged robots. Uses position-based task formulation with asymmetric actor-critic and proprioceptive-only deployment (blind climbing). Features mode-switching for activating stair behavior. Validated in simulation and on real Ascento robot climbing 15cm stairs without any exteroceptive sensing.

## Core Contributions
- Demonstrates blind (proprioceptive-only) stair climbing without any vision, LiDAR, or terrain estimation
- Proposes a position-based task formulation that simplifies stair climbing to a height-gain objective
- Employs asymmetric actor-critic where the critic uses privileged terrain information during training
- Implements a mode-switching mechanism that transitions between flat-ground and stair-climbing behaviors
- Validates across multiple robot morphologies: quadruped, biped, and wheeled-legged (Ascento)
- Achieves real-world deployment on Ascento climbing 15cm stairs with zero-shot sim-to-real transfer
- Shows that proprioceptive feedback alone (joint positions, velocities, IMU) provides sufficient information for stair navigation

## Methodology Deep-Dive
The position-based task formulation reframes stair climbing from a complex terrain-aware navigation problem to a simpler vertical progress objective. Instead of requiring explicit stair detection and foot placement planning, the policy is rewarded for achieving positive height change while maintaining forward velocity and body stability. This formulation allows the policy to discover its own stair-climbing strategy through RL exploration rather than following a prescribed foot placement pattern.

The asymmetric actor-critic architecture is central to the approach. During training in simulation, the critic receives privileged observations including a local terrain heightmap around the robot (1m × 1m grid at 5cm resolution), exact contact states for all feet, and ground truth body height. The actor, however, only receives proprioceptive observations: joint positions, joint velocities, body orientation (from IMU), angular velocity, gravity vector projection, and the previous action. This asymmetry allows the critic to provide accurate value estimates that guide the actor's learning without requiring exteroceptive sensing at deployment. The actor effectively learns an implicit terrain representation from proprioceptive signals—contact timing, joint torque feedback, and body pitch changes encode stair geometry.

The mode-switching mechanism addresses the challenge of deploying a single policy across flat ground and stairs. A binary mode signal is provided as input to the policy, indicating whether stair-climbing behavior should be activated. During flat-ground mode, the policy optimizes for standard locomotion objectives (velocity tracking, efficiency). During stair mode, the height-gain reward is activated and the policy shifts to stair-climbing behavior. The transition between modes is triggered by a simple user command or an external stair detector, keeping the proprioceptive-only deployment constraint intact.

Training uses PPO with the standard asymmetric formulation. The terrain curriculum during training includes flat ground, small steps (5cm), medium steps (10cm), and large steps (15cm), with automatic progression based on climbing success rate. Domain randomization covers step height variation (±2cm), step depth (25-35cm), friction coefficient (0.4-1.0), mass perturbation (±10%), and motor strength (±15%). A key finding is that training on a distribution of step heights produces policies that generalize to unseen heights better than training on a single height.

The cross-morphology validation demonstrates the generality of the approach. The same position-based formulation and asymmetric training framework are applied to three different robots with only reward weight adjustments and morphology-specific observation dimensions. All three platforms learn effective stair climbing, though the strategies differ: quadrupeds use a diagonal stepping pattern, bipeds use alternating foot placement with increased knee flexion, and the wheeled-legged robot combines wheel rolling on stair surfaces with leg-assisted lifting.

## Key Results & Numbers
- Blind stair climbing of 15cm stairs on real Ascento wheeled-legged robot
- Zero-shot sim-to-real transfer with proprioceptive-only policy
- 95% success rate on 10cm stairs, 82% on 15cm stairs in simulation across morphologies
- Asymmetric critic improves sample efficiency by 3x compared to symmetric actor-critic
- Mode-switching adds < 2% overhead to policy size and inference time
- Policy runs at 100 Hz with 500 Hz PD control on real hardware
- Cross-morphology validation on quadruped, biped, and wheeled-legged platforms
- Training converges in ~2 hours on a single GPU with 2048 parallel environments

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
Blind stair climbing is directly applicable to Mini Cheetah's indoor deployment scenarios where stairs are common obstacles. The position-based task formulation eliminates the need for explicit stair detection, which is valuable for Mini Cheetah's proprioceptive-focused design. The asymmetric actor-critic architecture with privileged terrain information during training matches the training paradigm for Mini Cheetah in MuJoCo (where terrain data is available) while deploying with only joint encoders and IMU. The mode-switching mechanism can be integrated into Mini Cheetah's command interface for triggering stair behavior. The specific domain randomization ranges validated for stair climbing provide a tested starting point for Mini Cheetah training.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Bipedal stair climbing is directly validated in this paper, making it highly relevant to Cassie. The asymmetric training paradigm aligns naturally with Project B's Dual Asymmetric-Context Transformer, where the exteroceptive context branch could serve a similar role to the privileged critic during training. The mode-switching concept maps to Project B's Primitives level, where stair climbing could be one of the learned motion primitives selected by the Planner. The proprioceptive-only deployment constraint is particularly important for Cassie, which has limited exteroceptive sensing. The position-based formulation could be integrated as a sub-objective within the hierarchical reward structure. The biped-specific strategies (alternating foot placement, increased knee flexion) provide direct behavioral targets for Cassie's stair climbing primitive.

## What to Borrow / Implement
- Implement the position-based height-gain objective for stair climbing training on Mini Cheetah
- Adopt the asymmetric actor-critic with terrain heightmap as privileged critic observation
- Add a stair-climbing mode as a motion primitive in Project B's hierarchy
- Use the validated domain randomization ranges for stair-specific training
- Implement the mode-switching mechanism as part of the command interface for both platforms
- Apply the cross-morphology insight: train on varied stair heights rather than a single target height

## Limitations & Open Questions
- 15cm is the maximum validated stair height; taller stairs (20cm+) may require different strategies
- Blind descending stairs is more challenging than ascending and is less thoroughly evaluated
- Mode-switching requires an external trigger rather than autonomous stair detection
- The approach does not handle spiral stairs, uneven stairs, or stairs with railings
- Long continuous stairwell climbing (50+ steps) is not evaluated for drift and stability
- How to combine stair climbing with other terrain traversal behaviors in a unified policy
