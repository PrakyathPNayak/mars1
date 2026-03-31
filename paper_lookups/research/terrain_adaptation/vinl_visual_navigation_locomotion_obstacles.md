# ViNL: Visual Navigation and Locomotion over Obstacles

**Authors:** Simar Kareer, Naoki Yokoyama, Dhruv Batra, Sehoon Ha, Joanne Truong
**Year:** 2023 | **Venue:** ICRA 2023
**Links:** https://github.com/SimarKareer/ViNL

---

## Abstract Summary
ViNL presents a modular system combining an egocentric-vision navigation policy that outputs velocity commands with a visually-driven locomotion policy that controls joint-level actions to step over obstacles. Both policies are trained end-to-end with RL in separate simulators and composed for zero-shot transfer to real environments. The system demonstrates autonomous navigation through unknown obstacle-rich indoor environments without any privileged maps.

## Core Contributions
- Modular decomposition of visual navigation and visual locomotion into separate RL policies
- Navigation policy outputs velocity commands from egocentric RGB-D; locomotion policy executes them with obstacle awareness
- Both modules trained independently in different simulators, composed at deployment
- Zero-shot sim-to-real transfer for the full navigation-locomotion stack
- No privileged maps or pre-built environment models required
- Demonstrates indoor navigation over obstacles including steps, thresholds, and clutter
- Open-source codebase for reproducibility

## Methodology Deep-Dive
ViNL decomposes the problem of visual navigation over obstacles into two distinct modules with a clean interface. The navigation policy operates at a high level: it receives egocentric RGB-D images and a goal specification, and outputs desired linear and angular velocity commands. This policy is trained in Habitat simulator with RL, learning to navigate toward goals while avoiding collisions. It treats the robot as a simplified kinematic agent.

The locomotion policy operates at the joint level: it receives the velocity commands from the navigation policy along with proprioceptive state and egocentric depth images, and outputs joint position targets for the robot's legs. This policy is trained in Isaac Gym with RL, learning to track velocity commands while using vision to step over obstacles that a blind policy would fail on. The depth images allow the locomotion policy to anticipate obstacles and adjust gait proactively.

The key insight enabling this modular approach is the velocity command interface between the two modules. The navigation policy doesn't need to know about joint-level control or obstacle stepping; it simply commands where to go. The locomotion policy doesn't need to know about global navigation or goals; it simply follows velocity commands while keeping the robot stable and stepping over nearby obstacles.

Training the locomotion policy involves domain randomization over robot dynamics, obstacle geometries, friction, and visual appearance. The policy learns to handle a distribution of obstacles rather than specific instances. A curriculum over obstacle difficulty is used to progressively train on harder obstacles.

At deployment, the two policies run in parallel: the navigation policy processes images at camera framerate and updates velocity targets, while the locomotion policy runs at a higher frequency to produce smooth joint motions. This two-rate architecture naturally handles the different computational requirements of visual processing and motor control.

## Key Results & Numbers
- Successful navigation through unknown indoor environments with obstacles
- Zero-shot sim-to-real transfer without any real-world fine-tuning
- Handles obstacles up to 15 cm height (steps, thresholds, scattered objects)
- Modular approach outperforms end-to-end single-policy baselines on composite tasks
- Navigation success rate significantly higher with vision-aware locomotion vs. blind locomotion
- Real robot experiments on Unitree A1 quadruped
- Decoupled training reduces total training time compared to monolithic approaches

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
ViNL's modular architecture is directly applicable to Mini Cheetah for autonomous exploration tasks. The velocity command interface between navigation and locomotion aligns with Mini Cheetah's command structure—the high-level navigation module outputs velocity targets that the existing PPO-trained locomotion policy can track. The locomotion policy's use of egocentric depth for obstacle-aware stepping can be integrated with the existing MuJoCo training pipeline. The two-rate architecture (navigation at camera rate, locomotion at 500 Hz PD control) matches Mini Cheetah's computational structure. Domain randomization strategies from ViNL complement existing sim-to-real approaches.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
ViNL's hierarchical navigation→locomotion structure directly mirrors Project B's Planner→Controller architecture. The velocity command interface between ViNL's modules is analogous to the interface between Project B's Planner level and the Primitives level. The modular training approach—each level trained independently—validates Project B's hierarchical training strategy. ViNL's navigation policy could inform the Planner's implementation, particularly how it processes egocentric RGB-D to produce high-level commands. The obstacle-aware locomotion policy demonstrates the value of visual input at the Controller level, supporting the integration of CPTE terrain features. The composition of separately-trained modules for zero-shot deployment is relevant to Project B's multi-level deployment strategy.

## What to Borrow / Implement
- Adopt the velocity command interface between navigation and locomotion policies for both projects
- Use the modular training approach: train navigation in Habitat/similar, locomotion in MuJoCo/Isaac Gym
- Implement the two-rate architecture for camera-rate navigation and high-frequency motor control
- Adapt the obstacle-aware locomotion policy training for Mini Cheetah's 12-DoF system
- Use ViNL's modular composition strategy as a template for Project B's hierarchical deployment
- Leverage the open-source codebase as a starting point for implementation

## Limitations & Open Questions
- Velocity command interface may be too simple for highly dynamic or precise maneuvers
- Modular approach may miss co-adaptation opportunities that end-to-end training could capture
- Indoor focus; unclear how well the navigation policy transfers to outdoor unstructured environments
- Limited to static obstacles; dynamic obstacle avoidance not demonstrated
- The navigation policy assumes flat ground for kinematic planning; stairs and ramps need special handling
- How to extend the modular approach to more than two levels (e.g., Project B's 4-level hierarchy)?
