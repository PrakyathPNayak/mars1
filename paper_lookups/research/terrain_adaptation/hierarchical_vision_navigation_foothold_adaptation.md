# Hierarchical Vision Navigation System for Quadruped Robots with Foothold Adaptation Learning

**Authors:** Various
**Year:** 2023 | **Venue:** MDPI Sensors
**Links:** https://www.mdpi.com/1424-8220/23/11/5194

---

## Abstract Summary
This paper combines a high-level visual navigation policy for route planning with a low-level RL-based foothold adaptation network for dynamic and cluttered environments. The high-level planner processes visual input to guide the robot toward goal locations while the low-level network adjusts individual leg placements for safe and stable stepping using egocentric perception. The system enables autonomous navigation in cluttered, obstacle-rich environments.

## Core Contributions
- Hierarchical architecture separating visual navigation planning from foothold-level control
- High-level navigation policy using visual input to produce directional guidance
- Low-level RL-based foothold adaptation network that adjusts leg placements for safety
- Egocentric perception pipeline feeding both navigation and foothold levels
- Demonstrated navigation in cluttered, dynamic environments with varied obstacles
- Safe foot placement as an explicit objective at the low-level controller
- Integration of visual route planning with physically-aware leg control

## Methodology Deep-Dive
The system architecture follows a clear hierarchical decomposition. The high-level navigation module processes visual input (egocentric camera images) to understand the environment layout and plan routes toward specified goals. This module outputs high-level directional commands—effectively steering the robot along collision-free paths. It operates at the planning timescale, updating at camera framerate.

The low-level foothold adaptation module receives these high-level commands along with egocentric depth/visual data of the immediate terrain and proprioceptive state. Its task is to execute the commanded direction while adapting individual foot placements for safety and stability. This module is trained with RL in simulation, where the reward includes terms for following high-level commands, foot placement safety (avoiding edges, holes, and unstable surfaces), energy efficiency, and gait stability.

The foothold adaptation network uses a terrain-aware architecture: local terrain patches around each foot's projected landing position are extracted from the depth image and processed by a CNN. These terrain features are concatenated with proprioceptive state and the high-level command, then processed by an MLP policy that outputs foot placement adjustments. The adjustments modify the nominal gait pattern to shift foot placements to safer locations—e.g., moving a foot away from a detected edge or gap.

Training uses domain randomization over terrain types (scattered objects, gaps, steps, slopes), obstacle sizes and placements, visual appearance, and robot dynamics. A curriculum progressively increases clutter density and obstacle difficulty. The high-level navigation module is trained separately in a visual navigation environment, then the two modules are composed at deployment.

The integration between levels is through the directional command interface. The navigation module provides desired heading and speed, while the foothold module handles the physical details of achieving that direction safely. This decomposition allows each module to focus on its core competency and enables independent improvement of either module.

## Key Results & Numbers
- Successful navigation through cluttered indoor environments with diverse obstacles
- Foothold adaptation reduces foot slip and instability by 60% compared to fixed gait patterns
- High-level navigation achieves 85%+ success rate in reaching goals in previously unseen environments
- Safe foot placement in 95% of steps on challenging terrain (gaps, edges, scattered objects)
- Hierarchical approach outperforms monolithic end-to-end policies on composite tasks
- Real-time operation with navigation at 10 Hz and foothold adaptation at 50 Hz
- Validated on quadruped robot platform in simulation with realistic physics

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper's hierarchical navigation system is directly applicable to Mini Cheetah for autonomous operation in cluttered environments. The foothold adaptation network addresses a capability gap in proprioception-only policies: explicit foot placement optimization. For Mini Cheetah's 12-DoF system, the foot placement adjustments map directly to modifications of the hip and knee targets produced by the RL policy. The terrain-aware CNN extracting local patches around each foot's projected landing position could be integrated with the depth camera. The 50 Hz foothold adaptation rate is compatible with the 500 Hz PD control loop—foothold targets update at 50 Hz while PD tracking runs at 500 Hz. The system could be integrated with Mini Cheetah's existing PPO-trained velocity-tracking policy.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The high-level/low-level split directly mirrors Project B's Planner→Controller architecture. The navigation module corresponds to the Planner level that outputs high-level commands based on visual perception. The foothold adaptation module corresponds to the Controller level that executes physical motions safely. This paper validates the hierarchical decomposition approach used in Project B. The foothold adaptation is particularly relevant for Cassie's bipedal locomotion, where each foot placement critically affects balance—the terrain-aware foot placement network could be integrated with the Differentiable Capture Point module. The safe foot placement objective aligns with the LCBF (Learned CBF + QP) safety layer. The local terrain patch extraction for each foot could enhance the MC-GAT's per-joint terrain features.

## What to Borrow / Implement
- Adopt the foothold adaptation network architecture for explicit foot placement optimization
- Extract local terrain patches around each foot's projected landing point from depth images
- Implement the hierarchical command interface: navigation outputs direction, controller handles physics
- Use the curriculum over clutter density for training in increasingly difficult environments
- Integrate foothold safety as a constraint in the LCBF safety layer for Project B
- Apply the dual-timescale architecture: navigation at camera rate, foothold control at policy rate

## Limitations & Open Questions
- High-level navigation module is relatively simple; may struggle with complex multi-room environments
- Foothold adaptation assumes terrain is static between observation and foot contact
- The fixed command interface (direction + speed) may be too restrictive for complex maneuvers
- Scalability to outdoor environments with diverse terrain types not fully demonstrated
- How to handle dynamic obstacles that change between planning and execution?
- Integration of foothold adaptation with whole-body balance control for bipedal robots needs further study
