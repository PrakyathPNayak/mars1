# ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots

**Authors:** David Hoeller, Nikita Rudin, Dhionis Sako, Marco Hutter
**Year:** 2024 | **Venue:** Science Robotics
**Links:** https://arxiv.org/abs/2306.14874

---

## Abstract Summary
This paper presents a fully learned hierarchical approach that enables the ANYmal quadrupedal robot to perform parkour-like agile locomotion, including walking, jumping, climbing over obstacles, and crouching under barriers. The system uses only onboard cameras and proprioceptive sensors, with all perception and control learned end-to-end via reinforcement learning in simulation, then transferred zero-shot to the real robot.

The architecture consists of a perception module that reconstructs the local terrain from noisy depth camera data, a high-level navigation policy that selects locomotion behaviors based on the perceived terrain, and a low-level locomotion controller that generates joint commands to execute the selected behavior. The perception module uses an elevation map representation that is learned to be robust to sensor noise, occlusions, and the robot's own body blocking the camera view. The high-level policy operates at a coarser temporal resolution, making decisions about which obstacle to approach and how to traverse it, while the low-level controller handles the detailed gait timing, foot placement, and balance at the control frequency.

Real-world demonstrations show ANYmal climbing over 0.5m boxes, jumping across 0.4m gaps, crouching under 0.4m barriers, and navigating complex obstacle courses that require sequencing multiple agile maneuvers. The system operates at real-time rates on the robot's onboard computer without any external computation or communication.

## Core Contributions
- **Fully learned parkour** for quadrupedal robots: walking, jumping, climbing, crouching, all from a single trained system
- **Learned elevation map perception** that reconstructs local terrain from noisy depth cameras, robust to occlusions and self-obstruction
- **Hierarchical policy architecture** with a high-level navigation policy and low-level locomotion controller, both trained via RL
- **Massive parallel simulation training** using Isaac Gym with procedurally generated obstacle courses for diverse training scenarios
- **Zero-shot sim-to-real transfer** of the complete perception-planning-control pipeline on ANYmal hardware
- **Real-time onboard execution** without any external computation, demonstrating practical deployment feasibility

## Methodology Deep-Dive
The perception module processes depth images from the robot's front-facing camera and converts them into a local elevation map centered on the robot's base frame. The elevation map is a 2D grid (typically 1m x 1m) with each cell storing the estimated ground height. The depth-to-elevation conversion is learned via a convolutional neural network that handles sensor noise, missing data from reflective or transparent surfaces, and the robot's legs partially blocking the camera view. The CNN is trained in simulation where ground-truth elevation maps are available, using domain randomization on sensor noise, camera pose, lighting, and surface properties.

The high-level navigation policy receives the elevation map and the robot's proprioceptive state (body pose, velocity, joint positions/velocities) and outputs a locomotion command consisting of target forward/lateral velocity, turning rate, and a discrete behavior mode (walk, jump, climb, crouch). This policy operates at 10 Hz and is trained with PPO in Isaac Gym. The training environment consists of procedurally generated obstacle courses with randomized configurations: boxes of varying heights (0.1-0.6m), gaps of varying widths (0.1-0.5m), low barriers of varying heights (0.3-0.6m), stairs, and slopes. Curriculum learning progressively increases obstacle difficulty as the agent's success rate improves.

The low-level locomotion controller receives the behavior mode and velocity commands from the high-level policy, plus the robot's proprioceptive state, and outputs 12 joint position targets at 50 Hz. The controller is trained separately for each behavior mode (walk, jump, climb, crouch) using PPO with mode-specific reward functions. For jumping, the reward emphasizes peak height and forward distance; for climbing, it emphasizes maintaining contact and upward progress; for crouching, it penalizes body height above the clearance threshold. A shared reward structure across modes includes velocity tracking, energy minimization, smoothness, and foot clearance during swing phase.

Domain randomization during training covers physics parameters (friction 0.2-1.5, restitution 0.0-0.5, mass variations of plus or minus 15%), sensor noise (joint encoder noise, IMU bias, camera depth noise), and actuator dynamics (motor response delay 0-20ms, torque limits variation of plus or minus 10%). The training uses approximately 4096 parallel environments and requires 12-24 hours on 4 NVIDIA A100 GPUs for the complete pipeline.

The sim-to-real transfer leverages several techniques beyond domain randomization: actuator network modeling (a learned neural network that maps commanded torques to actual torques, capturing motor dynamics), terrain adaptation through proprioceptive feedback (the low-level controller adjusts footstep timing based on ground contact detection), and conservative action bounds (clipping actions to safe joint ranges to prevent hardware damage).

## Key Results & Numbers
- **Obstacle climbing:** Successfully climbs over boxes up to 50cm high (approximately 70% of the robot's standing height)
- **Gap jumping:** Jumps across gaps up to 40cm wide while maintaining stable landing
- **Crouching:** Navigates under barriers as low as 40cm by lowering body height by 50%
- **Obstacle course completion:** 85% success rate on randomized multi-obstacle courses with 5-8 obstacles
- **Sim-to-real gap:** Less than 10% performance degradation between simulation and real-world execution
- **Speed:** Average forward velocity of 0.5 m/s through obstacle courses, with peak sprint speed of 1.2 m/s on flat ground
- **Perception robustness:** Elevation map remains accurate to within 3cm even with significant sensor noise and occlusion
- **Training time:** 12-24 hours for full pipeline on 4x A100 GPUs; inference runs at 50 Hz on onboard Jetson Orin

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
ANYmal Parkour's hierarchical learned approach with perception is directly applicable to Mini Cheetah obstacle navigation. The architecture can be adapted: use Mini Cheetah's forward-facing camera for elevation map perception, a high-level navigation policy selecting between locomotion modes, and a low-level PPO-trained controller for each mode. The procedurally generated obstacle course training environment provides a template for Mini Cheetah's curriculum learning design.

Specific elements to adopt include: the learned elevation map perception (robust to sensor noise), the multi-behavior locomotion controller with mode switching, and the domain randomization schedule for sim-to-real transfer. Mini Cheetah's lighter weight (9 kg vs ANYmal's 30+ kg) means it can potentially achieve more agile maneuvers, but its less powerful actuators limit the obstacle heights it can handle. The curriculum learning approach (progressively harder obstacles) aligns perfectly with Project A's training philosophy.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The hierarchical learned locomotion architecture is relevant to Cassie's multi-level control, though bipedal parkour has fundamentally different dynamics. ANYmal's four-point stance provides inherent stability during climbing and jumping that Cassie lacks. However, several architectural principles transfer: the elevation map perception module could feed into Cassie's CPTE for terrain-aware gait adaptation, the high-level behavior mode selection is analogous to Cassie's Primitives level selecting gait types, and the procedural obstacle course generation can be adapted for bipedal-appropriate challenges (stairs, ramps, uneven ground).

The domain randomization and actuator network modeling techniques are directly applicable to Cassie's sim-to-real pipeline. Cassie's compliant leg design (leaf springs, four-bar linkages) requires careful actuator modeling, and ANYmal Parkour's approach of learning an actuator network from hardware data provides a template for this.

## What to Borrow / Implement
- **Learned elevation map perception** from depth cameras, robust to noise and occlusion, for terrain-aware locomotion in both projects
- **Procedurally generated obstacle courses** for training curriculum with progressive difficulty scaling
- **Multi-behavior low-level controller** with mode-specific reward functions for different locomotion skills (walk, trot, climb, etc.)
- **Actuator network modeling** that learns motor dynamics from hardware data to improve sim-to-real transfer fidelity
- **Conservative action bounds** during deployment to prevent hardware damage while maintaining performance

## Limitations & Open Questions
- **Quadruped-specific:** The parkour maneuvers (climbing, crouching) rely heavily on ANYmal's quadrupedal stability; direct transfer to bipedal systems like Cassie is non-trivial
- **Fixed behavior modes:** The discrete behavior mode selection (walk/jump/climb/crouch) requires pre-defining the set of locomotion skills; continuous skill blending is not supported
- **Computational requirements:** Training requires 4x A100 GPUs for 12-24 hours, which is significant; the perception module adds inference cost beyond proprioception-only approaches
- **Limited manipulation:** The system is purely locomotion-focused; combining parkour with object interaction (carrying, pushing) is not addressed
