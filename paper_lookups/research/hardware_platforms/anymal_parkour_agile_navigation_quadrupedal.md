# ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots

**Authors:** David Hoeller, Nikita Rudin, Dhionis Sako, Marco Hutter
**Year:** 2024 | **Venue:** Science Robotics (2024) / arXiv 2306.14874
**Links:** https://arxiv.org/abs/2306.14874

---

## Abstract Summary
A fully learned hierarchical system enabling ANYmal to perform parkour: walking, jumping, climbing, crawling, and crouching. All modules (perception, locomotion, navigation) are trained in simulation via RL and transferred zero-shot to real hardware. The system achieves speeds up to 2 m/s across parkour-like environments without expert demonstrations, representing one of the most capable quadruped locomotion systems demonstrated on real hardware.

## Core Contributions
- Demonstrated a fully learned hierarchical system for quadruped parkour with no expert demonstrations or hand-designed motion primitives
- Achieved zero-shot sim-to-real transfer for complex agile behaviors (jumping, climbing, crawling, crouching) trained entirely in simulation
- Introduced a hierarchical architecture with separate perception, skill selection, and locomotion modules that compose complex behaviors from learned primitives
- Reached traversal speeds of up to 2 m/s and jumps up to 1 m on real ANYmal hardware
- Showed that RL-trained locomotion skills can be composed and sequenced by a learned navigation policy to handle diverse obstacles
- Trained all components end-to-end in simulation using domain randomization and curriculum learning for robust real-world transfer
- Provided a complete system-level demonstration: from perception to planning to execution, fully learned

## Methodology Deep-Dive
The ANYmal Parkour system uses a hierarchical architecture with three main levels: a navigation policy (high-level), a skill selection module (mid-level), and locomotion controllers (low-level). Each level is trained separately in simulation using RL, with domain randomization and curriculum learning ensuring robust transfer to real hardware.

At the low level, individual locomotion skills are trained as separate RL policies: walking, trotting, jumping, climbing up/down, crawling, and crouching. Each skill policy takes proprioceptive observations (joint angles, velocities, body orientation, angular velocity) and a command (velocity, heading) as input, and outputs target joint angles for the robot's PD controllers. Skills are trained in environments specifically designed for each behavior (e.g., stairs of varying heights for climbing, low-ceiling passages for crawling). Domain randomization over terrain parameters, friction, mass, motor dynamics, and observation noise ensures robustness.

The mid-level skill selection module is a learned policy that chooses which low-level skill to activate based on the current proprioceptive state and terrain perception. This module observes the robot's immediate surroundings through a learned terrain representation (from elevation maps or depth images processed by a CNN) and selects the most appropriate skill for the current situation. The selection is soft — skills can be blended during transitions — and the module is trained with a composite reward that evaluates navigation progress and stability.

The high-level navigation policy plans over longer horizons, providing velocity and heading commands to the skill selection module based on the desired goal location and the perceived environment structure. This policy uses a broader spatial representation of the environment and reasons about which sequence of obstacles and terrains lies ahead. It learns to choose paths that are within the robot's locomotion capabilities, avoiding obstacles that cannot be traversed.

Training uses a progressive curriculum: first, individual skills are trained on simple terrain, then terrain difficulty is gradually increased, then the skill selection module is trained to compose skills, and finally the navigation policy is trained in full environments. This curriculum avoids the challenge of learning everything simultaneously and ensures each component is robust before being integrated. The curriculum automatically adjusts difficulty based on the policy's current success rate, ensuring consistent learning progress.

Sim-to-real transfer relies heavily on domain randomization and a teacher-student training scheme. Teachers are trained with privileged information (ground-truth terrain heightmaps, exact friction values), and students are distilled from teachers using only observations available on real hardware (onboard depth cameras, proprioception). This privileged-information distillation approach is critical for handling the perception gap between simulation and reality.

## Key Results & Numbers
- Traversal speed up to 2 m/s across parkour-like environments
- Jumps up to 1 m height and 1.5 m length on real hardware
- Fully zero-shot sim-to-real transfer — no real-world fine-tuning
- No expert demonstrations needed — all skills learned from reward signals alone
- 6 locomotion skills learned: walking, trotting, jumping, climbing, crawling, crouching
- Curriculum training: ~24 hours on 8 GPUs in IsaacGym
- Domain randomization over 15+ parameters: terrain, friction, mass, motor strength, sensor noise, latency
- Teacher-student distillation for bridging the perception sim-to-real gap
- Successfully navigated 10 different parkour courses in real-world evaluation
- Failure rate: <5% on courses within the robot's physical capabilities

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The hierarchical skill architecture is directly applicable to Mini Cheetah agile locomotion. The separate skill training → skill selection → navigation pipeline provides a blueprint for achieving parkour-level capabilities on Mini Cheetah. The curriculum learning and domain randomization strategies (15+ parameters) provide concrete guidance for robust sim-to-real transfer. The teacher-student distillation approach for perception is especially relevant if adding visual terrain awareness to Mini Cheetah. The individual skill training could leverage the existing PPO + MuJoCo pipeline, with skills composed by a learned selector. The 24-hour training time on 8 GPUs is practical for academic research.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The hierarchical navigation + locomotion architecture directly parallels Project B's Planner → Primitives structure. ANYmal Parkour demonstrates that learned skill selection can compose diverse locomotion behaviors for obstacle traversal — exactly what Project B's primitives layer must do for Cassie. The teacher-student distillation for perception integration is relevant to the CPTE terrain encoder's sim-to-real transfer. The progressive curriculum (individual skills → composition → navigation) provides a training recipe for the hierarchical architecture. Key differences: Cassie's bipedal nature requires different skills (balance, dynamic walking, running) and the hierarchy must include explicit safety guarantees (LCBF) not present in ANYmal Parkour.

## What to Borrow / Implement
- Adopt the hierarchical architecture pattern: separate skill training → learned skill selector → high-level navigation
- Implement progressive curriculum learning: train skills individually, then train composition, then train navigation
- Use teacher-student distillation with privileged information for robust perception transfer
- Apply domain randomization over 15+ parameters for comprehensive sim-to-real robustness
- Train individual locomotion primitives (walk, trot, jump, climb) as separate PPO policies before composing
- Implement the automatic difficulty curriculum that adjusts based on current success rate
- Use the elevation map terrain representation for the perception module
- Evaluate on a set of standardized parkour courses to benchmark locomotion capability progression

## Limitations & Open Questions
- ANYmal is a heavy, expensive robot — Mini Cheetah and Cassie have different dynamics and capabilities
- No explicit safety layer — the system relies on learned behavior to avoid dangerous situations
- Skill transitions can be abrupt and may cause instability during switching
- Only demonstrated on structured parkour courses — unstructured natural terrain may pose different challenges
- The hierarchical training requires careful engineering of inter-level interfaces
- Teacher-student distillation introduces a performance gap compared to the privileged teacher
- Bipedal parkour is significantly harder than quadruped — direct transfer of the approach is non-trivial
- No recovery behaviors for falls — the system must avoid falling rather than recovering from it
- Computational requirements (8 GPUs, 24 hours) may be challenging for iterative development
