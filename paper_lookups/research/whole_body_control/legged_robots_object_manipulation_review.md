# Legged Robots for Object Manipulation: A Review

**Authors:** Various
**Year:** 2023 | **Venue:** Frontiers in Mechanical Engineering
**Links:** https://www.frontiersin.org/articles/10.3389/fmech.2023.1142421

---

## Abstract Summary
This paper provides a comprehensive review of legged robot manipulation, covering leg-based manipulation, dedicated arm integration, and RL-based loco-manipulation approaches. It catalogues approaches ranging from passive to active manipulation and identifies the significant gap between mature locomotion capabilities and emerging manipulation capabilities. The review highlights the growing importance of unified locomotion-manipulation frameworks for real-world deployment.

## Core Contributions
- Systematic taxonomy of legged robot manipulation: passive (pushing, dragging), leg-based (using feet as end-effectors), and arm-integrated (dedicated manipulators)
- Comprehensive review of RL-based loco-manipulation approaches that learn locomotion and manipulation jointly
- Identification of the "locomotion-manipulation gap" — locomotion is far more mature than manipulation for legged platforms
- Cross-platform comparison of manipulation capabilities across quadrupeds (Spot, ANYmal, A1), bipeds (Atlas, Cassie), and humanoids
- Analysis of sensor requirements for manipulation: proprioception, force/torque sensing, vision, and tactile feedback
- Review of sim-to-real challenges specific to manipulation (contact-rich interactions, object properties, deformable objects)
- Future directions including whole-body loco-manipulation, tool use, and collaborative manipulation

## Methodology Deep-Dive
The review organizes manipulation approaches by the physical mechanism used. Passive manipulation includes pushing objects with the body or feet, leveraging locomotion gaits to move objects without dedicated grasping. This is the simplest form of manipulation and has been demonstrated on quadrupeds using RL-trained locomotion policies augmented with object-relative reward terms.

Leg-based manipulation repurposes the robot's legs as manipulators. Quadrupeds can use one or more legs for manipulation while balancing on the remaining legs, though this significantly complicates balance control. RL approaches train policies that jointly optimize manipulation success and balance maintenance, often using curriculum learning to gradually increase manipulation complexity while maintaining locomotion stability. The review notes that this approach is limited by the workspace and dexterity of legs designed primarily for locomotion.

Arm-integrated approaches add dedicated manipulators to legged platforms (e.g., Spot with an arm, ANYmal with a gripper). The control challenge becomes coordinating whole-body motion — the legs must compensate for the arm's dynamics and reaction forces while maintaining stable locomotion. Hierarchical control approaches are common: a high-level planner coordinates locomotion and manipulation objectives, while low-level controllers execute the resulting whole-body commands. RL methods have shown promise in learning these coordination strategies end-to-end, avoiding the complex analytical modeling of whole-body dynamics.

The RL-based approaches section is particularly relevant. It reviews end-to-end policies that take object state and robot proprioception as input and output whole-body actions. These policies learn to coordinate locomotion and manipulation implicitly through reward shaping. The review notes challenges including sparse manipulation rewards (object barely moves for most of the episode), long-horizon credit assignment (locomotion decisions far in the past affect manipulation success), and sim-to-real transfer of contact-rich behaviors (grasping, pushing forces are hard to simulate accurately).

The review concludes by identifying key gaps. Locomotion has achieved robust sim-to-real transfer, but manipulation remains fragile, particularly for dexterous tasks. The integration of vision and force feedback is critical but underdeveloped. Deformable objects, tool use, and multi-robot collaboration remain largely unexplored for legged manipulators.

## Key Results & Numbers
- Quadruped pushing and dragging achieves 70-90% success rates in simulation, 40-60% in real-world
- Arm-integrated approaches (e.g., Spot with arm) achieve higher manipulation success but add mechanical complexity and weight
- RL-based loco-manipulation narrows the sim-to-real gap compared to classical control, achieving ~80% real-world success for simple tasks
- Leg-based manipulation is limited to simple tasks (pushing, stepping on) due to kinematic constraints
- Contact-rich manipulation remains 2-3x harder to transfer sim-to-real than pure locomotion
- Fewer than 20% of legged manipulation papers demonstrate real-world results (vs. ~60% for pure locomotion)
- Vision-based manipulation for legged robots is still in early stages with limited real-world validation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
While Project A focuses on locomotion, this review contextualizes Mini Cheetah within the broader loco-manipulation landscape. Understanding the manipulation frontier helps identify potential future extensions of the locomotion policy — for example, push recovery tasks where the robot must maintain locomotion while interacting with objects. The review's analysis of how manipulation objectives affect locomotion stability is relevant to designing robust reward functions that anticipate contact with objects in the environment. The coverage of quadruped-specific manipulation (using feet, body pushing) maps to capabilities Mini Cheetah could develop as an extension of its locomotion skills.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
This review informs potential future extensions of Cassie's hierarchy to include manipulation. While Cassie lacks dedicated manipulators, understanding the loco-manipulation control architecture helps design a hierarchy that could accommodate future manipulation modules. The review's discussion of hierarchical control for whole-body coordination (planner → locomotion/manipulation coordination → low-level control) mirrors Project B's Planner→Primitives→Controller→Safety structure. The identified challenges in contact-rich sim-to-real transfer are relevant to improving Cassie's ground contact modeling. The gap between locomotion and manipulation maturity validates the choice to focus on locomotion first while designing an extensible architecture.

## What to Borrow / Implement
- Use the locomotion-manipulation coordination framework as a design reference for future hierarchy extensions in Project B
- Apply the review's reward design insights for contact-rich scenarios to improve push recovery in both projects
- Consider the identified sensor requirements (force/torque, vision) for future hardware integration planning
- Reference the sim-to-real challenges for manipulation to anticipate difficulties when extending to contact-rich tasks
- Use the taxonomy to position both projects within the broader research landscape for publications and proposals

## Limitations & Open Questions
- Review is necessarily broad; implementation-specific guidance is limited
- The field is rapidly evolving; some recent RL-based manipulation results may not be covered
- Limited coverage of bimanual manipulation and tool use, which are emerging frontiers
- Does not provide quantitative comparison methodology across different manipulation approaches
- How can locomotion-focused hierarchies (like Project B) be extended to manipulation without redesigning the architecture?
- What is the minimum additional sensing required to enable useful manipulation on locomotion-focused platforms?
- How does the locomotion-manipulation gap affect the design of foundation models for legged robots?
