---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/sim_to_real_humanoid_box_loco_manipulation.md

**Title:** Sim-to-Real Learning for Humanoid Box Loco-Manipulation
**Authors:** Jeremy Dao, Helei Duan, Alan Fern, Jonathan Hurst
**Year:** 2024
**Venue:** ICRA 2024
**arXiv / DOI:** Available via NSF PAR (10581781)

**Abstract Summary (2-3 sentences):**
This paper presents the first sim-to-real reinforcement learning controller for bipedal loco-manipulation on the Digit humanoid robot, demonstrating a complete pick-carry-place task pipeline. The robot learns to walk to a box, pick it up, carry it to a target location, and set it down using a single policy trained with combined locomotion and manipulation rewards under extensive domain randomization. The work demonstrates that end-to-end RL can handle the complex coordination between bipedal balance and upper-body manipulation without explicit task decomposition.

**Core Contributions (bullet list, 4-7 items):**
- First successful sim-to-real bipedal loco-manipulation via reinforcement learning on the Digit humanoid
- Unified end-to-end policy architecture handling both locomotion and manipulation within a single network
- Combined reward formulation integrating locomotion stability, navigation, and manipulation task completion objectives
- Comprehensive domain randomization strategy addressing both locomotion dynamics and manipulation contact variability
- Demonstration of complete pick-carry-place task sequence on real hardware without human intervention
- Analysis of the interaction between bipedal balance maintenance and upper-body manipulation forces

**Methodology Deep-Dive (3-5 paragraphs):**
The framework trains a single neural network policy to control all of Digit's actuated joints (both legs and arms) for the complete loco-manipulation task sequence. The observation space includes full proprioceptive state (joint angles and velocities for all limbs), pelvis IMU readings, and task-relevant information such as the relative position and orientation of the target box and the goal placement location. The action space outputs target joint positions for both the leg joints (controlling locomotion) and the arm joints (controlling reaching, grasping, and carrying). This unified approach avoids the need for separate locomotion and manipulation modules that must be carefully coordinated, instead letting the RL optimizer discover the appropriate coordination strategies through end-to-end training.

The reward function is structured as a weighted combination of several components designed to encourage the full task sequence. Locomotion rewards include velocity tracking toward the box or goal location, upright posture maintenance, foot contact regularity, and energy efficiency. Manipulation rewards include reaching distance to the box, grasp success indicators, box stability during carrying (minimizing box orientation changes), and placement accuracy at the goal. A key challenge is balancing these competing objectives, as aggressive reaching for the box can destabilize bipedal balance, while overly conservative locomotion can prevent successful manipulation. The authors use a phased reward structure where different reward terms are emphasized at different stages of the task, guided by an automatic phase detector that identifies whether the robot is approaching, grasping, carrying, or placing.

Training is conducted in Isaac Gym using a simulated Digit model with full rigid-body dynamics and simplified contact models for the box interaction. Domain randomization is applied extensively across two categories: locomotion dynamics (ground friction, body mass, joint friction, motor strength, IMU noise) and manipulation dynamics (box mass ranging from 1-5 kg, box friction, box dimensions, initial box position variation, grasp point uncertainty). The randomization of manipulation-specific parameters is critical because contact dynamics between the robot's gripper and the box are particularly difficult to simulate accurately. The authors find that without manipulation-specific randomization, policies that work perfectly in simulation fail to grasp the box reliably on hardware.

The sim-to-real transfer pipeline deploys the trained policy at 50 Hz on Digit's onboard computer. The robot uses its onboard perception system to estimate the box position relative to its body, and this estimate is fed into the policy as part of the observation. The authors note that perception noise is a significant source of transfer difficulty for the manipulation component, as small errors in box position estimation can lead to grasp failures. To address this, observation noise is randomized aggressively during training, and the policy learns to approach the box slowly and use contact feedback (through arm joint torque observations) to refine its grasp strategy.

**Key Results & Numbers:**
- First successful bipedal loco-manipulation via sim-to-real RL
- Complete pick-carry-place task executed on real Digit robot
- Box mass range of 1-5 kg handled successfully
- Task completion rate of approximately 70-80% on real hardware
- Walking speed of up to 0.5 m/s while carrying the box
- Policy runs at 50 Hz on Digit's onboard compute
- End-to-end task completion in under 60 seconds for a 3-meter carry distance

**Relevance to Project A (Mini Cheetah):** LOW — The manipulation task on a bipedal robot is not directly relevant to quadruped locomotion. The Mini Cheetah does not have manipulation capabilities, and the core locomotion challenges differ substantially.

**Relevance to Project B (Cassie HRL):** MEDIUM — While Cassie does not have arms for manipulation, the loco-manipulation concepts are relevant to extending the bipedal controller for tasks that involve interaction with the environment. The unified reward formulation balancing multiple objectives is applicable to the hierarchical controller design, and the phased reward structure could inform curriculum design across hierarchy levels.

**What to Borrow / Implement:**
- Phased reward structure concept for multi-stage task completion in the hierarchical controller
- Domain randomization strategy covering both dynamics and perception uncertainty
- The insight that end-to-end RL can discover coordination strategies between different body parts and objectives
- Task sequence decomposition within a single policy as a comparison point for the explicit hierarchical approach

**Limitations & Open Questions:**
- Single specific task (box pick-carry-place); limited generalization to other manipulation tasks or object types
- Task completion rate of 70-80% on hardware leaves room for improvement
- Perception noise is a major bottleneck not fully addressed by the RL policy alone
- Simplified contact model for grasping may not capture all real-world contact phenomena
- No adaptation to box properties discovered during manipulation
- Digit platform differs from Cassie (has arms), limiting direct architectural transfer
---
