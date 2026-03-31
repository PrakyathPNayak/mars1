---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/dynamic_bipedal_maneuvers_sim_to_real_rl.md

**Title:** Dynamic Bipedal Maneuvers through Sim-to-Real Reinforcement Learning
**Authors:** Helei Duan, Ashish Malik, Mohitvishnu S. Gadde, Jeremy Dao, Alan Fern, Jonathan Hurst
**Year:** 2022
**Venue:** IEEE-RAS International Conference on Humanoid Robots (Humanoids 2022)
**arXiv / DOI:** arXiv:2207.07835

**Abstract Summary (2–3 sentences):**
This work presents an RL framework for achieving dynamic bipedal maneuvers on the Cassie robot, specifically demonstrating four-step 90-degree turns successfully transferred from simulation to the real world. The approach uses a recurrent policy architecture combined with epilogue terminal rewards and reference trajectory data generated from a single rigid body model. This represents one of the first sim-to-real demonstrations of highly dynamic, non-steady-state bipedal maneuvers.

**Core Contributions (bullet list, 4–7 items):**
- First sim-to-real demonstration of dynamic 90-degree turning maneuvers on Cassie bipedal robot
- Recurrent policy architecture (GRU-based) for handling partial observability in dynamic maneuvers
- Epilogue terminal reward formulation that encourages the policy to reach desirable terminal states after completing maneuvers
- Use of single rigid body (SRB) model to generate reference trajectories for dynamic maneuvers
- Novel reward shaping combining reference tracking with terminal state quality assessment
- Demonstration that complex dynamic behaviors can transfer zero-shot with appropriate domain randomization

**Methodology Deep-Dive (3–5 paragraphs):**
The methodology begins with generating reference trajectories for the target maneuver (90-degree turn) using a simplified single rigid body (SRB) model of Cassie. The SRB model captures the essential center-of-mass dynamics and ground reaction force profiles needed for the turn without the complexity of the full multi-body dynamics. These reference trajectories provide a coarse motion plan specifying desired center-of-mass positions, velocities, and orientation profiles over the four-step turning sequence. The reference serves as a guide rather than a strict constraint, allowing the full-body RL policy to discover its own joint-level strategies while respecting the overall dynamic feasibility established by the SRB plan.

The policy architecture employs a GRU (Gated Recurrent Unit) recurrent neural network to handle the partial observability inherent in bipedal locomotion, where contact states, terrain properties, and actuator dynamics are not directly measured. The recurrent network receives proprioceptive observations (joint angles, angular velocities, IMU orientation and angular velocity, and the phase variable indicating progress through the maneuver) and maintains a hidden state that implicitly encodes estimates of unobserved quantities. The policy outputs target joint positions for Cassie's 10 actuated joints, which are then tracked by low-level PD controllers operating at 2 kHz on the robot's real-time control computer.

Training is conducted using PPO in the MuJoCo simulator with a reward function that combines several components: reference trajectory tracking (penalizing deviations from the SRB-generated center-of-mass trajectory), joint regularization (penalizing excessive joint velocities and torques), and the novel epilogue terminal reward. The epilogue reward is computed by evaluating the robot's state at the end of the maneuver sequence against criteria for successful completion—specifically, whether the robot has achieved the desired heading change, maintained balance, and arrived at a state from which steady-state walking can resume. This terminal assessment prevents the policy from learning maneuvers that complete the turn but leave the robot in an unrecoverable state.

Domain randomization is applied extensively to bridge the sim-to-real gap, with randomization over ground friction coefficients (0.4–1.2), body mass (±15%), center of mass offsets, joint damping, motor strength scaling, and observation noise injection. The training environment also randomizes the initial conditions of the robot to ensure the policy is robust to variations in the approach velocity and heading at the start of the maneuver. The authors found that the combination of recurrent architecture and domain randomization was critical for successful transfer, as feedforward policies with the same randomization failed to execute the maneuver on hardware.

Real-world deployment involves running the trained policy at 50 Hz on Cassie's onboard Intel NUC computer, with the GRU hidden state maintained across control steps. The maneuver is triggered by a command signal, after which the policy autonomously executes the four-step turn sequence and transitions back to steady-state walking. Hardware experiments demonstrate consistent execution of the 90-degree turn with heading errors within ±10 degrees of the target.

**Key Results & Numbers:**
- Successfully executed 90-degree turns in 4 steps on real Cassie hardware
- First sim-to-real demonstration of complex dynamic bipedal maneuvers
- Heading accuracy within ±10 degrees of target 90-degree turn
- Zero-shot transfer from MuJoCo simulation to real hardware
- GRU-based policy significantly outperforms feedforward baselines on hardware
- Maneuver completion rate >90% across multiple real-world trials
- Policy inference runs at 50 Hz with negligible computational overhead

**Relevance to Project A (Mini Cheetah):** LOW — The work focuses on bipedal-specific dynamic maneuvers (turning while walking on two legs) that do not directly transfer to quadruped locomotion. The biomechanical constraints and balance requirements are fundamentally different for a 12-DoF quadruped.

**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to the Cassie dynamic skill repertoire and the primitive-level actions in the hierarchical controller. The SRB-based reference generation and epilogue reward concepts could inform how individual motion primitives are trained, and the recurrent policy architecture provides a baseline comparison for the proposed transformer-based approach.

**What to Borrow / Implement:**
- Epilogue terminal reward formulation for training individual motion primitives that must transition smoothly back to steady-state locomotion
- SRB model-based reference trajectory generation as a lightweight planning mechanism for the Planner level
- GRU hidden state maintenance protocol for real-time deployment on Cassie hardware
- Domain randomization parameter ranges validated for Cassie sim-to-real transfer
- The concept of maneuver-specific policies as a template for the Primitives level of the hierarchy

**Limitations & Open Questions:**
- Each maneuver requires separate training, leading to scalability issues as the skill repertoire grows
- Limited to specific predefined maneuver types (90-degree turns); generalization to arbitrary turn angles not demonstrated
- SRB reference trajectory quality directly limits the achievable maneuver complexity
- No mechanism for online adaptation or learning from real-world experience
- Transition between maneuver policy and steady-state walking policy not deeply analyzed
- Real-world evaluation limited to controlled indoor environment on flat ground
---
