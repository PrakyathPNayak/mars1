---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/learning_locomotion_skills_cassie_iterative_design.md

**Title:** Learning Locomotion Skills for Cassie: Iterative Design and Sim-to-Real
**Authors:** Zhaoming Xie, Patrick Clary, Jeremy Dao, Pedro Morais, Jonathan Hurst, Michiel van de Panne
**Year:** 2020
**Venue:** CoRL 2020 (Conference on Robot Learning), PMLR Vol. 100
**arXiv / DOI:** arXiv:1903.09537

**Abstract Summary (2–3 sentences):**
This paper presents an iterative reinforcement learning design approach for training locomotion policies on the Cassie bipedal robot, introducing the DASS (Deterministic Action Stochastic State) tuple formulation for transferring knowledge between training iterations and facilitating sim-to-real transfer. The framework achieves variable-speed bipedal walking on real hardware without requiring dynamics randomization, relying instead on careful iterative reward design and curriculum learning. The work demonstrates that systematic iteration on the reward function and training procedure can be as effective as aggressive domain randomization for sim-to-real transfer.

**Core Contributions (bullet list, 4–7 items):**
- DASS (Deterministic Action Stochastic State) tuple formulation enabling transfer between RL training iterations
- Iterative reward design methodology that progressively refines locomotion behavior across multiple design cycles
- Demonstration that sim-to-real transfer is achievable without dynamics randomization through careful reward engineering
- Variable-speed walking controller (0–1.0 m/s) deployed on real Cassie hardware
- Systematic analysis of how reward function modifications propagate through training to affect real-world behavior
- Curriculum-based training approach that gradually increases task difficulty across iterations
- Detailed documentation of the iterative design process and lessons learned for bipedal RL practitioners

**Methodology Deep-Dive (3–5 paragraphs):**
The DASS (Deterministic Action Stochastic State) tuple formulation is the central technical contribution. In standard RL, experience tuples consist of (state, action, reward, next_state) samples. The DASS formulation separates the deterministic component (the policy's output action given the observation) from the stochastic component (the environment's state transitions). This separation allows experience collected under one policy or in one version of the simulator to be partially reused in subsequent training iterations. Specifically, when the reward function is modified between iterations, the DASS tuples can be re-labeled with the new reward without re-collecting trajectories, dramatically accelerating the iterative design loop. This is particularly valuable for bipedal locomotion where each training run requires substantial computation.

The iterative design process follows a structured cycle: (1) train a policy with the current reward function in simulation, (2) evaluate the policy on real Cassie hardware, (3) identify failure modes and undesirable behaviors through observation, (4) modify the reward function to address identified issues, and (5) retrain using DASS tuples from previous iterations as a warm start. The authors document multiple iterations of this cycle, showing how early policies exhibited foot-dragging behaviors that were corrected by adding foot clearance rewards, how energy-inefficient gaits were refined through explicit torque penalties, and how speed-tracking accuracy improved through progressive tightening of velocity reward tolerances. Each iteration builds on the previous one rather than starting from scratch, creating an efficient refinement pipeline.

Training uses PPO in the MuJoCo simulator with a Cassie model that includes the full rigid-body dynamics, spring-loaded legs (Cassie's series elastic actuators), and ground contact dynamics. Notably, the authors deliberately avoid dynamics randomization, instead relying on the hypothesis that a well-shaped reward function producing natural, efficient gaits will inherently be more robust to the sim-to-real gap than a policy trained with randomized dynamics that may learn to exploit unrealistic simulator behaviors. The observation space includes joint positions, joint velocities, pelvis orientation (from IMU), pelvis angular velocity, and a commanded velocity target. The action space consists of target joint positions for Cassie's 10 actuated joints.

The curriculum component introduces speed commands gradually during training. Early training phases focus on a single comfortable walking speed (approximately 0.5 m/s), allowing the policy to first learn stable walking mechanics. Subsequent phases expand the commanded speed range, eventually covering 0–1.0 m/s. This curriculum prevents the common failure mode where the policy attempts to learn multiple speeds simultaneously and converges to a poor compromise that works at no speed. The authors also employ a terrain curriculum that begins with perfectly flat ground and gradually introduces small perturbations, though the real-world deployment is limited to relatively flat surfaces.

Sim-to-real transfer is evaluated by directly deploying the trained policy on the Cassie hardware running the policy at 40 Hz on the onboard computer. The authors report that the iterative approach consistently produced policies that transferred better than single-shot training with aggressive domain randomization, suggesting that understanding and correcting specific failure modes through the iterative loop is more effective than brute-force robustification. The real-world evaluation covers multiple walking speeds, start-stop transitions, and gentle turning commands.

**Key Results & Numbers:**
- Variable-speed walking achieved on real Cassie from 0 to 1.0 m/s
- No dynamics randomization required for successful sim-to-real transfer
- Successful iterative improvement across 5+ design cycles
- DASS tuple reuse reduced training time by approximately 40% per iteration
- Policy runs at 40 Hz on Cassie's onboard computer
- Velocity tracking error <0.15 m/s across the commanded speed range
- Multiple hours of continuous walking demonstrated on hardware without failure

**Relevance to Project A (Mini Cheetah):** LOW — The work is Cassie-specific, and the iterative design process is tailored to bipedal locomotion challenges. While the general philosophy of iterative reward refinement applies broadly, the specific techniques and lessons are less transferable to quadruped systems.

**Relevance to Project B (Cassie HRL):** HIGH — Foundational work for Cassie RL that provides essential context for any Cassie-based project. The DASS tuple concept could inform warm-starting strategies when iterating on the hierarchical controller's reward functions. The iterative design lessons and documented failure modes provide practical guidance for developing the controller on Cassie hardware.

**What to Borrow / Implement:**
- DASS tuple formulation for accelerating reward iteration cycles during hierarchical controller development
- Iterative reward design methodology: train → deploy → observe → fix → retrain
- Curriculum strategy for speed range expansion during locomotion policy training
- Cassie-specific reward terms that proved essential (foot clearance, energy efficiency, symmetry)
- The insight that careful reward engineering can substitute for aggressive domain randomization

**Limitations & Open Questions:**
- Slow iterative process requiring multiple training-deployment cycles, each taking days to weeks
- Walking-only capability with no running, jumping, or other dynamic maneuvers
- Limited terrain variety—flat or near-flat surfaces only
- Maximum speed of 1.0 m/s is conservative compared to Cassie's mechanical capabilities
- DASS tuple reuse is limited to reward function changes; dynamics model changes invalidate prior data
- The approach does not scale well to multi-skill policies where reward interactions become complex
---
