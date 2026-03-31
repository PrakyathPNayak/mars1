# Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors

**Authors:** ETH Zurich Robotic Systems Lab (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2508.19953)

---

## Abstract Summary
D3 (Divide, Discover, Deploy) addresses fundamental limitations of vanilla DIAYN and DADS for real-robot skill discovery by introducing factorized state spaces, symmetry priors, and style regularization. The core observation is that applying a single MI-based intrinsic reward to the full state vector leads to skill degeneracy — skills that exploit easy-to-distinguish but physically meaningless state dimensions (e.g., internal joint configurations) rather than producing diverse whole-body locomotion behaviors. D3 factors the state space into semantically meaningful groups and applies separate MI objectives to each factor.

The method incorporates morphological symmetry priors that enforce left-right consistency in discovered skills, preventing asymmetric gaits that are harmful on real hardware. Style regularization biases the learned skills toward physically plausible, energy-efficient motions without requiring reference trajectories. These structural inductive biases dramatically improve skill quality and enable zero-shot transfer from simulation to a real quadruped (ANYmal) without any sim-to-real fine-tuning.

D3 represents a significant step toward practical unsupervised skill discovery for legged robots, demonstrating that the gap between DIAYN/DADS theory and real-world deployment can be bridged through careful state-space factorization and physics-informed priors. The discovered skills include forward/backward locomotion, lateral shuffling, turning, and speed variations — all transferring directly to hardware.

## Core Contributions
- Proposed factorized state-space MI objectives that apply different intrinsic rewards to different state groups (base velocity, foot contacts, joint configurations) to prevent skill degeneracy
- Introduced morphological symmetry priors enforcing bilateral consistency in skill behaviors, critical for stable real-robot gaits
- Developed style regularization losses that bias skills toward energy-efficient, smooth motions without reference trajectories
- Demonstrated zero-shot sim-to-real transfer of discovered skills on ANYmal quadruped without any domain adaptation
- Provided systematic analysis of skill degeneracy modes in vanilla DIAYN/DADS and how factorization addresses each
- Achieved higher skill diversity and quality metrics than baseline DIAYN/DADS on both simulated and real quadrupeds
- Released a practical framework for deploying unsupervised skill discovery on real legged robots

## Methodology Deep-Dive
The factorization approach partitions the robot's state vector s into K factors: s = [s^1, s^2, ..., s^K], where typical factors include base linear velocity (s^vel), base angular velocity (s^ω), foot contact patterns (s^contact), and joint position deviations (s^joints). Each factor receives its own MI-based intrinsic reward with independent discriminators: r_total = Σ_k α_k · log q_φ^k(z | s^k), where α_k weights the importance of each factor. This prevents the discriminator from "cheating" by distinguishing skills based on a single easy dimension (like base height) while ignoring locomotion-relevant features.

The symmetry prior leverages the bilateral morphology of quadrupeds. For a robot with left-right symmetry, the method enforces that skill z and its mirrored version z̄ produce mirrored state trajectories: π(a_L | s, z) ≈ π(a_R | mirror(s), z̄) and vice versa. This is implemented as an auxiliary loss L_sym = E[||π(a|s,z) − mirror(π(a|mirror(s), z̄))||²] added to the policy optimization. The mirror operation swaps left/right joint positions, velocities, and contact states. This prior eliminates asymmetric gaits that, while potentially maximizing MI, would cause instability and wear on real hardware.

Style regularization introduces soft constraints on skill execution quality without prescribing specific motions. The style loss includes: (1) energy penalty L_energy = E[||τ||²] penalizing high joint torques, (2) smoothness penalty L_smooth = E[||a_t − a_{t-1}||²] penalizing jerky actions, (3) foot clearance reward encouraging adequate ground clearance during swing phase, and (4) contact timing regularity encouraging periodic gait patterns. These are combined as L_style = β₁L_energy + β₂L_smooth + β₃L_clearance + β₄L_contact, with coefficients tuned to bias toward natural motions without overly constraining skill diversity.

The training pipeline proceeds in three phases: (1) Divide — define state factors and discriminator architectures based on robot morphology; (2) Discover — train the skill-conditioned policy with factorized MI rewards, symmetry losses, and style regularization in simulation using Isaac Gym for parallel training across thousands of environments; (3) Deploy — directly transfer learned policies to real hardware using only proprioceptive observations, with no domain randomization required beyond standard actuator modeling.

Domain randomization during the Discover phase includes mass variations (±15%), friction coefficients (0.3–1.2), motor strength scaling (±10%), and observation noise injection. The policy architecture uses a 3-layer MLP with [512, 256, 128] hidden units and ELU activations, with z concatenated to the input layer.

## Key Results & Numbers
- Discovered 12 distinct locomotion skills on ANYmal with 100% zero-shot sim-to-real transfer success rate
- Skill diversity score (measured by trajectory mutual information) was 2.3× higher than vanilla DIAYN and 1.6× higher than DADS
- Symmetry prior reduced left-right asymmetry in gait by 78% compared to unconstrained training
- Energy efficiency of D3 skills was 35% better than DIAYN skills and within 15% of hand-tuned locomotion controllers
- Real-robot deployment sustained stable locomotion for >10 minutes across all discovered skills without falls
- Style-regularized skills showed 4× lower joint velocity variance (smoother motions) than unregularized baselines
- Training time: ~4 hours on a single NVIDIA A100 GPU using Isaac Gym with 4096 parallel environments

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
D3 is directly applicable to Mini Cheetah skill discovery. The factorized state-space approach solves the skill degeneracy problem that would plague vanilla DIAYN on Mini Cheetah's 30+ dimensional state space. The symmetry priors are immediately transferable since Mini Cheetah has bilateral symmetry. The zero-shot sim-to-real transfer pipeline provides a practical deployment roadmap. Specifically, factoring Mini Cheetah's state into base velocity, foot contacts, and joint positions — with separate discriminators — should yield a diverse, physically plausible gait repertoire including trotting, bounding, and turning skills. The style regularization ensures discovered gaits are safe for hardware execution.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
D3 directly addresses skill degeneracy issues in DIAYN/DADS that are critical for Project B's Primitives level on Cassie. The factorization approach should be adapted for Cassie's bipedal morphology — factoring into base velocity, step frequency, foot placement, and upper-body orientation. The symmetry prior maps to Cassie's left-right leg symmetry, enforcing balanced walking gaits. The style regularization biases are especially important for bipedal robots where energy-inefficient or asymmetric skills can cause falls. Incorporating D3's factorization into the existing DIAYN/DADS Primitives pipeline should significantly improve the quality and diversity of discovered bipedal locomotion skills.

## What to Borrow / Implement
- Factor Mini Cheetah state into [base_vel, base_angular_vel, foot_contacts, joint_pos_deviation] with separate discriminators for each factor and tunable weights α_k
- Implement the bilateral symmetry loss L_sym by defining the mirror operation for both Mini Cheetah (4-leg symmetry) and Cassie (2-leg symmetry) morphologies
- Add style regularization (torque penalty, action smoothness, foot clearance) to the DIAYN/DADS training loop in both projects
- Use Isaac Gym with 4096+ parallel environments for efficient skill discovery training
- Apply the three-phase Divide-Discover-Deploy pipeline for systematic real-robot skill transfer

## Limitations & Open Questions
- The factorization design requires domain expertise about which state dimensions to group — no automatic factorization discovery
- Symmetry priors assume bilateral morphology; would need adaptation for asymmetric robots or asymmetric terrains (slopes, stairs)
- Style regularization coefficients (β₁–β₄) require tuning and may overly constrain skill diversity if set too aggressively
- Zero-shot transfer demonstrated on ANYmal with high-quality actuators; transfer to Mini Cheetah's lower-cost motors may require additional domain randomization
