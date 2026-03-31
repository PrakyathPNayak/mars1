# Towards Jumping Skill Learning by Target-Guided Policy for Quadruped Robots

**Authors:** (Machine Intelligence Research 2024)
**Year:** 2024 | **Venue:** Machine Intelligence Research (Springer)
**Links:** [Springer](https://link.springer.com/article/10.1007/s11633-023-1429-5)

---

## Abstract Summary
This paper proposes a target-guided policy framework for quadruped jumping that explicitly decomposes the jumping skill into distinct take-off and flight/landing phases, each managed by a separate SAC (Soft Actor-Critic) policy. The take-off policy uses curiosity-driven exploration to discover effective launch strategies, while the flight/landing policy focuses on mid-air body control and stable ground contact. A target-specification module parameterizes jumps by desired landing location, enabling precision jumping to specified targets.

The phase decomposition addresses a fundamental challenge in learning agile locomotion: the jump take-off requires explosive, high-energy actuation patterns that are drastically different from the controlled, compliant behaviors needed for landing. By training separate policies for each phase with tailored reward functions and exploration strategies, the method avoids the compromise that a single policy must make between these conflicting objectives.

The paper also introduces explicit recovery skills for failed jump attempts, recognizing that in real-world deployment, not every jump will succeed. The recovery policy is trained on a distribution of post-failure states (tumbled, off-balance, partially fallen) and learns to return the robot to a stable standing configuration. This recovery capability is essential for continuous autonomous operation where the robot must handle its own failures.

## Core Contributions
- Phase-decomposed jumping with separate SAC policies for take-off and flight/landing, enabling specialized optimization for each dynamic regime
- Curiosity-driven exploration (ICM — Intrinsic Curiosity Module) for the take-off policy to discover diverse and effective launch strategies in the absence of reference motions
- Target-guided parameterization enabling precision jumping to specified (x, y) landing coordinates with distance and direction control
- Explicit recovery policy trained on post-failure state distributions for autonomous failure handling
- Analysis of the interplay between take-off velocity vector and landing stability, informing reward design
- Demonstration on simulated Unitree A1 quadruped with analysis of jump distance, accuracy, and recovery success

## Methodology Deep-Dive
The framework consists of three policy modules operating sequentially: the take-off policy π_launch(a|s, g), the flight/landing policy π_land(a|s, g), and the recovery policy π_recover(a|s). The target g = (d_target, θ_target) specifies the desired landing distance and direction relative to the robot's current heading.

The take-off policy π_launch is trained using SAC with a combined reward: r_launch = r_extrinsic + α·r_ICM, where r_extrinsic measures how close the achieved launch velocity vector matches the ballistically optimal velocity for reaching the target, and r_ICM is the intrinsic curiosity reward from the ICM module. The ICM consists of a forward dynamics model f(s_t, a_t) → ŝ_{t+1} and an inverse dynamics model g(s_t, s_{t+1}) → â_t. The curiosity reward r_ICM = ||ŝ_{t+1} − s_{t+1}||² incentivizes the policy to visit states where the forward model has high prediction error, promoting exploration of novel take-off strategies. The ICM is critical because the take-off phase is short (0.1–0.3s) with a massive action space, making random exploration extremely unlikely to discover effective launches.

The ballistically optimal launch velocity is computed analytically: given target distance d and height h, the optimal launch angle θ* and speed v* satisfy the projectile motion equations (adjusted for the robot's CoM height). The extrinsic reward measures: r_extrinsic = exp(−λ₁||v_launch − v*||² − λ₂||θ_launch − θ*||²), rewarding close alignment between the achieved and optimal launch parameters.

The flight/landing policy π_land takes over when all feet leave the ground. Its reward prioritizes: (1) body orientation control — keeping the torso level during flight via r_orient = exp(−||roll, pitch||²); (2) leg preparation — extending legs to appropriate landing configuration via r_prep = exp(−||q_legs − q_landing||²); (3) impact absorption — upon ground contact, minimizing CoM velocity and maintaining upright posture via r_impact = exp(−||v_com||²)·exp(−||orientation_error||²). The landing policy uses compliance-aware actions: instead of position targets, it outputs desired joint impedance parameters (stiffness and damping) that modulate the PD controller during impact, enabling the legs to absorb landing forces.

Phase transitions are detected by a contact state machine: Standing → Crouching (preparation detected) → Airborne (all contacts lost) → Landing (first contact regained) → Recovery (if stability criterion not met within 0.5s). The recovery policy π_recover is trained on a curriculum of post-failure initial states sampled from a distribution of tumbled configurations, using a simple reward: r_recover = exp(−||s − s_standing||²) encouraging convergence to the standing pose.

Both take-off and landing policies are trained with SAC for its off-policy sample efficiency and maximum entropy exploration. The take-off policy uses a 4-layer MLP [256, 256, 128, 64] due to the short but critical nature of the launch, while the landing policy uses a 3-layer MLP [256, 256, 128] with the target distance encoded in the input.

## Key Results & Numbers
- Maximum jump distance: 65cm on simulated Unitree A1 (approximately 1.5× body length)
- Target precision: landing within 8cm of specified target for jumps up to 50cm distance
- Take-off policy with ICM discovered 3 distinct launch strategies: squat-and-push, rocking launch, and asymmetric push
- Recovery policy achieved 87% success rate from tumbled/fallen states
- Curiosity-driven exploration improved jump distance by 28% compared to SAC without ICM
- Phase-decomposed training converged 2.5× faster than single-policy end-to-end training
- Flight time: 0.15–0.35 seconds for 30–65cm jumps
- Compliance-aware landing reduced peak ground reaction forces by 40% compared to position-controlled landing

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The phase-decomposed jumping approach is highly relevant to Mini Cheetah's agile locomotion capability. Mini Cheetah's backdrivable actuators are well-suited for the compliance-aware landing strategy (impedance modulation during impact). The curiosity-driven take-off exploration could discover launch strategies specific to Mini Cheetah's mass distribution and joint limits. The target-guided parameterization enables precision jumping for obstacle negotiation. The recovery policy is particularly valuable — Mini Cheetah needs autonomous recovery from failed agile maneuvers for continuous outdoor operation. The phase decomposition can be extended to other agile skills (flipping, rapid acceleration) by identifying distinct dynamic regimes.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Low**
Jumping is not a primary objective for Cassie bipedal locomotion. However, the phase decomposition concept (separate policies for distinct dynamic regimes) could inform Cassie's gait transition design — e.g., separate policies for stance-to-swing transitions in walking. The recovery policy concept is transferable: Cassie needs balance recovery skills from perturbations. The curiosity-driven exploration strategy could complement DIAYN/DADS for discovering novel bipedal behaviors.

## What to Borrow / Implement
- Implement phase-decomposed jumping with separate take-off and landing SAC policies for Mini Cheetah, with contact-state-machine-based phase detection
- Use ICM (Intrinsic Curiosity Module) for take-off exploration to discover diverse launch strategies without reference trajectories
- Adopt compliance-aware landing actions that output impedance parameters (stiffness, damping) instead of position targets for Mini Cheetah's backdrivable actuators
- Train a dedicated recovery policy on a curriculum of failure states for robust autonomous operation
- Apply the target-guided parameterization (distance, direction) for precision obstacle jumping

## Limitations & Open Questions
- The phase decomposition requires manual identification of phase boundaries and transition criteria, which may not generalize to other agile skills without redesign
- Only demonstrated in simulation on Unitree A1; no real-hardware validation is presented, leaving sim-to-real transfer quality unknown
- The analytically computed ballistic reference assumes rigid-body projectile motion, ignoring the robot's ability to generate mid-air angular momentum through leg movements
- Recovery policy success rate of 87% means 13% of failures lead to irrecoverable states, which may be problematic for unsupervised deployment
