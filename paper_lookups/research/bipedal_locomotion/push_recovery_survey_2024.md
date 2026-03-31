# A Comprehensive Survey of Push Recovery Techniques for Standing and Walking Humanoid Robots

**Authors:** (2024)
**Year:** 2024 | **Venue:** Al-Khwarizmi Engineering Journal
**Links:** [Al-Khwarizmi Engineering Journal](https://alkej.uobaghdad.edu.iq/index.php/alkej/article/view/949)

---

## Abstract Summary
This paper provides a comprehensive survey of push recovery techniques for humanoid robots, covering both standing and walking scenarios. The survey systematically categorizes recovery strategies into three fundamental approaches: ankle strategy (using ankle torques to maintain balance within the support polygon), hip strategy (using upper body angular momentum to reject disturbances), and stepping strategy (relocating the base of support to capture the falling motion). Each strategy is analyzed from both biomechanical and robotics perspectives, tracing the evolution from biological observations to engineering implementations.

The survey covers the full spectrum of push recovery methods, from classical model-based approaches to modern learning-based techniques. On the model-based side, the paper reviews Zero Moment Point (ZMP) control, Capture Point (CP) and Divergent Component of Motion (DCM) methods, Linear Inverted Pendulum Model (LIPM) control, and full-body optimization approaches. On the learning-based side, the survey covers reinforcement learning methods (PPO, SAC, DDPG), imitation learning from human demonstrations, and hybrid approaches that combine model-based structure with learned components.

The paper concludes with a comparative analysis of classical vs. learning-based methods across key metrics (disturbance rejection capability, computational requirements, generalization, real-world deployment readiness) and identifies future research directions including multi-strategy coordination, terrain-aware recovery, and safe RL for balance control.

## Core Contributions
- **Comprehensive taxonomy** of push recovery strategies organized by mechanism (ankle, hip, stepping) and methodology (model-based, learning-based, hybrid)
- **Historical evolution** tracing push recovery techniques from early ZMP-based methods through capture point theory to modern RL approaches, showing the field's trajectory
- **Biomechanical grounding** connecting each robotic strategy to its biological counterpart in human balance control research
- **Detailed comparison** of model-based vs. learning-based approaches with quantitative metrics where available (disturbance magnitude, recovery time, computational cost)
- **Analysis of reduced-order models** (LIPM, SLIP, Cart-Table) used in push recovery, with their assumptions, limitations, and applicability domains
- **Future research roadmap** identifying key open problems: multi-strategy blending, terrain-aware recovery, safe RL integration, and sim-to-real transfer for balance control

## Methodology Deep-Dive
The survey organizes push recovery techniques along two axes: the recovery mechanism (how the robot recovers) and the control methodology (how the recovery is computed). The three recovery mechanisms form a hierarchy of increasing disturbance rejection capability.

**Ankle Strategy**: The simplest recovery mechanism, the ankle strategy maintains balance by applying torques at the ankle joint to shift the Center of Pressure (CoP) within the support polygon. This is effective only for small disturbances where the required CoP shift stays within the foot boundary. The ZMP (Zero Moment Point) framework formalizes this: the ZMP must remain within the support polygon for dynamic balance. Classical ZMP control uses preview control or MPC to track a desired ZMP trajectory while maintaining CoM height. The limitation is clear: the maximum recoverable disturbance is bounded by the foot size and ankle torque capacity, typically handling pushes up to 20-40N on standard humanoids.

**Hip Strategy**: For moderate disturbances exceeding ankle strategy capacity, the hip strategy generates angular momentum by rapidly rotating the upper body (trunk, arms). This centroidal momentum change creates a ground reaction force that shifts the CoP toward the support polygon boundary, effectively extending the ankle strategy's range. The hip strategy is transient — it provides temporary balance correction while the ankle strategy converges to steady-state. Mathematical modeling uses the Angular Momentum Pendulum Model (AMPM) that extends LIPM with centroidal angular momentum. The combined ankle+hip strategy handles pushes up to 60-100N on typical humanoids.

**Stepping Strategy**: For large disturbances, stepping is necessary — the robot must relocate its support polygon to capture the diverging motion. The Capture Point (CP) theory provides the foundational framework: the CP is the point where the robot must step to halt its CoM motion. The DCM (Divergent Component of Motion) extends CP to continuous walking by tracking the unstable component of LIPM dynamics. The survey details the evolution from simple CP heuristics (step to the capture point) through optimized stepping (MPC over step placement and timing) to RL-based reactive stepping. The stepping strategy handles the largest disturbances (100-200N+) but requires the most complex control and the longest response time.

The learning-based section surveys RL applications to push recovery, categorizing approaches by: (1) what is learned (recovery policy, value function, dynamics model), (2) the RL algorithm (on-policy: PPO, TRPO; off-policy: SAC, DDPG, TD3), and (3) the training methodology (end-to-end RL, residual RL augmenting model-based control, RL for parameter tuning). Notable trends include: the shift from discrete to continuous action spaces, increasing use of sim-to-real transfer with domain randomization, and the emergence of hybrid approaches that use RL to enhance model-based controllers rather than replace them entirely.

The survey identifies that hybrid methods — combining model-based structure with learned components — currently offer the best balance of performance, reliability, and generalization. For example, using RL to learn a residual policy on top of a CP-based stepping controller preserves the model-based safety properties while learning to handle scenarios that the model doesn't capture well.

## Key Results & Numbers
- **Ankle strategy range**: recovers pushes up to **20-40N** on typical humanoids (limited by foot size and ankle torque)
- **Hip strategy extension**: increases recoverable push to **60-100N** when combined with ankle strategy
- **Stepping strategy range**: recovers pushes up to **100-200N+** depending on stepping speed and kinematic workspace
- **CP-based methods**: achieve recovery within **150-300ms** step initiation time, with foot placement accuracy of **2-5cm**
- **RL-based methods**: achieve **20-40% larger** recoverable disturbances compared to model-based CP control in simulation studies
- **Sim-to-real gap**: typical **30-50% performance degradation** when transferring RL push recovery policies from simulation to hardware without domain randomization; reduced to **5-15%** with domain randomization
- **Computational requirements**: model-based CP control runs at **>500Hz**; RL policy inference at **>1kHz**; full-body optimization at **100-200Hz**
- **Real-world deployments**: only **~15%** of surveyed RL-based methods demonstrate real-world push recovery (as of 2024)
- **Multi-strategy coordination**: papers combining ankle+hip+stepping strategies show **50-70%** improvement over single-strategy methods

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
While the survey focuses on bipedal/humanoid robots, the general balance control concepts are transferable to quadruped locomotion. The ankle strategy maps to the Mini Cheetah's stance leg force control, the hip strategy maps to body attitude adjustment, and the stepping strategy maps to foothold replanning during dynamic gaits. The comparison of model-based vs. learning-based approaches provides useful context for choosing between PPO-based end-to-end learning and hybrid approaches for Mini Cheetah's perturbation handling.

The survey's finding that hybrid methods (model-based structure + learned residuals) offer the best performance-reliability trade-off is relevant for the Mini Cheetah project's methodology selection. The domain randomization analysis provides validated ranges and techniques for sim-to-real transfer.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This survey is critical to Project B as it provides the comprehensive theoretical background for Cassie's balance control and push recovery architecture. The three-strategy hierarchy (ankle→hip→stepping) maps directly to Cassie's Safety level design: the LCBF should incorporate all three strategies with smooth blending based on disturbance magnitude. The detailed coverage of ZMP, CP, and LIPM theory provides the mathematical foundations for Cassie's Differentiable Capture Point module.

The survey's analysis of hybrid model-based + RL approaches directly validates Cassie's architectural choice: using model-based capture point computation as structure (in the Controller and Safety levels) while learning residual policies and parameters through RL. The DCM (Divergent Component of Motion) framework discussed in the survey extends basic capture point theory to continuous walking, which is more relevant than static capture point for Cassie's dynamic locomotion. The finding that multi-strategy coordination improves recovery by 50-70% motivates Cassie's multi-strategy Safety level that integrates ankle torque modulation, angular momentum control, and reactive stepping.

## What to Borrow / Implement
- **Multi-strategy balance recovery** — implement ankle, hip, and stepping strategies as a coordinated recovery hierarchy in Cassie's Safety level, with smooth blending based on disturbance magnitude estimation
- **DCM (Divergent Component of Motion) tracking** — extend capture point to DCM for continuous walking balance control in Cassie's Controller level, providing stable walking reference trajectories
- **Hybrid model-based + RL architecture** — validate the design choice of using model-based CP structure with learned residual policies, following the survey's finding that hybrid approaches outperform pure model-based or pure RL
- **Recovery strategy selection criteria** — implement disturbance magnitude thresholds for transitioning between ankle/hip/stepping strategies, calibrated to Cassie's specific morphology and actuation limits
- **Sim-to-real transfer best practices** — apply the survey's domain randomization guidelines (mass ±20%, friction ±30%, delay randomization) for Cassie's training pipeline

## Limitations & Open Questions
- **Survey scope** — focuses primarily on humanoid robots; application to bipedal robots with different morphologies (like Cassie with spring-loaded legs) requires adaptation of the surveyed methods
- **Limited quantitative comparison** — many surveyed papers use different robots, simulators, and evaluation metrics, making direct quantitative comparison difficult
- **Terrain-aware recovery gap** — the survey identifies terrain-aware push recovery as an open problem; most methods assume flat ground, which is insufficient for real-world deployment on varied terrain
- **Safe RL integration** — while the survey identifies safe RL as a future direction, concrete methodologies for combining CBF-based safety with RL-based push recovery are still nascent and underexplored
