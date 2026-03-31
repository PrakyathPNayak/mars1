# Partition-Aware Stability Control for Humanoid Robot Push Recovery With Stepping Strategy

**Authors:** (ASME 2024)
**Year:** 2024 | **Venue:** ASME Journal of Mechanisms and Robotics
**Links:** [ASME Digital Collection](https://asmedigitalcollection.asme.org/mechanismsrobotics/article/16/1/011005/1159763)

---

## Abstract Summary
This paper introduces a partition-aware stability control framework for humanoid robot push recovery using stepping strategies. The core contribution is the formalization of n-step capturability regions — the set of states from which the robot can recover balance within n steps. By partitioning the state space into balanced, 1-step capturable, 2-step capturable, and non-capturable regions, the framework provides a systematic method for determining the minimum number of recovery steps needed and optimizing each step's location and timing.

Unlike prior approaches that use a single capture point for stepping decisions, this framework accounts for the full-body state including swing leg dynamics, actuation constraints, and kinematic reachability when computing capturability. The partition boundaries are computed offline using the LIPM with realistic constraint models and stored as lookup tables for real-time use. During operation, the robot's state is classified into the appropriate partition, and the corresponding recovery strategy is executed. States in the balanced region require no stepping, 1-step capturable states trigger a single recovery step, and multi-step capturable states initiate sequential recovery stepping.

Experimental validation on a simulated humanoid demonstrates that partition-aware control recovers from significantly larger disturbances than single-step capture point methods, by enabling multi-step recovery strategies that exploit the robot's full kinematic workspace.

## Core Contributions
- **N-step capturability regions** — formal definition and computation of state-space partitions based on the number of steps required for balance recovery, extending single-step capture point theory
- **Full-body state consideration** in capturability computation, including swing leg position/velocity, hip torque limits, and foot placement reachability constraints
- **Offline partition computation** with online lookup enabling real-time recovery strategy selection without solving complex optimization problems during execution
- **Step location and timing co-optimization** within each capturability region, maximizing the post-recovery stability margin
- **Recovery zone classification** — clear distinction between balanced (no action needed), recovery (stepping required), and non-capturable (falling unavoidable) zones with quantified boundaries
- **Comparison framework** demonstrating that n-step capturability significantly expands the recoverable state space compared to 1-step capture point methods

## Methodology Deep-Dive
The capturability analysis begins with the Linear Inverted Pendulum Model (LIPM) state, defined as (x, ẋ) — the CoM position and velocity in the horizontal plane. The Capture Point ξ = x + ẋ/ω₀ characterizes the 1-step capturability boundary: if ξ lies within the reachable foot placement region, the robot can recover in one step. This paper extends this analysis to multiple steps. For n-step capturability, the question becomes: does there exist a sequence of n foot placements, each within the kinematic reachability set, such that the robot's state after n steps lies in the balanced region?

The computation proceeds backward from the balanced region (0-step capturable set). The 0-step set is the region where the CP lies within the current support polygon — the robot is inherently stable without stepping. The 1-step capturable set is computed by finding all states from which a single optimal foot placement (within the reachability constraint) brings the state into the 0-step set. This computation uses the LIPM dynamics to propagate the state forward through the step (single support → double support transition) and checks whether any feasible foot placement results in a 0-step capturable state. The 2-step capturable set similarly finds states recoverable by first stepping into the 1-step set.

The critical extension over standard capture point theory is the inclusion of swing leg dynamics and actuation constraints. The standard capture point assumes instantaneous foot placement, but physical stepping requires time for leg swing, during which the robot's state continues to evolve. This paper models the swing phase explicitly: the swing leg has finite velocity limits and acceleration constraints, creating a time-dependent reachability set for foot placement. The longer the robot waits to step, the farther the CP drifts (potentially outside the support polygon), but the larger the reachable foot placement set becomes. The optimal step timing balances these competing effects.

Actuation constraints are modeled as maximum hip torque and knee torque limits, which restrict the achievable step length and the rate of step execution. These constraints are particularly important for large recovery steps where the robot approaches its kinematic limits. The constraint model is derived from the robot's joint torque specifications and validated against full-body dynamics simulation.

The partition boundaries are computed offline using a grid-based backward reachability analysis. The CoM state space is discretized (typically 200×200 grid covering ±1m in position and ±2m/s in velocity), and each grid cell is classified based on n-step capturability. Boundary refinement uses bisection to achieve sub-centimeter accuracy near critical boundaries. The computed partitions are stored as lookup tables (approximately 2MB of data) for real-time access.

During online operation, the controller pipeline is: (1) estimate current CoM state (x, ẋ) from joint encoders and IMU, (2) look up the state in the precomputed partition table, (3) if in the balanced region, maintain current stance, (4) if in the n-step capturable region, execute the precomputed optimal n-step recovery sequence, (5) if in the non-capturable region, attempt maximum-effort recovery and trigger fall protection.

## Key Results & Numbers
- **Recovery capability expansion**: 2-step capturability recovers from **40% larger disturbances** (measured by impulse magnitude) than 1-step capture point methods
- **3-step capturability** extends recovery further by **15%** beyond 2-step, with diminishing returns for n>3
- **Swing leg dynamics impact**: including swing leg constraints reduces the 1-step capturable set by **18%** compared to the idealized (instantaneous placement) assumption
- **Actuation constraint impact**: torque limits reduce the reachable foot placement set by **22%** for maximum-length recovery steps
- **Optimal step timing**: the balance between CP drift and reachability expansion yields an optimal step timing of **0.15-0.25 seconds** after disturbance detection for most initial states
- **Partition computation time**: offline computation takes **~4 hours** for the full 2D state space discretization on a standard workstation
- **Online lookup time**: **<0.1ms** per control cycle for state classification and recovery strategy retrieval
- **Comparison with DCM-based methods**: partition-aware control matches or exceeds DCM (Divergent Component of Motion) control on all test scenarios while providing explicit capturability guarantees

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
The stepping strategy framework is primarily designed for bipedal robots where discrete foot placement decisions critically determine balance recovery. For the Mini Cheetah quadruped, the wider support polygon and four-legged stance provide inherently greater stability, making n-step capturability less critical. However, the concept of state-space partitioning into stability regions could be adapted for quadruped recovery analysis, particularly during dynamic gaits where only two legs are in contact (trotting).

The offline computation of stability boundaries could provide safety constraints for Mini Cheetah RL training, defining regions of state space where the policy should avoid entering.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critical to Project B as it directly informs Cassie's Differentiable Capture Point module and Safety level design. The n-step capturability formalization provides the theoretical foundation for Cassie's balance recovery strategy selection. The partition-aware approach — classifying states into balanced/recovery/non-capturable zones — maps directly to Cassie's Safety level intervention logic: the LCBF (Learned CBF) needs to identify when the robot's state crosses partition boundaries and trigger appropriate recovery actions.

The inclusion of swing leg dynamics and actuation constraints in capturability computation is essential for Cassie, whose leg dynamics are more complex than idealized LIPM assumptions. The step timing optimization (balancing CP drift against reachability expansion) should be incorporated into Cassie's capture point module. The precomputed lookup tables could serve as a fast safety check in the CBF-QP framework, providing hard capturability constraints that the QP must satisfy. Alternatively, a neural network approximation of the partition boundaries could enable differentiable capturability assessment for end-to-end learning.

## What to Borrow / Implement
- **N-step capturability computation** — implement backward reachability analysis to compute capturability regions for Cassie's specific kinematics and actuation limits
- **State-space partitioning for Safety level** — use capturability partitions as safety boundaries in Cassie's LCBF, triggering recovery interventions at partition boundaries
- **Swing leg dynamics model** — incorporate realistic swing leg constraints into capture point computation for more accurate balance assessment
- **Neural network partition approximation** — train a neural network to approximate the precomputed partition boundaries for differentiable safety assessment in the CBF-QP framework
- **Multi-step recovery sequencing** — implement sequential recovery step planning for Cassie's Controller level when single-step recovery is insufficient

## Limitations & Open Questions
- **2D analysis only** — the partition computation is performed in the sagittal plane; extension to 3D (lateral stepping, diagonal recovery) significantly increases computational complexity
- **LIPM assumptions persist** — despite including swing leg dynamics, the CoM dynamics still use the simplified pendulum model; Cassie's actual dynamics include significant non-linearities
- **Static environment assumption** — the capturability regions assume fixed terrain; on uneven surfaces, the reachable set and capturability boundaries change dynamically
- **Offline computation scalability** — extending to 3D or including additional state variables (angular momentum, trunk orientation) makes the grid-based computation intractable; approximate methods needed
