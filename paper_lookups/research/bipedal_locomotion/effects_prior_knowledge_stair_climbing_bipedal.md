# Effects of Prior Knowledge for Stair Climbing of Bipedal Robots Based on Reinforcement Learning

**Authors:** Various
**Year:** 2024 | **Venue:** IEEE (IROS Workshop 2024)
**Links:** https://ieeexplore.ieee.org/document/10715938

---

## Abstract Summary
This paper investigates how incorporating prior knowledge—specifically terrain geometry and stair parameters—affects bipedal robot stair climbing performance when trained with reinforcement learning. The study demonstrates that providing explicit stair information (step height, step depth, number of steps) in the observation space significantly stabilizes learning and improves dynamic balance during stair ascent and descent. The approach is evaluated across varied stair heights, showing consistent improvements over blind policies.

## Core Contributions
- Systematically evaluates the effect of prior terrain knowledge on bipedal stair climbing RL
- Demonstrates that explicit stair parameters in observations stabilize training and improve performance
- Shows that dynamic balance maintenance improves significantly with prior knowledge integration
- Evaluates across varied stair heights (10-20cm) and configurations
- Compares blind policies, partial-knowledge policies, and full-knowledge policies
- Provides guidelines for what terrain information matters most for bipedal stair climbing
- Analyzes the interaction between prior knowledge and reward design for stair locomotion

## Methodology Deep-Dive
The study uses a controlled experimental design comparing three policy types. The "blind" policy receives only standard proprioceptive observations (joint angles, velocities, IMU, body pose) with no terrain information. The "partial-knowledge" policy receives terrain type classification (flat/stairs) and basic geometry (upcoming surface height relative to current foot position). The "full-knowledge" policy receives detailed stair parameters: step height, step depth, number of remaining ascending/descending steps, distance to first step, and stair inclination angle.

Training uses PPO in a physics simulator with a bipedal robot model. The training environments include flat ground and staircases with randomized parameters: step height (10-20cm), step depth (20-35cm), number of steps (3-10), and stair width. Domain randomization covers robot mass (±10%), motor strength (±15%), and ground friction (0.5-1.0). Each policy type is trained with the same hyperparameters and reward function, isolating the effect of observation content.

The reward function includes velocity tracking, energy efficiency, body orientation stability, foot clearance, and a stair-specific component. The stair-specific reward encourages: proper foot placement on stair surfaces (center of step rather than edge), appropriate body lean during ascent/descent, and smooth step-to-step transitions. Importantly, the stair-specific reward components are the same across all three policy types—only the observations differ—ensuring a fair comparison.

Dynamic balance analysis is a key contribution. The paper measures several balance metrics during stair climbing: Zero Moment Point (ZMP) deviation from support polygon center, Center of Mass (CoM) acceleration variance, ankle torque magnitude, and recovery step frequency after perturbations. These metrics quantify not just whether the robot climbs successfully but how stably it does so. The prior knowledge policies show lower ZMP deviation and CoM acceleration variance, indicating more controlled, stable climbing.

The interaction between prior knowledge and curriculum learning is also explored. When curriculum learning is used (starting with low stairs and gradually increasing height), the gap between blind and full-knowledge policies narrows for easy stairs but widens for challenging stairs (height >16cm). This suggests that prior knowledge becomes more critical as difficulty increases, and curriculum learning partially but not fully compensates for lack of terrain information.

## Key Results & Numbers
- Full-knowledge policy: 92% success rate on varied staircases (10-20cm step height)
- Partial-knowledge policy: 78% success rate on the same staircase distribution
- Blind policy: 55% success rate, with most failures on stairs >15cm
- ZMP deviation reduced by 35% with full prior knowledge vs. blind
- CoM acceleration variance reduced by 40% with full knowledge
- Training convergence: full-knowledge converges in 50% fewer timesteps than blind
- Step height is the most important prior knowledge variable (ablation shows 60% of the improvement)
- Step depth information provides additional 15% improvement beyond height alone
- Dynamic balance maintained during descent (harder than ascent) with full knowledge
- Recovery from 30N lateral perturbations: 90% with full knowledge vs. 60% blind

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Low**
This paper focuses on bipedal stair climbing, which has fundamentally different dynamics from quadruped locomotion. Quadrupeds have inherently more stable stair climbing due to multi-point support. However, the general principle that prior terrain knowledge improves RL performance applies to Mini Cheetah as well. The finding that step height is the most critical parameter could inform observation design for quadruped stair climbing. The curriculum learning interaction analysis is relevant to Mini Cheetah's curriculum learning pipeline.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This paper is directly applicable to Cassie's stair climbing capabilities. The finding that explicit stair parameters dramatically improve bipedal climbing performance should inform the Controller-level observation design. The CPTE (Contrastive Pre-trained Terrain Encoder) should be designed to extract stair-specific features (step height, depth, count) as the paper identifies these as critical. The dynamic balance analysis provides metrics (ZMP, CoM variance) for evaluating Cassie's stair climbing quality. The interaction with curriculum learning informs the Adversarial Curriculum design—harder stairs require more explicit terrain information. The Safety layer (LCBF) can use the balance metrics to define safety constraints during stair climbing. The Neural ODE Gait Phase should adapt phase timing based on stair parameters, as the paper shows gait timing significantly affects climbing stability.

## What to Borrow / Implement
- Include explicit stair parameters (height, depth, remaining steps) in Cassie's Controller observations
- Design CPTE to extract the stair features identified as most critical: step height > step depth > count
- Use ZMP deviation and CoM variance as evaluation metrics for Cassie stair climbing
- Inform LCBF safety constraints with the dynamic balance metrics from this paper
- Apply the curriculum learning progression (easy stairs → hard stairs) with prior knowledge in the Adversarial Curriculum
- Adapt Neural ODE Gait Phase timing based on detected stair parameters
- Consider partial-knowledge as a fallback when full terrain sensing is unavailable

## Limitations & Open Questions
- Evaluation limited to straight staircases; spiral or curved stairs not addressed
- Prior knowledge assumes accurate terrain sensing—sensor errors could degrade performance
- No evaluation of transition from flat to stairs (only steady-state stair climbing)
- Bipedal model used may differ significantly from Cassie's specific kinematics
- Open question: How does prior knowledge interact with adversarial training and domain randomization?
- No analysis of stair climbing speed; all experiments at conservative slow speeds
- Descending stairs with varied/unknown step heights is only briefly addressed
- Real-hardware validation not included; all results are simulation-only
