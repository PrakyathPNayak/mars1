# Robust Underactuated Point-Feet Bipedal Locomotion Using Deep Reinforcement Learning

**Authors:** Various
**Year:** 2025 | **Venue:** ASME Letters in Dynamic Systems and Control
**Links:** https://asmedigitalcollection.asme.org/lettersdynsys/article/5/4/041004/1219806

---

## Abstract Summary
This paper addresses the challenging problem of underactuated point-feet bipedal locomotion using deep RL. Point-feet robots lack ankle torques, making balance extremely difficult as the robot cannot generate corrective moments at the ground contact point. The work demonstrates that well-designed RL with appropriate reward shaping can achieve stable walking without ankle actuation, directly relevant to Cassie's point-foot design.

## Core Contributions
- Stable bipedal walking achieved on a point-feet robot with no ankle actuation, using deep RL alone
- Reward shaping methodology specifically designed for underactuated bipedal balance, addressing the unique challenges of point-feet contact
- Robustness to external perturbations (pushes, pulls) despite the lack of ankle-based recovery strategies
- Analysis of emergent balance strategies: the RL policy discovers hip-based and stepping-based balance recovery without explicit programming
- Comparison against full-actuation baselines showing that point-feet policies develop fundamentally different (and in some ways more robust) strategies
- Demonstration that domain randomization is especially critical for point-feet robots due to heightened sensitivity to model errors
- Insights into the minimal actuation requirements for stable bipedal locomotion

## Methodology Deep-Dive
Point-feet (or point-contact) bipedal robots present a fundamental control challenge. With flat feet, a biped can generate ankle torques to maintain balance — adjusting the center of pressure (CoP) within the support polygon. With point feet, the CoP is fixed at the contact point, eliminating this primary balance mechanism. The robot must rely entirely on hip-based balance strategies (moving the upper body to shift center of mass) and stepping strategies (placing the foot at the correct location to prevent falling).

The RL formulation uses a standard actor-critic architecture trained with PPO. The observation space includes joint positions, joint velocities, body orientation (roll, pitch, yaw), body angular velocity, body linear velocity, and a command velocity signal. The action space consists of target joint positions for all actuated joints (hips, knees — no ankles). The control frequency is set to match typical hardware rates (around 100-500 Hz), with PD controllers tracking the target positions.

The reward function is the paper's key contribution. Standard bipedal locomotion rewards (velocity tracking, energy minimization) are insufficient for point-feet robots because they don't explicitly encourage the specific balance strategies needed. The authors design a multi-component reward that includes: (1) velocity tracking with smooth velocity profiles to avoid jerky motions; (2) body orientation penalties that are more aggressive for pitch than roll, reflecting the primary fall direction; (3) foot placement rewards that encourage the swing foot to land in positions that support forward momentum; (4) angular momentum regulation that penalizes excessive body rotation; and (5) alive bonuses with progressive scaling that strongly incentivizes not falling.

Domain randomization is applied with particular emphasis on parameters that affect point-feet balance. Ground friction is critical — with point contact, even small friction variations dramatically change the available lateral forces. Mass distribution randomization is more important than for flat-foot robots because the center of mass location relative to the point contact directly determines stability margins. Motor strength randomization accounts for the policy's heavy reliance on fast hip actuation for balance corrections.

The training curriculum starts with standing balance (zero velocity command) and progressively introduces forward walking, turning, and perturbation recovery. The authors find that the standing balance phase is significantly longer for point-feet robots than for flat-feet equivalents, reflecting the greater difficulty of static balance without ankle torques. Only after robust standing balance is achieved does the curriculum advance to dynamic walking.

## Key Results & Numbers
- Stable forward walking at speeds up to 0.6 m/s on a point-feet biped (comparable to flat-feet baselines)
- Recovery from lateral pushes up to 40 N (60% of flat-feet recovery capacity, but using fundamentally different strategies)
- Zero ankle actuation — all balance achieved through hip and stepping strategies
- Domain randomization improves point-feet success rate by 45% (vs. 15-20% improvement for flat-feet robots)
- Emergent stepping strategies discovered by RL match analytical predictions from capture point theory
- Point-feet policies show faster step frequency (15-20% higher) as compensation for reduced per-step balance authority
- Training requires 2-3x more samples than flat-feet equivalents due to the more constrained solution space

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Low**
Quadrupeds have fundamentally different balance constraints than bipeds — four contact points provide inherent stability that makes ankle actuation less critical. The point-feet challenge is largely specific to bipedal locomotion. However, some insights may transfer to scenarios where Mini Cheetah lifts one or two legs (three-point or two-point contact), temporarily creating underactuated balance situations. The reward shaping methodology could inform how to handle reduced contact states.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Cassie has point-feet with very limited ankle actuation, making this paper directly applicable to its core balance challenge. The reward shaping methodology designed for point-feet balance should be incorporated into Project B's controller-level reward design. The finding that RL discovers hip-based and stepping-based balance strategies validates the potential for the learned controller to develop natural balance behaviors. The connection between emergent stepping strategies and capture point theory directly relates to Project B's Differentiable Capture Point module. The increased importance of domain randomization for point-feet robots highlights the need for extensive randomization in Cassie's training. The 2-3x sample complexity increase for point-feet training informs expectations for training time and motivates the use of sample-efficient methods (world models, model-based RL) in Project B's pipeline.

## What to Borrow / Implement
- Adopt the point-feet-specific reward components (foot placement, angular momentum regulation) for Project B's controller level
- Use the standing balance curriculum phase as a prerequisite before walking training for Cassie
- Validate that Project B's Differentiable Capture Point module produces stepping strategies consistent with this paper's emergent behaviors
- Apply the heightened domain randomization ranges (especially friction and mass distribution) to Cassie's training
- Use the 2-3x sample complexity finding to justify sample-efficient methods in Project B's architecture
- Analyze Cassie's learned policies for the hip-based vs. stepping-based balance strategies identified in this paper

## Limitations & Open Questions
- Point-feet walking speeds are lower than flat-feet equivalents; whether RL can close this gap is an open question
- Recovery from perturbations is reduced compared to flat-feet robots; safety implications for real-world deployment
- Limited to walking; point-feet running introduces flight phases where contact assumptions change entirely
- Does not address point-feet balance on uneven or compliant terrain (combining with compliant terrain paper would be valuable)
- How does Cassie's series elastic actuation (providing passive compliance) interact with point-feet balance strategies?
- Can the capture point-based stepping strategies be explicitly incorporated into the reward to accelerate training?
- What is the minimum sensor fidelity required for point-feet balance (IMU noise, joint encoder resolution)?
