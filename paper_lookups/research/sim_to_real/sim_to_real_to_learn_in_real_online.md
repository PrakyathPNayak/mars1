# From Sim-to-Real to Learn-in-Real: Real-World Online Learning for Legged Locomotion

**Authors:** Various
**Year:** 2025 | **Venue:** Springer (ICIRA 2025)
**Links:** https://link.springer.com/chapter/10.1007/978-981-95-2098-5_44

---

## Abstract Summary
This paper proposes a "Learn-in-Real" paradigm where legged robots continue broad online learning in the real world after initial pre-training in simulation. Rather than treating sim-to-real transfer as a one-shot deployment step, the approach addresses the last-mile reality gap through continuous online adaptation. The framework combines simulation pre-training with structured real-world fine-tuning to achieve robust locomotion that adapts to conditions impossible to fully simulate.

## Core Contributions
- Proposes the "Learn-in-Real" paradigm extending beyond traditional sim-to-real transfer
- Addresses the last-mile reality gap through online real-world adaptation
- Combines simulation pre-training with structured real-world fine-tuning
- Demonstrates robust locomotion on real hardware after online learning
- Identifies key challenges in safe online learning for legged robots
- Provides a framework for continuous improvement beyond initial sim-to-real deployment
- Analyzes the trade-offs between exploration and safety during real-world learning

## Methodology Deep-Dive
The paper identifies a fundamental limitation of standard sim-to-real: no matter how much domain randomization is applied, there remain aspects of real-world dynamics that cannot be captured in simulation. These include complex ground contact dynamics, actuator thermal effects, cable interference, sensor noise patterns, and wear-induced parameter drift. The "Learn-in-Real" paradigm treats the sim-trained policy as an initialization rather than a final product.

The pre-training phase follows standard practice: PPO training in a physics simulator (MuJoCo/Isaac Gym) with domain randomization over masses, friction, motor parameters, and sensor noise. The key difference is that the pre-training explicitly prepares the policy for continued learning by maintaining exploration capacity. This is achieved through entropy regularization that prevents premature convergence to a narrow behavior mode, and by training an ensemble of value functions that provide uncertainty estimates for guiding real-world exploration.

The online learning phase operates on the real robot with several safety mechanisms. A learned safety critic estimates the probability of catastrophic failure (falling, joint limit violations) for candidate actions and vetoes dangerous exploration. The policy update uses a conservative variant of PPO with smaller step sizes and trust region constraints tighter than in simulation. Real-world data is collected in short episodes with human-supervised reset when necessary. The system interleaves exploitation (using the current best policy) with exploration (trying policy variations) in a structured manner.

The adaptation mechanism targets specific sim-to-real gaps. Rather than fine-tuning all parameters, the system identifies which aspects of locomotion are underperforming (e.g., specific terrain types, velocity ranges, turning) and focuses exploration on those modes. This is achieved through a performance monitoring system that tracks reward components across behavior modes and allocates exploration budget to underperforming areas.

Real-world data is combined with simulation data in a mixed replay buffer. This prevents catastrophic forgetting of simulation-learned behaviors while allowing adaptation to real-world conditions. The mixing ratio is adapted over time, starting with heavy simulation replay and gradually increasing the real-world data proportion as more real experience is collected.

## Key Results & Numbers
- Closes 30-50% of the residual sim-to-real performance gap through online learning
- Robust locomotion maintained during online learning (safety critic prevents >95% of potential falls)
- Convergence to improved policy within 2-4 hours of real-world interaction
- Real-world fine-tuning improves velocity tracking by 20-35% over sim-only policies
- Handles terrains and conditions not represented in simulation training
- Mixed replay buffer prevents catastrophic forgetting of simulation-learned behaviors
- Online adaptation demonstrates continued improvement over weeks of deployment

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper directly addresses the next step after Mini Cheetah's PPO training in MuJoCo with domain randomization. The Learn-in-Real framework provides a principled approach to closing the sim-to-real gap that domain randomization alone cannot bridge. The safety critic concept is essential for safe exploration on the physical Mini Cheetah hardware. The mixed replay buffer combining MuJoCo simulation data with real-world data prevents forgetting while enabling adaptation. The focused exploration mechanism could target specific failure modes identified during initial real-world deployment (e.g., specific floor surfaces, inclines, or velocity ranges where the 12 DoF controller with PD at 500 Hz underperforms).

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The Learn-in-Real paradigm applies to all levels of Cassie's 4-level hierarchy. The Planner can be fine-tuned with real-world navigation experience. The Primitives can adapt gait patterns to Cassie's actual dynamics. The Controller can adjust to real actuator characteristics. The Safety layer (LCBF) can be refined with real failure data. The safety critic aligns with the existing Learned Control Barrier Function (LCBF) component—both aim to constrain exploration to safe regions. The mixed replay buffer approach could combine data from the RSSM/Dreamer world model with real-world experience. The focused exploration mechanism could target specific hierarchy levels or terrain types where the gap is largest.

## What to Borrow / Implement
- Implement mixed replay buffer combining simulation and real-world data for both projects
- Adopt the safety critic as a complement to LCBF for safe real-world exploration on Cassie
- Use focused exploration to target specific sim-to-real failure modes after initial deployment
- Apply entropy regularization during simulation pre-training to maintain adaptability
- Implement performance monitoring to identify underperforming behavior modes for targeted improvement
- Design real-world data collection protocols following the paper's safe exploration framework

## Limitations & Open Questions
- Online learning requires supervised real-world sessions, increasing deployment cost
- Safety critic must be highly reliable—a single failure can damage hardware
- 2-4 hours of real-world interaction may be insufficient for complex terrains
- The approach assumes the sim-trained policy is close to optimal—large sim-to-real gaps may not be bridgeable
- Open question: How to perform online learning at different hierarchy levels simultaneously without interference?
- Catastrophic forgetting risk increases as real-world data grows relative to simulation data
- Thermal and wear effects on actuators create non-stationary adaptation targets
