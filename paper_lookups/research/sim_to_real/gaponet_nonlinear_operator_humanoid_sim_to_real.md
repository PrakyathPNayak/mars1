# GapONet: Nonlinear Operator Learning for Bridging the Humanoid Sim-to-Real Gap

**Authors:** Various
**Year:** 2024 | **Venue:** OpenReview / Workshop 2024
**Links:** https://openreview.net/forum?id=AC1XQjy8Sa

---

## Abstract Summary
GapONet uses neural operator learning to model and predict the sim-to-real gap as a nonlinear function, enabling targeted correction of simulation inaccuracies. Instead of brute-force domain randomization that uniformly broadens the training distribution, GapONet learns a residual model that maps simulation predictions to real-world outcomes for humanoid control. This approach achieves improved transfer accuracy by directly modeling and compensating for the systematic discrepancies between simulation and reality.

## Core Contributions
- Neural operator framework that learns the sim-to-real gap as a nonlinear mapping from simulation trajectories to real-world trajectories
- Targeted gap correction that addresses specific simulation inaccuracies rather than relying on brute-force domain randomization
- Residual model architecture: simulation output + learned correction = improved real-world prediction
- Data-efficient approach requiring only modest amounts of real-world data to calibrate the gap model
- Demonstration on humanoid locomotion showing improved transfer accuracy over standard domain randomization
- Analysis of which simulation aspects contribute most to the sim-to-real gap (actuator dynamics, contact models, latency)
- Combination with domain randomization for further improvement — the approaches are complementary, not competing

## Methodology Deep-Dive
The sim-to-real gap is the discrepancy between what a simulator predicts and what happens in reality. Standard domain randomization addresses this by training across a range of simulator parameters, hoping the real world falls within the randomized distribution. This approach is effective but inefficient — it wastes capacity on parameter combinations that don't match reality and may still miss the true real-world dynamics if they fall outside the randomization range.

GapONet takes a fundamentally different approach. It learns a neural operator G that maps simulation trajectories to corrected trajectories: x_real ≈ x_sim + G(x_sim). The operator G captures the systematic, nonlinear discrepancies between simulation and reality. This is not simple system identification (fitting simulator parameters to match reality) — the gap may be due to unmodeled effects (cable dynamics, thermal motor effects, detailed contact geometry) that cannot be captured by adjusting existing simulator parameters.

The neural operator architecture is based on DeepONet or Fourier Neural Operator principles, chosen because the sim-to-real gap is fundamentally a mapping between function spaces (simulation trajectories → correction trajectories) rather than a point-to-point mapping. The operator takes a window of simulation states (positions, velocities, actions) and outputs a correction to each state. The architecture is designed to capture both instantaneous corrections (e.g., a motor is consistently 5% weaker than simulated) and temporal corrections (e.g., the real system has 3ms more latency than simulated).

Training the gap model requires paired simulation-real data. The same control commands are executed in simulation and on real hardware, and the resulting trajectories are compared. The gap model is trained to predict the difference. Critically, this requires only open-loop trajectory data, not closed-loop policy deployment — random or scripted motions provide sufficient excitation to characterize the gap. The authors show that 15-30 minutes of real-world data is sufficient to train an accurate gap model.

Once trained, GapONet can be used in two ways: (1) "Sim-to-real correction" — deploy the simulation-trained policy on the real robot and apply GapONet to correct the policy's predicted next state, enabling more accurate model-predictive control; (2) "Corrected simulation" — integrate GapONet into the simulator during training, so the policy trains in a "corrected simulation" that more closely matches reality. The second approach is more powerful as it allows the policy to adapt to realistic dynamics during training rather than at deployment time.

The combination with domain randomization is analyzed. GapONet captures the systematic, deterministic component of the sim-to-real gap, while domain randomization handles the stochastic component (parameter variation across trials, environmental changes). Using both together outperforms either alone — GapONet reduces the mean gap while randomization handles the variance.

## Key Results & Numbers
- 30-50% reduction in sim-to-real transfer error compared to standard domain randomization alone
- Only 15-30 minutes of real-world data needed to train the gap model
- Actuator dynamics identified as the largest contributor to the sim-to-real gap (40-60% of total error)
- Contact model discrepancies account for 20-30% of the gap
- Latency and communication delays account for 10-15% of the gap
- Combination of GapONet + domain randomization achieves best results, reducing transfer error by 40-60%
- Gap model generalizes across tasks — trained on walking data, improves transfer for turning and speed changes
- Computational overhead during training is modest (~15% increase in wall-clock time)

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The residual gap modeling approach is applicable to Mini Cheetah's motor dynamics mismatch. Mini Cheetah's proprietary actuators have complex dynamics (friction, saturation, thermal effects) that MuJoCo's motor model may not capture accurately. GapONet could learn these discrepancies from a small amount of real-world data, improving sim-to-real transfer beyond what domain randomization alone achieves. The finding that actuator dynamics dominate the sim-to-real gap is particularly relevant, as Mini Cheetah's high-performance motors are likely the primary source of simulation error. The 15-30 minute data requirement makes this practically feasible even with limited hardware access. However, Project A's current scope may not extend to real-world deployment, making this a future enhancement.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The neural operator for sim-to-real gap correction directly addresses Cassie's complex actuator dynamics challenge. Cassie's series elastic actuators (SEAs) are notoriously difficult to simulate accurately — the spring dynamics, cable routing, friction, and thermal effects create a large and complex sim-to-real gap. Standard domain randomization may be insufficient because the gap is systematic and nonlinear rather than random. GapONet's ability to learn and correct these systematic discrepancies from modest real-world data makes it a powerful complement to Project B's domain randomization and adversarial curriculum. The residual model could be particularly effective for Cassie's SEAs, where the gap is structured (consistent spring-related errors) rather than random. The finding that the gap model generalizes across tasks suggests training on simple walking data would improve transfer for all of Cassie's primitives.

## What to Borrow / Implement
- Implement GapONet as a residual dynamics model for both projects' simulators, trained on real-world trajectory data
- Focus gap modeling on actuator dynamics (the dominant gap source) for maximum impact
- Collect 15-30 minutes of open-loop trajectory data from real hardware for gap model training
- Combine GapONet with existing domain randomization — use both the systematic (GapONet) and stochastic (randomization) corrections
- For Cassie specifically, train the gap model to capture SEA dynamics discrepancies
- Use the gap analysis methodology (decomposing error into actuator, contact, latency components) to identify the primary sim-to-real bottlenecks for both robots
- Integrate the corrected simulator into the training loop for policy improvement

## Limitations & Open Questions
- Requires real-world data collection, which may be impractical for early-stage projects without hardware access
- Gap model is trained on specific operating conditions; performance outside these conditions (novel gaits, extreme speeds) is uncertain
- Neural operator training can be sensitive to data quality and coverage — insufficient excitation leads to poor gap modeling
- Does not address time-varying gaps (e.g., motor heating changes dynamics over a deployment session)
- How well does the gap model generalize from flat-terrain data to complex terrain scenarios?
- Can the gap model be updated online during real-world deployment for continual improvement?
- What is the interaction between GapONet correction and safety guarantees (LCBF) — does gap correction improve or complicate safety?
- The approach assumes the real system is deterministic given the same inputs — stochastic real-world effects may limit correction accuracy
