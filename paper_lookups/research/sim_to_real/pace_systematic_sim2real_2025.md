# Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots (PACE)

**Authors:** Marko Bjelonic, Pascal Tischhauser, Marco Hutter (ETH Zurich)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv:2509.06342](https://arxiv.org/abs/2509.06342)

---

## Abstract Summary
PACE introduces a systematic, bottom-up methodology for sim-to-real transfer that focuses on identifying and calibrating the most impactful simulation parameters—particularly actuator dynamics—rather than relying on broad domain randomization. The key insight is that a small, carefully chosen set of actuator parameters (communication delay, torque response, friction characteristics) accounts for the majority of the sim-real gap in legged locomotion.

The authors propose using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to optimize a compact parameter vector that aligns simulated actuator behavior with real-world measurements. This evolutionary optimization approach efficiently searches the parameter space without requiring gradients, making it practical for physical system identification. The method is validated with zero-shot transfer across 10 different legged robot platforms, achieving a 32% reduction in Cost of Transport (CoT) on the ANYmal quadruped compared to untuned simulation baselines.

The paper argues that minimizing the dimensionality of the identification problem—targeting only the parameters that matter most—yields better generalization than exhaustive randomization over hundreds of parameters. This philosophy of principled, minimal parameter identification represents a shift from the brute-force domain randomization paradigm toward more structured sim-to-real pipelines.

## Core Contributions
- Bottom-up system identification framework prioritizing actuator dynamics (delay, torque curves, friction) as the dominant source of sim-real mismatch
- CMA-ES evolutionary optimization for compact parameter identification without gradient computation
- Zero-shot sim-to-real transfer validated on 10 diverse legged robots spanning different morphologies and sizes
- 32% Cost of Transport reduction on ANYmal through precise actuator calibration
- Demonstration that a minimal parameter set (< 20 parameters per actuator) captures the critical sim-real gap
- Systematic ablation showing relative importance of delay modeling, torque saturation, and friction terms
- Open methodology applicable to any torque-controlled legged robot with minimal hardware instrumentation

## Methodology Deep-Dive
The PACE framework decomposes the sim-to-real problem into a hierarchy of subsystem identification stages. At the lowest level, individual actuator dynamics are characterized through a series of targeted experiments: step response tests measure communication and computational delays, ramp torque tests capture torque bandwidth and saturation, and constant-velocity tests isolate friction coefficients (Coulomb, viscous, and Stribeck components). Each test is designed to excite specific dynamic modes while keeping other states controlled.

The actuator model parameterizes each joint with a transfer function capturing: (1) a pure delay τ_d representing communication latency and computation time, (2) a first-order low-pass filter with time constant τ_a for actuator bandwidth, (3) a nonlinear torque saturation function based on motor thermal limits and current constraints, and (4) a velocity-dependent friction model combining Coulomb friction F_c, viscous damping b_v, and Stribeck friction with decay constant v_s. This gives approximately 6-8 parameters per actuator, yielding a total parameter vector of 72-96 dimensions for a 12-DoF quadruped.

CMA-ES is employed to minimize the discrepancy between simulated and real actuator responses. The fitness function computes the L2 norm between real and simulated joint trajectories across the battery of identification experiments. CMA-ES is particularly well-suited here because it handles non-convex, noisy fitness landscapes without requiring gradients, and its population-based search naturally explores multiple parameter modes. The authors use a population size of 50-100 with 200-500 generations, converging in approximately 2-4 hours of computation.

After actuator-level identification, the framework performs a whole-body validation phase where the fully parameterized simulation is tested against real-robot locomotion data. The key metric is tracking error between commanded and actual joint trajectories during dynamic gaits. If residual errors exceed a threshold, secondary parameters (link inertias, ground contact models) are refined in a subsequent identification stage, though the authors report that actuator dynamics alone account for 70-85% of the total sim-real gap.

The zero-shot transfer protocol trains a locomotion policy entirely in the calibrated simulation using PPO with standard reward shaping, then deploys directly on hardware without any fine-tuning. The 32% CoT improvement on ANYmal is measured against a policy trained in an uncalibrated simulation with default parameters, demonstrating that precise system identification provides substantially more value than increasing domain randomization ranges.

## Key Results & Numbers
- Zero-shot transfer success on 10 different legged robots (quadrupeds, bipeds, hexapods)
- 32% reduction in Cost of Transport on ANYmal C quadruped
- Actuator dynamics account for 70-85% of total sim-real gap
- Compact parameter vector: 6-8 parameters per actuator joint
- CMA-ES convergence: 200-500 generations with population size 50-100
- Identification procedure requires ~2 hours of real-robot data collection per platform
- Joint tracking RMSE reduced from 0.15 rad to 0.04 rad after calibration
- Communication delay identification accuracy within ±0.5 ms of ground truth

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

This paper is directly applicable to the Mini Cheetah sim-to-real pipeline. The Mini Cheetah's proprietary actuators have well-documented communication delays and torque bandwidth limitations that create significant sim-real gaps in MuJoCo. PACE's bottom-up actuator identification methodology can be applied almost directly: characterize each of the 12 joints' delay, bandwidth, and friction parameters using CMA-ES, then train PPO policies in the calibrated MuJoCo environment.

The compact parameter identification philosophy aligns perfectly with the Mini Cheetah project's constraint of limited real-robot access time. Instead of spending weeks tuning hundreds of domain randomization ranges, PACE's targeted approach requires only ~2 hours of structured data collection to capture the dominant dynamics. The 32% CoT improvement demonstrates that this investment pays off substantially in deployment performance. The Mini Cheetah's relatively simple actuator design (direct-drive or quasi-direct-drive) may even yield cleaner identification than the gear-driven actuators in ANYmal.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchical RL system, PACE provides a rigorous foundation for simulation fidelity at the lowest level of the hierarchy—the Controller and Safety layers. Cassie's actuators include both direct-drive and geared joints with significant compliance and backlash, making accurate actuator modeling critical for the CBF-QP safety filter to function correctly in the real world. If the simulation's actuator dynamics diverge from reality, the safety constraints computed by the CBF may be overly conservative or insufficiently protective.

The CMA-ES identification approach is well-suited for Cassie's complex leg mechanism, where the leaf-spring compliance and four-bar linkage kinematics create highly nonlinear actuator-to-joint mappings. PACE's systematic decomposition into subsystem identification stages can be extended to include these mechanical transmission effects as additional parameters in the identification vector.

## What to Borrow / Implement
- Apply CMA-ES actuator identification to Mini Cheetah's 12 joints in MuJoCo before policy training
- Implement the structured test protocol (step response, ramp torque, constant velocity) for each joint
- Use the compact actuator model (delay + bandwidth + friction) as MuJoCo actuator plugins
- Adopt the validation metric (joint tracking RMSE during dynamic gaits) to verify simulation fidelity
- Extend the methodology to Cassie's leaf-spring and four-bar linkage dynamics

## Limitations & Open Questions
- Assumes actuator dynamics are the dominant sim-real gap source; may underweight contact dynamics for rough terrain
- CMA-ES requires real-robot data collection, which may be impractical for damaged or prototype robots
- Static friction (Stribeck) identification requires very slow motions that may not excite all dynamic modes
- Does not address how actuator parameters drift over time due to wear, temperature, or damage
