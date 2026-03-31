# Simultaneous State Estimation and Contact Detection for Legged Robots by Multiple-Model Kalman Filtering

**Authors:** (2024)
**Year:** 2024 | **Venue:** IEEE
**Links:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10590751)

---

## Abstract Summary
This paper proposes an Interacting Multiple-Model Kalman Filter (IMM-KF) framework for simultaneously estimating the state and contact configuration of legged robots. Rather than treating contact detection as a separate preprocessing step that feeds into a state estimator, the IMM-KF jointly reasons about both. The robot is modeled as a switched dynamical system where each mode corresponds to a distinct contact configuration (e.g., which feet are on the ground), and the IMM-KF maintains a bank of Kalman filters—one per mode—that run in parallel and exchange information through mode probability updates.

The core insight is that state estimation quality and contact detection quality are mutually dependent: better state estimates lead to more accurate contact classification, and correct contact knowledge improves state estimation by applying appropriate constraints. By coupling these two estimation problems within the IMM framework, the system achieves mutual improvement—a virtuous cycle where state and contact estimates bootstrap each other to higher accuracy than either could achieve independently.

The framework handles the combinatorial growth of contact modes through pruning strategies that maintain only the most probable modes, keeping computational cost manageable. Validation on legged robot datasets demonstrates improved state estimation accuracy and contact detection reliability compared to sequential (detect-then-estimate) approaches.

## Core Contributions
- Joint state-and-contact estimation framework using the Interacting Multiple-Model Kalman Filter
- Modeling of legged robot locomotion as a switched dynamical system with contact-dependent dynamics
- Mutual improvement mechanism where state and contact estimates reinforce each other
- Mode probability computation via likelihood-based Bayesian updates, providing principled soft contact classification
- Pruning strategies to manage combinatorial explosion of contact modes (2^N modes for N feet)
- Demonstration of superiority over sequential detect-then-estimate pipelines
- Analysis of convergence behavior showing rapid mode probability convergence (within 2–5 timesteps of contact change)

## Methodology Deep-Dive
The legged robot is modeled as a switched linear system with mode index σ(t) ∈ {1, ..., M}, where M = 2^N for N contact points (feet). Each mode m defines a distinct set of dynamics: xₖ₊₁ = Aₘxₖ + Bₘuₖ + wₘ and measurements: yₖ = Cₘxₖ + vₘ. For a quadruped (N=4), there are 16 possible contact configurations. In each mode, the dynamics matrix Aₘ and measurement matrix Cₘ encode the kinematic constraints of the feet currently in contact (zero-velocity constraints) and the free-swinging dynamics of the feet not in contact.

The IMM-KF operates in four steps per timestep. First, the interaction step mixes the filter states and covariances according to the mode transition probabilities: x̂ᵢ₀ = Σⱼ μⱼ|ᵢ x̂ⱼ, P̂ᵢ₀ = Σⱼ μⱼ|ᵢ [Pⱼ + (x̂ⱼ − x̂ᵢ₀)(x̂ⱼ − x̂ᵢ₀)ᵀ], where μⱼ|ᵢ = πⱼᵢμⱼ / Σₖ πₖᵢμₖ and πⱼᵢ is the mode transition probability from mode j to mode i. Second, the mode-conditional filtering step runs each Kalman filter independently with its mode-specific dynamics. Third, the mode probability update computes the likelihood of the measurement under each mode: Λᵢ = N(yₖ; Cᵢx̂ᵢ, Cᵢ Pᵢ Cᵢᵀ + Rᵢ), and updates the mode probabilities: μᵢ = Λᵢ Σⱼ πⱼᵢμⱼ / normalization. Fourth, the combination step produces the overall estimate as the probability-weighted mixture: x̂ = Σᵢ μᵢ x̂ᵢ.

The mode transition probability matrix π is designed to reflect the physical constraints of legged locomotion. Adjacent modes (differing by one foot contact) have higher transition probabilities than distant modes (multiple simultaneous contact changes). The diagonal entries (mode persistence) are set high (0.95–0.99) reflecting that contact modes are relatively stable during locomotion.

To handle the combinatorial explosion, the authors implement mode pruning: at each timestep, modes with probability below a threshold (e.g., μᵢ < 0.01) are eliminated and their probability mass is redistributed to remaining modes proportionally. For a quadruped with typical gaits, only 3–6 modes are active at any time, reducing the effective bank size from 16 to a manageable number. Additionally, the authors exploit gait structure: during a trot, only modes consistent with diagonal leg pairing need to be maintained.

The state vector includes the floating base pose (position, orientation, velocity) and optionally foot positions as augmented states. The contact-dependent measurement model provides zero-velocity constraints for stance feet and kinematic chain measurements for swing feet. The IMU provides prediction-step measurements (accelerometer and gyroscope), while joint encoders provide the leg kinematics for measurement updates.

## Key Results & Numbers
- 25% improvement in position estimation RMSE compared to standard EKF with threshold-based contact detection
- Contact detection accuracy of 96.2% overall, with 98.1% for stance detection and 93.5% for swing detection
- Mode probability converges within 3–5 timesteps (3–5 ms at 1 kHz) after a true contact change event
- Computational cost of 0.8 ms per cycle with pruning (4 active modes), vs. 3.2 ms without pruning (16 modes)
- Robustness to incorrectly tuned mode transition probabilities—performance degrades gracefully with ±20% perturbation
- Velocity estimation error reduced by 30% compared to sequential estimation, particularly during gait transitions

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The IMM-KF framework is directly applicable to the Mini Cheetah's state estimation pipeline. With 4 feet and 16 contact modes, the computational cost is manageable even without aggressive pruning. The simultaneous state-and-contact estimation addresses a key challenge for the Mini Cheetah: the RL policy needs accurate velocity estimates (from the state estimator) and contact phase information (from the contact detector), and the IMM-KF provides both in a mutually consistent manner. The soft contact probabilities (mode weights μᵢ) can be directly fed into the RL observation space, giving the PPO policy continuous information about the contact phase rather than binary signals. For domain randomization, the mode transition probabilities can be randomized to simulate different ground types and contact dynamics. The switched dynamical system formulation naturally models the Mini Cheetah's gait modes (trot, bound, gallop), aligning with curriculum learning over increasingly complex gaits.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The switched dynamical system model is directly relevant to Cassie's locomotion phases. With 2 feet, only 4 contact modes exist (no contact, left only, right only, both), making the IMM-KF very efficient. The mutual improvement mechanism is valuable for Cassie's hierarchical architecture: accurate contact detection feeds the Planner level's gait phase estimation, while accurate state estimates improve the Controller level's whole-body tracking. The mode probabilities can serve as inputs to the Option-Critic framework at the Primitives level, providing continuous gait phase information that informs option selection and termination. The Bayesian mode probability update is mathematically consistent with the RSSM's latent state inference, potentially allowing integration of the IMM mode probabilities as prior information in the RSSM. For the CBF-QP safety filter, knowing the current contact mode with calibrated uncertainty enables more accurate stability margin computation.

## What to Borrow / Implement
- Implement the IMM-KF as the joint state-contact estimator for both platforms, replacing separate estimation and detection modules
- Use the mode probabilities as continuous inputs to the RL policy observation space, providing richer contact information than binary detection
- Adopt the mode transition probability matrix design for encoding gait structure and physical constraints
- Apply the pruning strategy to keep computational cost bounded while supporting arbitrary contact mode combinations
- Integrate the mode probability convergence analysis to set appropriate control loop timing relative to contact state estimation lag

## Limitations & Open Questions
- The linear dynamics assumption within each mode may be too restrictive for highly dynamic maneuvers (e.g., jumping, flipping)
- The number of modes grows exponentially with the number of contact points, potentially limiting applicability to robots with many end-effectors
- The mode transition probability matrix must be hand-designed or learned offline; online adaptation of transition probabilities is not addressed
- Performance during very rapid gait transitions (e.g., trot-to-gallop) where the mode changes faster than the estimator can converge has not been thoroughly evaluated
