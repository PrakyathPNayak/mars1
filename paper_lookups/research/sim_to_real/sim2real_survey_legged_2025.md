# Sim-to-Real Transfer for Locomotion Tasks on Legged Robots: A Survey

**Authors:** Luis Scheuch
**Year:** 2025 | **Venue:** Survey/Thesis (HULKs, Hamburg University of Applied Sciences)
**Links:** [PDF](https://hulks.de/_files/PA_Luis-Scheuch.pdf)

---

## Abstract Summary
This comprehensive survey reviews over 250 publications on sim-to-real transfer for legged robot locomotion, providing a structured taxonomy of methods spanning domain randomization, system identification, history-aware observations, curriculum learning, simulation grounding, and predictive control integration. The survey is organized around the fundamental question: how can policies trained in simulation be reliably transferred to real legged robots despite inevitable modeling discrepancies?

The author categorizes sim-to-real approaches into three broad families: (1) robustification methods that make policies inherently tolerant of sim-real gaps (domain randomization, observation noise, action delays), (2) adaptation methods that explicitly estimate and correct for sim-real discrepancies at deployment time (system identification, latent adaptation, history encoders), and (3) grounding methods that improve simulation fidelity to minimize the gap a priori (physics parameter tuning, learned simulators, data-augmented dynamics). For each category, the survey traces historical development, identifies key algorithmic innovations, and compares empirical results across standardized benchmarks.

A central conclusion is that no single technique suffices for reliable sim-to-real transfer; instead, state-of-the-art systems combine multiple methods into structured pipelines. The most successful deployments (e.g., ANYmal, MIT Mini Cheetah, Cassie) use domain randomization during training, history-aware observation encoders for implicit adaptation, curriculum learning for progressive skill acquisition, and careful system identification for critical subsystems. The survey provides actionable recommendations for designing such pipelines.

## Core Contributions
- Comprehensive taxonomy of 250+ sim-to-real methods organized by mechanism (robustification, adaptation, grounding)
- Historical tracing of each technique family from inception to current state-of-the-art
- Cross-comparison of methods on standardized legged locomotion benchmarks (flat ground, stairs, rough terrain)
- Identification of synergistic method combinations that yield best transfer performance
- Analysis of failure modes unique to each approach (e.g., over-conservatism from excessive randomization)
- Practical pipeline design recommendations for quadruped and biped platforms
- Coverage of emerging techniques: learned simulators, neural ODEs for dynamics, diffusion-based policy transfer

## Methodology Deep-Dive
The survey's domain randomization section covers the evolution from uniform parameter perturbation (Tobin et al., 2017) through structured randomization schedules to automatic domain randomization (ADR) that adapts randomization ranges based on policy performance. Key parameters identified for legged locomotion include: ground friction (μ ∈ [0.2, 1.5]), payload mass (±30%), motor strength scaling (±20%), joint damping, communication delay (0-20 ms), and terrain geometry. The survey notes that excessive randomization leads to overly conservative policies that sacrifice performance for robustness—a fundamental trade-off quantified across several studies.

The history-aware observation section details how temporal context enables implicit system identification at deployment. Architectures range from simple observation stacking (concatenating the last N timesteps) through recurrent networks (LSTM, GRU) to Transformer-based encoders. The survey identifies a key design choice: whether the history encoder is trained end-to-end with the policy (simpler but potentially less interpretable) or as a separate adaptation module (more modular but requires careful interface design). Empirical results suggest that 0.5-2 seconds of observation history (25-100 timesteps at 50 Hz) captures sufficient dynamics for most sim-real adaptation needs.

The curriculum learning section distinguishes between environment curricula (progressively harder terrains), task curricula (progressively faster speeds or complex gaits), and reward curricula (progressively tighter performance requirements). The survey finds that terrain-based curricula with automatic difficulty adjustment (tracking success rate and advancing when >80% success) are the most widely adopted approach for legged locomotion, with the IsaacGym/Legged Gym framework establishing the standard implementation.

The system identification section covers both classical approaches (Bayesian optimization, CMA-ES) and learned approaches (neural network dynamics models, sim-to-real residual learning). A key finding is that actuator modeling is consistently the highest-impact identification target, with ground contact models as a close second. The survey highlights recent work combining learned residual dynamics with physics-based simulators to get the best of both approaches.

The simulation grounding section evaluates different physics engines (MuJoCo, Isaac Gym, Bullet, DART) for legged locomotion, comparing contact model fidelity, computational speed, and ease of parameter tuning. MuJoCo's soft contact model is noted as particularly well-suited for legged locomotion due to its differentiability and stability, while Isaac Gym's GPU parallelization enables larger-scale domain randomization.

## Key Results & Numbers
- Domain randomization alone achieves 60-80% success rate on flat-ground transfer; combined with history encoding reaches 90-95%
- History encoders with 50-100 timestep windows outperform memoryless policies by 25-40% on unseen terrains
- Curriculum learning accelerates training convergence by 2-5× compared to fixed difficulty
- System identification reduces sim-real joint tracking error by 50-75% compared to default parameters
- Optimal domain randomization ranges: friction μ ∈ [0.3, 1.2], mass ±25%, motor strength ±15%
- Combined pipeline (DR + history + curriculum + SysID) achieves >95% success across terrain types
- Survey covers 250+ papers spanning 2017-2025

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

This survey is an essential reference for designing the Mini Cheetah's complete sim-to-real pipeline. The taxonomy directly maps to the project's needs: domain randomization ranges for MuJoCo parameters, history-aware observation architecture choices for the PPO policy, curriculum learning schedules for progressive terrain difficulty, and system identification priorities for the Mini Cheetah's actuators. The survey's recommendation to combine multiple methods into a structured pipeline validates the project's planned approach.

Specific actionable guidance includes: using 50-timestep observation history with a GRU encoder, randomizing friction in [0.3, 1.2] and payload mass ±25%, implementing terrain curriculum with automatic advancement at 80% success rate, and prioritizing actuator delay and torque bandwidth for system identification. The cross-comparison tables allow direct selection of the best-performing technique combinations for quadruped locomotion.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchical system, the survey provides critical context for each level of the hierarchy. At the Planner level, the world model literature review covers RSSM and Dreamer variants that directly inform Cassie's RSSM-based planner. At the Primitives level, the curriculum learning section guides how to train diverse locomotion skills (DIAYN/DADS) with progressive difficulty. At the Controller level, the system identification section informs actuator modeling fidelity for the CBF-QP safety filter. At the Safety level, the domain randomization analysis helps determine what perturbations the safety layer must be robust to.

The survey's coverage of bipedal-specific challenges—underactuation, narrow support polygon, higher fall risk—is particularly relevant to Cassie. The recommended approach of combining model-based safety constraints with learned adaptive policies aligns with the project's CBF-QP + RL architecture.

## What to Borrow / Implement
- Adopt the recommended DR ranges (friction, mass, motor strength) as starting points for both Mini Cheetah and Cassie MuJoCo training
- Implement history-aware observation encoder (GRU, 50-100 timestep window) for implicit sim-real adaptation
- Use terrain curriculum with automatic difficulty advancement at 80% success threshold
- Follow the pipeline ordering: SysID → calibrated sim → DR + curriculum → history encoder → deploy
- Reference the survey's failure mode analysis to diagnose transfer issues during deployment

## Limitations & Open Questions
- Survey breadth means individual methods receive limited depth of analysis; may need to consult original papers for implementation details
- Cross-comparison results are compiled from different papers with different experimental setups, limiting direct comparability
- Biped-specific analysis is thinner than quadruped coverage, reflecting the field's current focus
- Does not cover very recent (late 2025) developments in foundation model-based sim-to-real transfer
