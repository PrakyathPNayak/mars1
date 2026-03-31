# ODE-RSSM: Learning Stochastic Recurrent State Space Model from Irregularly Sampled Data

**Authors:** Zhaolin Yuan, Dai Wei, et al.
**Year:** 2023 | **Venue:** AAAI 2023
**Links:** https://www.comp.hkbu.edu.hk/~henrydai/pubs/ODE-RSSM-AAAI23.pdf

---

## Abstract Summary
ODE-RSSM extends the RSSM by incorporating ODE networks to handle irregularly sampled time-series data. Uses efficient reparameterization for batch training over variable time steps, modeling continuous evolution of latent states between observations. Outperforms standard RSSM on dynamic, noisy datasets by naturally interpolating between observation timestamps.

## Core Contributions
- Extended the RSSM framework with Neural ODE components that model continuous-time latent state evolution between discrete observations
- Developed efficient reparameterization tricks enabling batch training over sequences with variable time intervals
- Demonstrated superior open-loop prediction compared to discrete RSSM on datasets with irregular sampling patterns
- Provided a principled approach to handling asynchronous sensor data in model-based RL
- Unified the deterministic-stochastic RSSM structure with continuous-time ODE dynamics, maintaining the benefits of both
- Showed that ODE-based state transitions avoid the fixed-timestep assumption that limits standard RSSMs

## Methodology Deep-Dive
The standard RSSM assumes observations arrive at fixed time intervals, with the GRU transition model implicitly learning a single-step dynamics function. ODE-RSSM relaxes this assumption by replacing the discrete state transition with a Neural ODE that defines continuous-time dynamics. The deterministic state evolves according to a learned ODE: dh/dt = f_θ(h, z, a), where h is the deterministic state, z is the stochastic latent, and a is the action. This ODE is integrated over the actual time interval Δt between observations using a numerical solver (e.g., Dormand-Prince).

The stochastic component is preserved from the original RSSM: at each observation time, a posterior distribution q(z_t | h_t, o_t) is inferred, and a prior p(z_t | h_t) is used during planning. The key difference is that h_t is now obtained by integrating the ODE from the previous observation time rather than applying a fixed GRU step. This means the model naturally handles varying control frequencies, sensor dropout, and asynchronous multi-modal observations.

For efficient training, the authors introduce a reparameterization scheme that allows batched ODE integration even when sequences have different time intervals. Rather than solving each sequence individually (which would be prohibitively slow), they group time intervals and use interpolation techniques to approximate the ODE solutions in parallel. This brings the training efficiency close to discrete RSSM while maintaining the continuous-time benefits.

The model is trained using the same variational framework as standard RSSM: minimize reconstruction loss (observations and rewards), maximize information in the stochastic latent (KL divergence), and optionally predict future states for planning. The ODE solver adds computational overhead compared to a simple GRU step, but this is offset by better prediction accuracy, especially over long horizons where the continuous dynamics provide more physically plausible state evolution.

Experiments focus on time-series prediction tasks with deliberately introduced irregular sampling (missing data, variable rates), where ODE-RSSM significantly outperforms vanilla RSSM and other baselines. The continuous-time formulation provides natural interpolation between observations and extrapolation beyond the training time resolution.

## Key Results & Numbers
- Superior open-loop prediction accuracy versus discrete RSSM on irregularly sampled benchmarks
- Handles asynchronous sensor data without preprocessing or imputation
- Efficient batch training via reparameterization, approaching discrete RSSM training speed
- Robust to varying levels of missing data (10%-50% observation dropout)
- Maintains prediction quality when test-time sampling rate differs from training rate
- Computational overhead of ODE solver is ~2-3x compared to GRU step, mitigated by batching

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Useful for handling varying control frequencies or sensor dropout in real-world deployment. The Mini Cheetah runs PD control at 500 Hz, but real-world sensor data may arrive at different rates (IMU at 1000 Hz, joint encoders at 500 Hz, vision at 30 Hz). ODE-RSSM provides a principled way to fuse these multi-rate observations into a single world model. If deploying a world-model-based approach on real hardware, the continuous-time dynamics would handle the inevitable timing jitter and occasional dropped sensor readings more gracefully than a fixed-timestep RSSM.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
ODE-RSSM directly combines two core Project B components — Neural ODE and RSSM — into a unified framework, making it highly relevant to the architecture design. The Neural ODE Gait Phase module in Project B models continuous gait evolution, and ODE-RSSM provides a tested approach for integrating this with the RSSM-based planner. Cassie's hierarchical architecture operates at multiple time scales (planner at low frequency, controller at high frequency), and ODE-RSSM naturally handles this multi-rate structure. The continuous-time latent dynamics would allow the planner to reason about gait phase evolution between discrete decision points, enabling smoother transitions between locomotion primitives.

## What to Borrow / Implement
- Implement ODE-based state transitions in the planner's RSSM to handle multi-rate sensing on Cassie
- Use the reparameterization trick for efficient batch training of the ODE-RSSM
- Adopt the continuous-time dynamics formulation for the Neural ODE Gait Phase module, potentially unifying it with the RSSM planner
- Test robustness to sensor dropout by training with artificially introduced missing observations
- Use the ODE solver choice (Dormand-Prince with adaptive step size) for the Neural ODE components
- Evaluate whether continuous-time RSSM improves long-horizon prediction for locomotion planning

## Limitations & Open Questions
- ODE solver adds 2-3x computational overhead, which may be prohibitive for real-time control at 500 Hz
- Unclear how well the approach scales to high-dimensional observation spaces (e.g., full proprioception + exteroception)
- Batch reparameterization introduces approximation errors that may compound over long sequences
- Not directly validated on locomotion tasks — results are on time-series benchmarks
- Integration with model-based RL (policy training on imagined trajectories) not fully explored
- Adaptive ODE solvers may have unpredictable compute times, complicating real-time deployment
- Interaction between ODE dynamics and contact discontinuities (foot strikes) may cause numerical issues
