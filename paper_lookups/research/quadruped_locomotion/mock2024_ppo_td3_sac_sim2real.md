# Sim-to-Real: A Performance Comparison of PPO, TD3, and SAC Reinforcement Learning Algorithms for Quadruped Walking Gait Generation

**Authors:** Mock, Muknahallipatna
**Year:** 2024 | **Venue:** SCIRP (Scientific Research Publishing)
**Links:** [SCIRP Paper](https://www.scirp.org/journal/paperinformation?paperid=131938)

---

## Abstract Summary
This paper provides a systematic head-to-head comparison of three widely used RL algorithms—Proximal Policy Optimization (PPO), Twin Delayed DDPG (TD3), and Soft Actor-Critic (SAC)—for quadruped walking gait generation with sim-to-real transfer. The study controls for all confounding factors (network architecture, observation space, reward function, domain randomization) to isolate algorithmic differences in learning speed, asymptotic performance, gait quality, and transferability.

The authors evaluate across multiple sensor configurations (proprioceptive only, proprioceptive + IMU, proprioceptive + IMU + foot contact) to understand how observation richness interacts with algorithm choice. Performance metrics include forward velocity tracking accuracy, quaternion RMS error for body orientation stability, energy efficiency, and sim-to-real transfer success rate. Experiments use a Unitree-class quadruped in PyBullet simulation with transfer to real hardware.

The central finding is that SAC achieves the best overall performance with domain randomization enabled, producing smoother and more energy-efficient gaits that transfer more reliably to real hardware. However, PPO remains the most stable to train (fewer hyperparameter-sensitive failures) and converges faster in wall-clock time for simple gait targets. TD3 struggles with the high-dimensional continuous action space and exhibits significant variance across random seeds.

## Core Contributions
- First controlled comparison of PPO, TD3, and SAC for quadruped locomotion under identical conditions
- Analysis of observation space richness (3 sensor configurations) and its interaction with algorithm choice
- Quantitative sim-to-real transfer results with domain randomization ablations for each algorithm
- SAC identified as achieving best gait quality and transfer, PPO as most robust to hyperparameters
- Metrics spanning velocity tracking, orientation stability (quaternion RMSE), energy efficiency, and transfer success
- Practical recommendations for algorithm selection based on project constraints (compute budget, sim-to-real requirements)
- Reproducible experimental protocol with full hyperparameter specifications

## Methodology Deep-Dive
The experimental setup uses a 12-DoF quadruped model (3 joints per leg: hip abduction/adduction, hip flexion/extension, knee flexion/extension) in PyBullet physics simulation at 240 Hz. The policy outputs target joint positions at 50 Hz, converted to torques via PD control. All three algorithms share an identical MLP architecture: 2 hidden layers of 256 units with ReLU activations, outputting Gaussian (PPO, SAC) or deterministic + noise (TD3) actions.

Three observation configurations are tested: (A) **Proprioceptive-only** (joint positions, velocities: 24 dims); (B) **Proprio + IMU** (adding body orientation quaternion, angular velocity, linear acceleration: 37 dims); (C) **Full** (adding binary foot contact sensors: 41 dims). The reward function is shared across all algorithms: r = w_v · exp(-||v - v_target||²) + w_θ · exp(-||θ_rp||²) + w_τ · (-||τ||²) + w_s · (-||a_t - a_{t-1}||²), combining velocity tracking, orientation stability, torque penalty, and action smoothness.

Domain randomization applies uniform perturbations to: ground friction (μ ∈ [0.5, 1.2]), body mass (±15%), center-of-mass offset (±2 cm), motor strength (±10%), observation noise (Gaussian, σ = 0.01), and action delay (0–2 steps). Each algorithm is trained with and without domain randomization, producing 6 conditions × 5 random seeds = 30 training runs.

PPO uses clipping ratio ε = 0.2, GAE λ = 0.95, minibatch size 64, 10 epochs per update, and learning rate 3e-4. SAC uses automatic entropy coefficient tuning (α), replay buffer size 1M, batch size 256, τ = 0.005 for soft target updates, and learning rate 3e-4. TD3 uses delayed policy updates (every 2 critic updates), target policy smoothing (σ = 0.2, clip = 0.5), replay buffer 1M, batch size 256, and learning rate 1e-3.

Sim-to-real transfer is evaluated by deploying the trained policy on a real quadruped on three surfaces: rubber mat, carpet, and tile. Each policy is tested 10 times per surface, walking forward for 5 meters. Transfer success is binary: the robot completes the walk without falling. Gait quality is additionally scored by forward velocity error and body sway magnitude.

## Key Results & Numbers
- **SAC with domain randomization** achieves best overall performance: 92% sim-to-real success, velocity error 6.3%, quaternion RMSE 0.034
- **PPO with domain randomization** is close second: 88% sim-to-real success, velocity error 7.1%, quaternion RMSE 0.039
- **TD3 with domain randomization** underperforms: 74% sim-to-real success, velocity error 11.2%, quaternion RMSE 0.058
- SAC converges in ~5M environment steps; PPO in ~3M steps; TD3 in ~8M steps
- Without domain randomization: PPO 52%, SAC 61%, TD3 38% sim-to-real success (drastic reduction)
- Full observation space (Config C) improves all algorithms by 5–12% success rate over proprioceptive-only (Config A)
- SAC produces 18% lower cost-of-transport than PPO, indicating more energy-efficient gaits
- PPO training variance across seeds: ±4% success; SAC: ±6%; TD3: ±15% (PPO most consistent)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Critical**
This paper directly informs the Mini Cheetah project's core design decision: algorithm selection for quadruped locomotion learning. The current PPO-based approach is validated as a strong choice (88% transfer success, lowest training variance), but the results suggest that switching to SAC could yield superior gait quality and sim-to-real transfer (92% success, 18% better energy efficiency).

The domain randomization ablation is particularly valuable: it quantifies exactly how much DR contributes to transfer success for each algorithm, informing the Mini Cheetah's randomization strategy. The observation space comparison suggests that including foot contact sensors (Config C) is worth the additional sensing complexity. The reward function formulation can be directly adopted for Mini Cheetah training.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
While the paper focuses on quadruped locomotion, the algorithm comparison insights transfer to Cassie's low-level Controller module, which also uses PPO. The finding that SAC outperforms PPO in gait quality suggests that Cassie's Controller could benefit from off-policy learning, especially given SAC's better sample efficiency (important for the expensive bipedal simulation). However, PPO's superior training stability may be more critical for Cassie's hierarchical training, where instability in the low-level controller would propagate to all higher levels.

The domain randomization findings are relevant to Cassie's sim-to-real pipeline, particularly the quantified impact of each randomization parameter (friction and mass being most impactful).

## What to Borrow / Implement
- Run SAC alongside PPO for Mini Cheetah and compare on identical evaluation metrics
- Adopt the full observation space configuration (proprio + IMU + foot contact) for maximum transfer success
- Use the paper's domain randomization parameter ranges as starting points for Mini Cheetah (friction [0.5, 1.2], mass ±15%, motor strength ±10%)
- Implement the quaternion RMSE metric for orientation stability evaluation alongside forward velocity tracking
- Consider SAC for Cassie's low-level controller if sample efficiency is a bottleneck

## Limitations & Open Questions
- Evaluation limited to flat surfaces (rubber mat, carpet, tile); no rough or uneven terrain tested for sim-to-real
- Only forward walking evaluated; turning, lateral movement, and speed transitions not compared
- PyBullet simulation may yield different results than MuJoCo (different contact models, integrators)
- TD3's poor performance may be partly due to suboptimal hyperparameters for the locomotion domain
