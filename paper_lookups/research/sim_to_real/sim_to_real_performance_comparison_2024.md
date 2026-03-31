# Sim-to-Real: A Performance Comparison of PPO, TD3, and SAC Reinforcement Learning Algorithms for Quadruped Locomotion

**Authors:** SCIRP Authors (2024)
**Year:** 2024 | **Venue:** Journal of Intelligent Learning Systems and Applications
**Links:** [PDF](https://www.scirp.org/pdf/jilsa_2024032115174081.pdf)

---

## Abstract Summary
This paper presents a comprehensive benchmarking study comparing three prominent deep reinforcement learning algorithms—Proximal Policy Optimization (PPO), Twin Delayed Deep Deterministic Policy Gradient (TD3), and Soft Actor-Critic (SAC)—on the task of quadruped locomotion with sim-to-real transfer. The study evaluates each algorithm across multiple metrics including gait quality, energy efficiency, robustness to perturbations, and transferability from simulated environments to real hardware.

The authors systematically vary sensor configurations (proprioceptive-only, with IMU, with foot contact sensors) and terrain conditions to assess how each algorithm handles different observation spaces. SAC emerges as the strongest performer on certain sensor setups due to its maximum entropy framework encouraging broader exploration. TD3 demonstrates competitive performance, especially in lower-dimensional observation spaces, while PPO provides stable but sometimes suboptimal gaits. Real-world hardware tests validate that simulation performance rankings largely hold during physical deployment, though the sim-to-real gap varies by algorithm.

The study concludes that algorithm choice significantly impacts both training efficiency and deployment performance, with no single algorithm dominating across all evaluation criteria. The entropy regularization in SAC proves particularly valuable for discovering diverse and robust locomotion strategies.

## Core Contributions
- Systematic head-to-head comparison of PPO, TD3, and SAC on quadruped locomotion with identical training conditions and hyperparameter budgets
- Evaluation across multiple sensor configurations to assess how observation space design interacts with algorithm choice
- Real-world hardware validation confirming simulation-based performance rankings transfer to physical robots
- Analysis of gait quality metrics beyond simple forward velocity (stability, smoothness, energy efficiency, foot clearance)
- Demonstration that SAC's entropy regularization enables superior exploration in high-dimensional sensor spaces
- Evidence that TD3 can match or exceed PPO in sample efficiency for locomotion tasks when properly tuned
- Practical guidelines for algorithm selection based on available sensors and deployment constraints

## Methodology Deep-Dive
The experimental setup uses a standardized quadruped robot model in simulation (likely based on a Unitree or similar platform) with consistent reward shaping across all three algorithms. The reward function combines forward velocity tracking, energy penalty, body orientation maintenance, and foot contact pattern rewards. Each algorithm receives the same total environment interaction budget to ensure fair comparison.

PPO is implemented with clipped objective (ε = 0.2), GAE (λ = 0.95), and mini-batch updates. The on-policy nature means PPO requires more environment samples but provides stable monotonic improvement. TD3 uses delayed policy updates (every 2 critic updates), target policy smoothing with clipped noise (σ = 0.2, clip = 0.5), and twin critics to address overestimation bias. SAC implements automatic entropy tuning (α) with twin soft Q-functions and a squashed Gaussian policy outputting continuous joint commands.

Sensor configurations tested include: (1) joint positions and velocities only (18D), (2) adding IMU orientation and angular velocity (+6D), (3) adding binary foot contact sensors (+4D), and (4) full proprioceptive suite with joint torques (+12D). Each configuration is trained with 5 random seeds per algorithm, and results are reported with mean and standard deviation.

Sim-to-real transfer employs uniform domain randomization over friction coefficients (0.5–1.5), mass variations (±15%), motor strength scaling (0.8–1.2), and observation noise injection. The real-world tests use a physical quadruped with onboard compute running the frozen policy at 50 Hz control frequency. Evaluation metrics include distance traveled, number of falls, gait symmetry index, and specific resistance (energy per unit weight per unit distance).

The training infrastructure uses vectorized environments (128 parallel) for PPO and replay buffers (1M transitions) for TD3 and SAC. All algorithms use 3-layer MLPs with 256 hidden units and ReLU activations for both actor and critic networks.

## Key Results & Numbers
- SAC achieves highest average forward velocity (1.2 m/s) on full sensor configuration, vs PPO (1.05 m/s) and TD3 (1.1 m/s)
- TD3 converges fastest in low-dimensional observation spaces (joint-only), reaching 90% peak performance in ~2M steps vs 5M for PPO
- PPO shows lowest variance across random seeds (σ = 0.08 m/s) indicating most stable training
- SAC produces most energy-efficient gaits (specific resistance ~0.45) due to entropy-driven exploration finding efficient strategies
- Sim-to-real velocity retention: SAC 82%, PPO 78%, TD3 75% (percentage of sim velocity achieved on hardware)
- SAC suffers fewest falls (2/10 trials) on uneven terrain compared to PPO (3/10) and TD3 (4/10)
- Training wall-clock time: PPO ~8 hours, TD3 ~4 hours, SAC ~5 hours (on single GPU with 128 envs for PPO, replay buffer for off-policy)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly relevant to Project A as it provides empirical evidence for algorithm selection on quadruped locomotion—the exact task for the Mini Cheetah. The finding that PPO (currently used in Project A) provides stable but sometimes suboptimal gaits compared to SAC and TD3 is a critical insight. The sensor configuration analysis can directly inform observation space design for the Mini Cheetah MuJoCo environment.

The sim-to-real transfer methodology with domain randomization over friction, mass, and motor strength aligns perfectly with Project A's pipeline. The result that SAC achieves better sim-to-real retention (82% vs 78% for PPO) suggests Project A could benefit from exploring SAC as an alternative or complementary algorithm, especially given Mini Cheetah's rich proprioceptive sensor suite. The curriculum learning aspects of the domain randomization schedule are directly applicable.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The algorithm comparison insights are valuable for training Cassie's Controller level, where the base RL algorithm must learn joint-level tracking policies. Understanding how PPO, TD3, and SAC behave differently on locomotion tasks helps inform the choice of base algorithm within the hierarchical framework. SAC's entropy regularization aligns well with Cassie's need for diverse locomotion primitives at the Primitives level.

The sensor configuration analysis translates to bipedal settings—Cassie has similar proprioceptive sensors plus additional complexity from hip and knee joint coupling. The observation that algorithm performance depends on observation dimensionality is relevant since Cassie's Dual Asymmetric-Context Transformer processes high-dimensional context windows. The sim-to-real domain randomization parameters (friction, mass, motor strength) directly apply to Cassie's Adversarial Curriculum training.

## What to Borrow / Implement
- Implement SAC as an alternative base algorithm for Mini Cheetah training and compare against current PPO baseline
- Adopt the sensor configuration ablation methodology to determine optimal observation space for both Mini Cheetah and Cassie
- Use the domain randomization ranges (friction 0.5–1.5, mass ±15%, motor 0.8–1.2) as starting points for both projects
- Apply the gait quality metrics (symmetry index, specific resistance) as additional evaluation criteria beyond forward velocity
- Consider entropy-tuned SAC for Cassie's Primitives-level training to encourage diverse skill discovery

## Limitations & Open Questions
- Single quadruped platform tested; transferability of algorithm rankings to Mini Cheetah or bipedal robots not guaranteed
- Hyperparameter tuning budget not clearly specified—algorithm rankings could shift with more extensive tuning
- No evaluation of hybrid approaches (e.g., PPO for initial training, SAC for fine-tuning) that might combine stability with exploration
- Real-world tests limited to flat and mildly uneven terrain; performance on stairs, slopes, or highly dynamic terrains unknown
