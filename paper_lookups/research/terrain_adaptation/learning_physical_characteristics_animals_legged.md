# Learning Physical Characteristics Like Animals for Legged Robots

**Authors:** Various
**Year:** 2023 | **Venue:** National Science Review (Oxford Academic)
**Links:** https://academic.oup.com/nsr/article/10/5/nwad045/7051244

---

## Abstract Summary
This paper proposes an unsupervised framework for legged robots to learn terrain physical characteristics (friction, softness, compliance) online and incrementally, similar to how animals learn through experience. The robot builds an internal model of terrain properties from proprioceptive feedback and adapts locomotion accordingly, achieving improved stability across varied terrain types.

## Core Contributions
- Develops an unsupervised online learning framework for terrain property estimation from proprioceptive data alone
- Mimics animal-like adaptation where the robot builds an incremental internal model of terrain characteristics
- Learns physical properties (friction coefficient, surface compliance, damping) without any labeled terrain data
- Demonstrates that proprioceptive signals (joint torques, foot contact forces, slip detection) contain sufficient information for terrain classification
- Shows that online adaptation improves locomotion stability by 30%+ on previously unseen terrain types
- Validates on real legged robots across sand, gravel, grass, concrete, and deformable surfaces

## Methodology Deep-Dive
The core framework draws inspiration from how animals learn about their environment through interaction. When an animal encounters a new surface, it initially moves cautiously, gradually building a model of the surface's properties through proprioceptive feedback (how the feet feel against the ground, how much they slip, how the body responds). This paper replicates this process computationally.

The terrain property estimator uses a recurrent neural network that processes a rolling window of proprioceptive observations: joint positions, velocities, torques, foot contact forces, and body IMU data. The network outputs estimates of terrain friction, compliance, and damping. Crucially, this network is trained in an unsupervised manner — there are no ground-truth terrain labels. Instead, the network is trained using a self-supervised prediction objective: predict the next proprioceptive observation given the current state and action. Terrain properties emerge as latent variables that improve prediction accuracy.

The online learning component allows the model to adapt incrementally during deployment. When the robot encounters a new terrain type, the initial prediction errors are high, causing the model to update its internal representation. Over time (typically 5-10 steps), the model converges to an accurate terrain characterization. This incremental learning uses a small replay buffer and online gradient updates, similar to continual learning approaches.

The locomotion adaptation module takes the estimated terrain properties and modifies the gait controller's parameters. On low-friction surfaces, the controller reduces foot velocity during stance phase to prevent slipping. On compliant surfaces, it increases foot penetration depth and modifies the ground reaction force profile. On rigid surfaces, it optimizes for energy efficiency. These adaptations are learned through RL in simulation with terrain randomization.

The simulation training environment includes a physics engine that models diverse terrain properties: friction coefficients from 0.1 (ice) to 1.5 (rubber), surface compliance from rigid concrete to soft sand, and various damping profiles. The terrain property estimator and locomotion adaptation module are trained jointly, ensuring the estimated properties are directly useful for locomotion control.

## Key Results & Numbers
- Online terrain property estimation converges within 5-10 steps (0.5-1 second at 10 Hz control)
- Unsupervised terrain classification accuracy of 85%+ across 6 terrain types (sand, gravel, grass, concrete, ice, mud)
- Locomotion stability improvement of 30-40% on deformable terrain compared to fixed-gait baselines
- Friction estimation error <15% after convergence on real hardware
- Real-world validation across 5 distinct terrain types with autonomous transitions
- No catastrophic forgetting when transitioning between terrain types due to incremental learning design

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The self-supervised terrain property learning is directly applicable to Mini Cheetah. The proprioceptive-based estimation avoids the need for expensive terrain sensors or labeled datasets. Mini Cheetah's 500 Hz PD control generates rich proprioceptive data that can be used for terrain estimation. The online adaptation capability is particularly valuable for outdoor deployment where terrain changes unpredictably. The framework can be integrated into Mini Cheetah's observation space — estimated terrain properties become additional inputs to the locomotion policy trained with PPO.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This paper is directly relevant to the CPTE (Contrastive Terrain Encoder) module in Project B. The unsupervised terrain property learning from proprioceptive data provides an alternative or complementary approach to contrastive learning for terrain encoding. The estimated terrain properties (friction, compliance, damping) can serve as the target representation for the CPTE, providing physically meaningful terrain features rather than abstract learned embeddings. The incremental online learning aspect is crucial for Cassie's deployment — the terrain encoder must adapt to new environments without retraining. The proprioceptive feature extraction can feed into the MC-GAT (GATv2 on kinematic tree) to provide terrain-aware node features.

## What to Borrow / Implement
- Implement the self-supervised terrain property estimator using proprioceptive data for both projects
- Use the estimated terrain properties as additional observation inputs to PPO policies
- Integrate terrain property estimates with the CPTE module in Project B as a grounding signal
- Adopt the incremental online learning mechanism for deployment-time adaptation
- Use the rolling window proprioceptive feature extraction for terrain classification
- Train the terrain estimator jointly with the locomotion policy in MuJoCo simulation
- Apply the terrain randomization strategy from the paper to improve sim-to-real transfer

## Limitations & Open Questions
- Unsupervised learning may produce terrain representations that are not physically interpretable
- Online adaptation requires careful learning rate tuning to avoid catastrophic forgetting
- Proprioceptive-only estimation may miss visual terrain features (gaps, edges) that don't manifest until contact
- The 5-10 step convergence time may be too slow for rapidly changing terrain
- Scaling to bipedal robots with fewer ground contact points (2 vs 4) may reduce estimation accuracy
- The self-supervised prediction objective may not capture all safety-relevant terrain properties
