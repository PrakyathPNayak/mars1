# Learning Latent Dynamics for Planning from Pixels (PlaNet)

**Authors:** Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee
**Year:** 2019 | **Venue:** ICML 2019
**Links:** https://arxiv.org/abs/1811.04551

---

## Abstract Summary
PlaNet introduces the Recurrent State-Space Model (RSSM) that combines deterministic and stochastic latent states for learning environment dynamics from pixels. Plans are executed in latent space using the cross-entropy method (CEM), achieving strong performance on continuous control tasks with 50x less environment interaction than model-free methods. This foundational work established the architecture that all subsequent Dreamer variants build upon.

## Core Contributions
- Introduced the Recurrent State-Space Model (RSSM), combining deterministic recurrent paths with stochastic latent variables for dynamics modeling
- Demonstrated that online planning in learned latent spaces (via CEM) can solve continuous control tasks from raw pixels
- Achieved 50x sample efficiency improvement over model-free baselines (A3C) on DeepMind Control Suite tasks
- Showed that purely deterministic or purely stochastic models are insufficient — the hybrid RSSM design is critical
- Established the variational inference framework for training world models with observation reconstruction, reward prediction, and KL regularization
- Solved 6 continuous control tasks (cartpole, reacher, cheetah, finger, cup, walker) directly from 64×64 pixel observations

## Methodology Deep-Dive
PlaNet's central contribution is the RSSM architecture, which models environment dynamics in a learned latent space. The key insight is that neither purely deterministic models (which cannot represent stochastic environments) nor purely stochastic models (which struggle to remember information over long horizons) are sufficient. The RSSM combines both: a deterministic recurrent state (GRU hidden state) that maintains long-term memory, and a stochastic latent variable that captures per-step uncertainty and multimodality.

The RSSM defines four components: (1) a deterministic state transition model that updates the GRU hidden state given the previous state and action, (2) a stochastic prior that predicts the next latent state from the deterministic state alone (used during planning when observations are unavailable), (3) a stochastic posterior that incorporates the actual observation to correct the prior (used during training), and (4) observation and reward decoders that reconstruct outputs from the combined deterministic-stochastic state. Training minimizes a variational bound combining reconstruction loss, reward prediction loss, and KL divergence between posterior and prior.

For planning, PlaNet uses the Cross-Entropy Method (CEM), a derivative-free optimization algorithm. At each timestep, CEM samples a population of action sequences, evaluates them by rolling out the world model in latent space, selects the top performers, refits a Gaussian distribution to the elites, and repeats for several iterations. Only the first action of the best sequence is executed, and replanning occurs at every step (model predictive control). This avoids the need to train a separate policy network but incurs computational cost at inference time.

The observation encoder is a convolutional neural network that maps 64×64 RGB images to latent vectors. The decoder mirrors this architecture to reconstruct images, providing a training signal for learning meaningful latent representations. The reward predictor is a small MLP that estimates expected reward from the latent state, enabling reward-based planning without decoding back to pixel space.

A key ablation study demonstrates the necessity of the hybrid architecture: models with only deterministic states fail to capture stochastic dynamics, while models with only stochastic states suffer from compounding errors over long horizons. The RSSM's hybrid design achieves the best of both worlds, with the deterministic path providing a stable backbone and the stochastic path capturing environmental uncertainty.

## Key Results & Numbers
- 50x more sample-efficient than A3C on DeepMind Control Suite tasks from pixels
- Solved 6 continuous control tasks: cartpole balance/swingup, reacher easy, cheetah run, finger spin, cup catch, walker walk
- Planning horizon of 12 steps with CEM (1000 candidates, 100 elites, 10 iterations)
- Latent state dimension: 30 stochastic + 200 deterministic (GRU)
- Training on 100-500 episodes sufficient for strong performance
- Ablation: RSSM > deterministic-only > stochastic-only, confirming hybrid design necessity
- Real-time planning at ~10 Hz (CEM optimization takes ~100ms per step)

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The RSSM architecture is foundational for world model approaches applicable to Mini Cheetah locomotion. While PlaNet itself uses CEM planning (too slow for 500 Hz control), the RSSM dynamics model could be trained on proprioceptive locomotion data and used for trajectory optimization at a higher level. The 50x sample efficiency gain, if achievable for locomotion, would significantly reduce the MuJoCo simulation budget. The CEM planner could be useful for longer-horizon motion planning (e.g., planning footstep sequences) while low-level PD control handles execution. However, the pixel-based approach is less relevant — Mini Cheetah primarily uses proprioceptive observations.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The RSSM is the backbone of Project B's world model in the planner level, making PlaNet the foundational paper for the entire hierarchical architecture. The planner uses the RSSM to imagine future states and plan locomotion trajectories, directly following PlaNet's framework. The deterministic-stochastic decomposition is especially relevant for bipedal locomotion: the deterministic path can capture the cyclic gait pattern while the stochastic path models contact uncertainty and terrain variability. The variational training framework (reconstruction + KL) established here is exactly how the planner's world model would be trained. While Project B replaces CEM with a learned policy (Dreamer-style), the core RSSM architecture and training procedure from PlaNet remain central.

## What to Borrow / Implement
- Implement the RSSM architecture (deterministic GRU + stochastic latent) as the core dynamics model for both projects
- Use the variational training framework (reconstruction loss + reward prediction + KL divergence) for world model training
- Adopt the hybrid state representation design (deterministic for memory, stochastic for uncertainty) rather than purely one or the other
- Consider CEM planning for high-level trajectory optimization (footstep planning) while using learned policies for low-level control
- Use the ablation methodology (compare hybrid vs. deterministic-only vs. stochastic-only) to validate architectural choices
- Start with latent dimensions similar to PlaNet (30 stochastic + 200 deterministic) and tune from there

## Limitations & Open Questions
- CEM planning is too slow for high-frequency control (500 Hz) — requires a learned policy for real-time execution
- Pixel-based observations are unnecessary for proprioceptive locomotion, adding computational overhead
- No mechanism for adapting to changing dynamics (unlike later works like HiP-RSSM)
- Planning horizon limited to ~12 steps due to compounding model error
- No explicit contact modeling — the world model must implicitly learn contact dynamics from data
- Single-task learning — no transfer or generalization across different locomotion behaviors
- KL regularization can be unstable without later innovations like free bits and KL balancing (addressed in DreamerV3)
