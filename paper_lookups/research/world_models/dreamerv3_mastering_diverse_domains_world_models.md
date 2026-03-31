# Mastering Diverse Domains through World Models (DreamerV3)

**Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap
**Year:** 2023 | **Venue:** arXiv / JMLR (under review)
**Links:** https://arxiv.org/abs/2301.04104

---

## Abstract Summary
DreamerV3 is a general model-based RL algorithm that learns a world model and trains policies entirely within imagined trajectories. It achieves strong performance across 150+ tasks spanning 7 domains (Atari, DMC, Minecraft, robotics) using a single set of hyperparameters, via symlog predictions, free bits KL balancing, and percentile-scaled returns. This eliminates the need for per-domain hyperparameter tuning, making it a truly general-purpose agent.

## Core Contributions
- Introduced symlog predictions that normalize targets across vastly different reward scales, enabling a single configuration to work across all domains
- Developed KL balancing with free bits that prevents posterior collapse while maintaining informative latent states
- Proposed percentile-scaled returns (return normalization via running percentiles) that remove the need for reward clipping or manual scaling
- Unified architecture achieving state-of-the-art or competitive results on 150+ tasks across 7 benchmark domains with zero hyperparameter tuning
- First algorithm to collect diamonds in Minecraft from scratch without human demonstrations or curricula
- Demonstrated that model-based RL can be fully general-purpose, matching or exceeding specialized algorithms in every tested domain
- Open-sourced implementation enabling reproducibility across the full suite of benchmarks

## Methodology Deep-Dive
DreamerV3 builds on the Dreamer family of model-based RL agents, centered around the Recurrent State-Space Model (RSSM). The world model consists of an encoder, a sequence model (GRU-based recurrence), and decoder heads for reconstructing observations, rewards, and episode continuation signals. The RSSM maintains both deterministic (recurrent) and stochastic (categorical) latent states, allowing the model to capture both predictable dynamics and environmental stochasticity. Training proceeds by encoding real experience into latent states and optimizing the world model via variational inference.

The key innovation is a set of robust normalization techniques. Symlog predictions apply a symmetric logarithmic transform to prediction targets, compressing the dynamic range of values the network must predict. This is critical because reward magnitudes vary by orders of magnitude across domains (e.g., Atari scores vs. DMC returns). By predicting in symlog space and inverting for actual values, the same network capacity handles all scales gracefully.

KL balancing with free bits addresses the classic VAE training dilemma. The KL divergence loss is split asymmetrically: 80% of the gradient updates the posterior toward the prior, and 20% updates the prior toward the posterior. Additionally, a free bits threshold of 1 nat prevents the KL term from pushing the latent distribution to be completely uninformative. This balance ensures the model learns rich latent representations without posterior collapse.

Policy learning occurs entirely in imagination. The actor-critic is trained on trajectories imagined by the world model, using percentile-scaled returns. Rather than normalizing returns by a fixed statistic, DreamerV3 tracks the 5th and 95th percentiles of returns in a running buffer and scales advantages accordingly. This removes sensitivity to absolute reward magnitude and provides stable policy gradients across domains.

The architecture uses discrete categorical latent variables (32 categories × 32 classes) rather than continuous Gaussians, which empirically improves training stability and representation quality. The full system alternates between collecting real environment data, training the world model on replay buffer data, and training the actor-critic on imagined trajectories — a clean separation that enables efficient learning.

## Key Results & Numbers
- First algorithm to collect diamonds in Minecraft from scratch (no demos, no curriculum)
- Matches or exceeds domain-specialized algorithms across all 7 benchmark suites (Atari 100K, Atari 200M, DMControl, BSuite, Crafter, DMLab, Minecraft)
- Single hyperparameter configuration used across all 150+ tasks with no per-task tuning
- On DMControl proprioceptive tasks: competitive with SAC and TD-MPC using ~10x less environment interaction
- On Atari 200M: outperforms Rainbow, IQN, and other model-free baselines
- Crafter: achieves highest reported score, unlocking all 22 achievements

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
DreamerV3's world model approach could dramatically improve sample efficiency for Mini Cheetah training. Currently, PPO requires millions of environment steps in MuJoCo simulation; a world model could reduce this by 10-50x by generating imagined training data. The RSSM architecture is directly relevant to learning locomotion dynamics — the deterministic path captures smooth dynamics while the stochastic path models contact events and terrain variability. Symlog predictions would handle the varying reward scales between velocity tracking, energy penalty, and stability terms without manual tuning. The single-hyperparameter promise means less engineering effort when experimenting with different reward formulations or domain randomization settings. The 500 Hz PD control loop could be modeled efficiently in the world model's latent space rather than requiring full physics simulation at every step.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
RSSM is a core component in Project B's planner level, making DreamerV3's stability tricks (symlog, free bits, percentile returns) directly applicable to training the world model component. The planner's RSSM/Dreamer module would benefit from all three normalization innovations — symlog for handling diverse terrain reward signals, free bits for maintaining informative latent terrain representations, and percentile returns for stable planning across different locomotion modes. The categorical latent space (32×32) could encode discrete terrain types and gait phases naturally. DreamerV3's demonstration of zero-hyperparameter-tuning generalization supports the goal of a single hierarchical policy that handles all of Cassie's locomotion scenarios. The imagined trajectory training is directly how the planner level would generate candidate motion plans for the primitives layer to execute.

## What to Borrow / Implement
- Implement symlog prediction heads in the RSSM world model for both projects to handle multi-scale rewards
- Adopt the free bits (1 nat) + KL balancing (80/20 split) for stable RSSM training without posterior collapse
- Use percentile-scaled returns (5th/95th) as the default return normalization in both PPO and imagined trajectory training
- Port the 32×32 categorical latent space design for discrete representation of terrain and gait states
- Use DreamerV3's replay buffer strategy (uniform sampling with sequence chunks) for efficient world model training
- Benchmark world model sample efficiency against current PPO pipeline on Mini Cheetah locomotion tasks

## Limitations & Open Questions
- Imagined trajectories may diverge from reality for long horizons, especially during dynamic locomotion with frequent contacts
- Computational overhead of world model training may not be justified if simulation is already fast (e.g., MuJoCo at 10K+ FPS)
- Categorical latents may not capture the continuous nature of joint angles and contact forces as well as Gaussian latents
- No explicit mechanism for sim-to-real transfer — world model learned in sim may not transfer directly
- Unclear how well the single-hyperparameter promise holds for high-frequency control (500 Hz) with real-time constraints
- Model error compounds during long imagination horizons, potentially leading to exploitable artifacts in the learned policy
