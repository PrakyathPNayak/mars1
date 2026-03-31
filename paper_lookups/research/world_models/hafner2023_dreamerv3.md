# Mastering Diverse Domains through World Models (DreamerV3)

**Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap
**Year:** 2023 | **Venue:** Nature (also arXiv)
**Links:** [Nature](https://www.nature.com/articles/s41586-025-08744-2)

---

## Abstract Summary
DreamerV3 represents a landmark achievement in model-based reinforcement learning: a single algorithm with fixed hyperparameters that masters over 150 tasks across diverse domains including continuous control, Atari games, DMLab 3D navigation, Minecraft, and robotic manipulation. The algorithm learns a world model using the Recurrent State-Space Model (RSSM) architecture that compresses observations into a compact latent state combining deterministic (recurrent) and stochastic (discrete categorical) components. The policy and value function are then trained entirely within the learned world model through imagined rollouts, never directly interacting with the environment during policy optimization.

The key innovations enabling domain-general performance are: (1) symlog scaling of predictions and rewards, which normalizes the learning signal across domains with vastly different reward magnitudes; (2) KL balancing between the prior and posterior in the RSSM, which prevents both posterior collapse and prior divergence; and (3) policy entropy regularization with a return-normalized objective, which maintains exploration across domains with different reward densities. These three techniques together eliminate the need for domain-specific hyperparameter tuning, a persistent challenge in model-based RL.

DreamerV3 demonstrates that world model-based RL can match or exceed model-free methods (PPO, SAC, Rainbow) on their home domains while requiring 10-50x fewer environment interactions. On the challenging Minecraft diamond benchmark, DreamerV3 is the first algorithm to collect a diamond from scratch without human demonstrations or curriculum design, demonstrating long-horizon planning capability through learned world models.

## Core Contributions
- Achieves state-of-the-art performance across 150+ tasks in 7 domains with a single set of hyperparameters, demonstrating unprecedented generality in RL
- Introduces symlog scaling (symlog(x) = sign(x) * log(|x| + 1)) for predictions and rewards, normalizing learning signals across domains with reward magnitudes ranging from 0.01 to 10,000+
- Proposes KL balancing with separate learning rates for the RSSM prior and posterior, preventing both posterior collapse (prior dominates) and unbounded latent complexity (posterior diverges)
- Introduces return normalization for the policy objective using percentile-based scaling (5th and 95th percentiles of imagined returns), enabling consistent policy gradients across reward scales
- Demonstrates that RSSM world models with discrete categorical latent variables (32 categories x 32 classes) outperform continuous Gaussian latent variables for diverse domain representation
- First algorithm to collect a diamond in Minecraft from scratch without demonstrations, requiring approximately 100-step plans through the crafting tree
- Provides comprehensive ablation showing each component contributes 10-30% performance improvement on the hardest benchmarks

## Methodology Deep-Dive
The RSSM world model at the heart of DreamerV3 maintains a latent state s_t = (h_t, z_t) with two components. The deterministic component h_t is updated by a GRU recurrence: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), capturing long-range temporal dependencies. The stochastic component z_t is a discrete categorical variable (32 categories, each with 32 classes, giving 32^32 possible states) sampled from a learned posterior q(z_t | h_t, o_t) during training and from a learned prior p(z_t | h_t) during imagination. The posterior uses the observation o_t for grounding, while the prior must predict z_t from the deterministic context alone.

The world model is trained on sequences from the replay buffer using three losses. The reconstruction loss trains the decoder to predict observations from the latent state: L_recon = -log p(o_t | h_t, z_t). The reward loss trains a reward predictor: L_reward = -log p(r_t | h_t, z_t). The KL loss aligns the prior with the posterior: L_KL = KL[q(z_t|h_t,o_t) || p(z_t|h_t)]. The KL balancing technique weights this loss asymmetrically: the prior is trained 10x faster than the posterior to prevent the posterior from collapsing to the prior. Additionally, a free-bits threshold (KL >= 1 nat) prevents the KL term from forcing the latent to be completely uninformative.

Symlog scaling addresses the challenge of reward and observation magnitudes varying by orders of magnitude across domains. All predictions (observations, rewards, values) are made in symlog space: the network predicts symlog(x) and the loss is computed on symlog(x). The inverse mapping (exp(|y|) - 1) * sign(y) recovers the original scale. This logarithmic compression prevents large-magnitude domains from dominating the gradient while maintaining sensitivity in small-magnitude domains.

Policy learning occurs entirely within the world model. Starting from states sampled from the replay buffer, the RSSM imagines H=15 step trajectories using the prior transition model and the current policy. The policy is trained to maximize imagined returns using an actor-critic architecture. The actor loss uses reinforce with baseline: L_actor = -E[sum_t (R_t - V(s_t)) * log pi(a_t|s_t)], where R_t is the lambda-return and V is the learned value function. The returns are normalized using running percentiles: R_norm = (R - perc_5) / (perc_95 - perc_5), ensuring consistent gradient magnitudes. Entropy regularization H(pi(.|s_t)) is added with coefficient 3e-4 to maintain exploration.

The architecture uses an encoder (CNN for images, MLP for proprioceptive) that maps observations to a feature vector, which together with h_t parameterizes the posterior distribution over z_t. The decoder mirrors the encoder. The GRU for the deterministic state h_t has 1024 units. The policy and value networks are MLPs with 512-unit hidden layers. All networks use LayerNorm and SiLU activations. Training uses Adam with learning rate 1e-4 and batch size 16 sequences of length 64.

For robotics applications, DreamerV3 processes proprioceptive observations (joint angles, velocities, body IMU) through the MLP encoder. The RSSM learns a dynamics model in latent space that captures the robot's response to actions, and the policy is optimized through imagined rollouts of this dynamics model. This is sample-efficient because the policy explores vast numbers of scenarios in imagination without requiring environment interaction for each.

## Key Results & Numbers
- 150+ tasks mastered with fixed hyperparameters across: Atari 100K (26 games), Atari 200M (57 games), DMC proprioceptive (15 tasks), DMC visual (20 tasks), DMLab (8 tasks), Minecraft (1 task), and BSuite (23 tasks)
- Sample efficiency: 10-50x fewer environment steps than PPO on DMC tasks; matches PPO performance at 1M steps vs PPO's 10M steps
- Minecraft diamond: first algorithm to achieve this from scratch; requires ~100 sequential crafting steps over 36,000 environment steps
- DMC proprioceptive tasks: achieves 95-100% of maximum score on all 15 tasks within 5M environment steps
- RSSM latent state: 32x32 discrete categorical (1024 possible latent values per category) achieves 8-15% higher performance than 64-dim continuous Gaussian latent
- Symlog scaling adds zero computational overhead and improves performance by 10-25% on domains with large reward variance
- KL balancing prevents posterior collapse in 100% of runs (vs 15% collapse rate without balancing)
- Training speed: approximately 15 environment steps per second on 1 NVIDIA V100 for image-based domains; 100+ steps per second for proprioceptive domains

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
DreamerV3 offers a compelling alternative to the model-free PPO approach currently planned for Mini Cheetah. The RSSM world model could dramatically reduce the number of MuJoCo simulation steps needed to train effective locomotion policies, as the policy is optimized through imagined rollouts rather than direct environment interaction. The DMC proprioceptive results (95-100% score at 5M steps) suggest that DreamerV3 can handle the Mini Cheetah's proprioceptive observation space efficiently.

Practical considerations for Project A: (1) DreamerV3's fixed hyperparameters eliminate extensive tuning needed for PPO on locomotion tasks. (2) The world model provides an implicit dynamics model that can be used for model-predictive control at deployment time, complementing the learned policy. (3) The imagination-based training naturally supports domain randomization: the world model can be conditioned on randomized dynamics parameters, and the policy is optimized across all imagined scenarios simultaneously.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
DreamerV3's RSSM is a core component of Project B's architecture. The Controller level uses an RSSM-based world model to learn Cassie's dynamics in latent space, enabling the Controller to plan short-horizon motor commands through imagination. The specific RSSM design choices validated in DreamerV3 should be adopted directly.

Key design decisions from DreamerV3 to apply to Project B: (1) Use discrete categorical latent variables (32x32) rather than continuous Gaussians for the RSSM stochastic component, as this representation is more expressive and stable. (2) Apply KL balancing with 10x asymmetric learning rates to prevent posterior collapse during the Controller's world model training. (3) Use symlog scaling for Cassie's heterogeneous reward signals (tracking rewards in radians, velocity rewards in m/s, energy rewards in Watts), which span different magnitudes. (4) The H=15 step imagination horizon provides a template for the Controller level's planning horizon in Project B. (5) The return normalization via percentiles ensures stable policy gradients despite the multi-objective reward structure in the hierarchical system.

The main adaptation needed is integrating the RSSM with Project B's hierarchical architecture: the Controller's RSSM should receive commands from the Primitives level as goals, not just the raw proprioceptive state. This conditioning mechanism is not present in standard DreamerV3 but can be achieved by concatenating the primitive command with the RSSM's input.

## What to Borrow / Implement
- Adopt the DreamerV3 RSSM architecture (GRU + 32x32 discrete categorical latent) as the world model backbone for Project B's Controller level
- Implement symlog scaling for all prediction targets in both projects to handle heterogeneous reward/observation magnitudes
- Use KL balancing with 10x asymmetric learning rates and 1-nat free-bits threshold for stable RSSM training
- Apply return normalization via running 5th/95th percentiles for consistent policy gradients in both projects
- For Project A, evaluate DreamerV3 as an alternative to PPO for sample-efficient Mini Cheetah training in MuJoCo

## Limitations & Open Questions
- DreamerV3's imagination-based training assumes the world model is accurate; for contact-rich locomotion (leg impacts, foot slip), RSSM prediction errors may accumulate over the H=15 step horizon
- The single-hyperparameter-set claim holds across domains but within-domain performance is sometimes 5-10% below domain-tuned baselines
- Real-time deployment requires running the RSSM forward model at control frequency; the GRU's sequential nature may limit inference speed on embedded hardware
- Integration with hierarchical RL architectures (multiple RSSM levels) is not addressed; interactions between world models at different hierarchy levels remain unexplored
