# Privileged Sensing Scaffolds Reinforcement Learning (Scaffolder)

**Authors:** Penn PAL Lab
**Year:** 2024 | **Venue:** arXiv/GitHub
**Links:** [GitHub](https://github.com/penn-pal-lab/scaffolder)

---

## Abstract Summary
Scaffolder introduces the Sensory Scaffolding Suite (S3), a comprehensive benchmark and algorithmic framework for training RL agents that leverage privileged sensory information during training but deploy with only a subset of sensors. The key metaphor is "scaffolding" — just as construction scaffolding supports a building during construction but is removed afterward, privileged sensors support the learning process but are removed at deployment. The paper provides a standardized evaluation protocol across eight tasks spanning locomotion (quadruped rough terrain, biped balance), manipulation (dexterous grasping, tool use), and navigation (maze solving, visual navigation), enabling fair comparison of privileged information methods.

The core algorithmic contribution is a world model-based scaffolding approach: a Dreamer-style RSSM world model is trained with access to all sensors (privileged + deployable), learning a rich latent dynamics model that captures information from privileged sensors. During imagination-based policy training, the policy is conditioned only on deployable sensor reconstructions from the latent state, effectively learning to act based on information that privileged sensors help encode into the latent space but that is ultimately grounded in deployable observations. At deployment, the world model's encoder is replaced with a deployable-only encoder, and the latent space retains structure learned from privileged sensors.

The benchmark reveals that naive approaches (training only with deployable sensors, or distilling from a privileged policy) are suboptimal compared to scaffolding, especially in tasks requiring long-horizon reasoning about unobserved state. Scaffolder achieves the best performance on 6/8 S3 tasks and establishes a strong baseline for future privileged information research.

## Core Contributions
- Introduction of the Sensory Scaffolding Suite (S3), a standardized 8-task benchmark for privileged information RL
- World model-based scaffolding approach using RSSM trained with all sensors but deploying with subset
- Demonstration that world model latent space can encode privileged information accessible through deployable observations
- Comparison of scaffolding against 6 baselines: end-to-end, asymmetric critic, teacher-student, DAgger, RMA, and privileged oracle
- Finding that scaffolding is most beneficial for tasks requiring inference of hidden state (terrain type, object properties)
- Open-source implementation with standardized evaluation protocol for reproducible research
- Analysis of latent space structure showing privileged sensor information is geometrically preserved after encoder swap

## Methodology Deep-Dive
The Scaffolder architecture builds on DreamerV3 with a key modification: the observation encoder is split into a privileged encoder E_priv and a deployable encoder E_dep. During training, the RSSM receives observations from both encoders: the latent state h_t is updated using features from E_priv(o_priv_t) and E_dep(o_dep_t) concatenated before the recurrent update. The RSSM's generative model learns to predict both privileged and deployable observations from the latent state, with separate decoder heads: D_priv(h_t, z_t) → ô_priv and D_dep(h_t, z_t) → ô_dep. The privileged decoder ensures the latent space encodes privileged information, while the deployable decoder ensures this information is expressed in terms of deployable observations.

During imagination-based actor-critic training (Dreamer's standard approach), the actor π(a_t | h_t, z_t) and critic V(h_t, z_t) operate on the latent state that was trained with privileged information. However, the actor is explicitly constrained: it receives only the "deployable projection" of the latent state, computed as h_t^dep = MLP(E_dep(D_dep(h_t, z_t))). This ensures the actor learns to act based on information recoverable from deployable observations, even though the latent dynamics benefited from privileged sensors during world model training.

At deployment, the RSSM runs in real-time with only E_dep active. The key finding is that the deployable encoder, trained alongside the privileged encoder in a shared latent space, produces latent representations that capture much of the structural information originally provided by privileged sensors. Probing experiments show that the deployable latent h_t^dep has a linear relationship with privileged features (terrain height, friction, contact forces) with R² values of 0.6–0.8, despite never directly observing these quantities.

The S3 benchmark tasks are designed to systematically vary the type of privileged information: (1) Quadruped Terrain: privileged = terrain heightmap, deploy = proprioception; (2) Biped Balance: privileged = center of mass velocity, deploy = joint angles + IMU; (3) Dexterous Grasp: privileged = object pose + contact forces, deploy = tactile + proprioception; (4-8) additional tasks varying spatial, temporal, and physical privileged information types. Each task has standardized train/eval splits, reward functions, and domain randomization parameters.

The world model training objective is: L = L_recon_priv + L_recon_dep + L_reward + L_continuation + L_KL, following DreamerV3's formulation with the addition of the privileged reconstruction loss. The privileged reconstruction loss is weighted equally with the deployable reconstruction loss (both at weight 1.0), ensuring the latent space is not dominated by either information source. The actor-critic is trained using Dreamer's standard λ-return objective on imagined trajectories from the scaffolded world model.

## Key Results & Numbers
- Scaffolder achieves best performance on 6/8 S3 benchmark tasks, outperforming the next best method (asymmetric critic) by 15–30% average return
- Quadruped rough terrain: Scaffolder 820 vs. asymmetric critic 690 vs. end-to-end 520 (normalized return)
- Biped balance: Scaffolder 750 vs. teacher-student 700 vs. end-to-end 480
- Deployable latent probing: terrain height R²=0.78, friction R²=0.65, contact force R²=0.72 (linear decoder on h_t^dep)
- World model training with privileged sensors adds 20% compute overhead vs. deployable-only Dreamer
- Policy transfer from scaffolded to deployment encoder retains 90–95% of scaffolded performance
- Scaffolding advantage is largest for tasks with high privileged/deployable information gap (terrain navigation: +45%, simple manipulation: +8%)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Scaffolder provides a standardized framework for evaluating the Mini Cheetah's privileged training pipeline. The quadruped rough terrain task in S3 is directly comparable to Mini Cheetah's terrain locomotion objective. The world model-based scaffolding approach offers an alternative to the current teacher-student pipeline: instead of distilling a privileged policy, train a scaffolded world model that encodes terrain information into the latent space and deploy with proprioceptive-only encoder. The S3 benchmark enables standardized comparison against other privileged information methods.

The deployable latent probing results (terrain R²=0.78) suggest that the Mini Cheetah's proprioceptive encoder can learn significant terrain awareness through scaffolding, potentially rivaling explicit terrain estimation modules.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Scaffolder's world model-based approach is directly applicable to Cassie's RSSM-based Planner. The Planner already uses a Dreamer-style world model; adding privileged sensor scaffolding (terrain heightmap, contact forces) during training and removing it at deployment aligns perfectly with the existing architecture. The S3 biped balance task validates the approach on bipedal platforms.

The scaffolding concept also applies to Cassie's hierarchical structure: privileged information can scaffold different levels — terrain maps scaffold the Planner's world model, contact forces scaffold the Controller's dynamics model, and obstacle information scaffolds the Safety layer's CBF. This multi-level scaffolding is a natural extension of the S3 framework.

## What to Borrow / Implement
- Use S3 benchmark tasks as standardized evaluation for Mini Cheetah and Cassie privileged information approaches
- Implement world model scaffolding in Cassie's RSSM Planner: train with terrain + contact privileged encoders, deploy with proprioceptive-only
- Adopt the dual-encoder (privileged + deployable) architecture for Mini Cheetah's Dreamer-style world model if model-based components are added
- Add deployable latent probing experiments to verify that proprioceptive encoders learn implicit privileged feature representations
- Compare scaffolding against current teacher-student distillation pipeline on both platforms

## Limitations & Open Questions
- World model-based scaffolding requires a Dreamer-style MBRL setup, which may not be applicable to purely model-free PPO pipelines without modification
- The deployable encoder swap at deployment introduces a distribution shift in the latent space that may degrade long-horizon predictions
- S3 benchmark tasks are simulated; real-world sim-to-real transfer with scaffolding is not evaluated
- Multi-level scaffolding (different privileged info at different hierarchy levels) is not explored in the paper
