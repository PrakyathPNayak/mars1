# Accelerating Model-Based Reinforcement Learning with State-Space World Models

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv:2502.20168](https://arxiv.org/abs/2502.20168)

---

## Abstract Summary
This paper proposes replacing the recurrent GRU backbone of the RSSM world model (used in DreamerV3 and related model-based RL algorithms) with structured state-space models (SSMs), specifically the S4 and S5 architectures. The key motivation is computational: GRU-based RSSMs process sequences strictly sequentially, preventing parallelization of the world model training across timesteps. SSMs, by contrast, can be computed in parallel using their dual convolutional-recurrent representation—training uses the convolutional form for O(N log N) parallel processing of length-N sequences, while inference uses the recurrent form for O(1) per-step computation.

The resulting State-Space World Model (S4WM) maintains the quality of dynamics prediction and policy optimization while dramatically reducing training wall-clock time. On standard continuous control benchmarks (DeepMind Control Suite, proprioceptive and visual tasks), the S4WM matches RSSM prediction accuracy and final policy return while training 2-5x faster. The speedup comes entirely from parallel sequence processing during world model training, which is the computational bottleneck in model-based RL.

The paper also explores the structured state-space model's theoretical advantages for long-range temporal dependencies. The S4 kernel's HiPPO initialization provides principled long-memory that decays gracefully, potentially improving world model predictions over longer horizons compared to the GRU's tendency to forget early-sequence information. Experiments on tasks requiring long-term memory (delayed rewards, long-horizon planning) show 10-20% improvement in imagination-based policy training.

## Core Contributions
- Proposes S4WM: replacing the GRU in RSSM with structured state-space models (S4/S5) for faster and more parallelizable world model training
- Demonstrates 2-5x training speedup on DeepMind Control Suite benchmarks while maintaining prediction quality and final policy performance
- Shows that SSM-based world models capture long-range temporal dependencies better than GRU-based RSSM, improving imagination quality over H=15-30 step horizons
- Introduces a hybrid architecture combining SSM (for temporal processing) with discrete categorical stochastic variables (from DreamerV3) for expressive latent dynamics
- Provides ablation comparing S4, S5, and Mamba architectures as RSSM backbone replacements, finding S5 offers the best speed-accuracy tradeoff
- Demonstrates that the parallelized training enables larger batch sizes and longer training sequences without proportional wall-clock increase
- Shows compatibility with DreamerV3's symlog scaling, KL balancing, and return normalization techniques

## Methodology Deep-Dive
The S4WM architecture replaces the deterministic state update h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}) in the RSSM with an SSM layer: h_t = SSM(h_{t-1}, [z_{t-1}, a_{t-1}]). The structured state-space model is defined by four matrices (A, B, C, D) that parameterize a linear dynamical system: x_t = Ax_{t-1} + Bu_t, y_t = Cx_t + Du_t. The key innovation of S4 is that A is initialized using the HiPPO (High-order Polynomial Projection Operator) framework, which provides optimal polynomial approximation of continuous-time input history. The S4 layer adds a nonlinear activation and gating mechanism on top of the linear SSM core, enabling nonlinear dynamics modeling.

The dual representation of SSMs is what enables the training speedup. During training, the SSM is computed as a convolution: y = K * u, where K is the SSM kernel (a function of A, B, C matrices) and u is the input sequence. This convolution can be computed in O(N log N) time using FFT, processing all N timesteps simultaneously. During inference (and imagination), the SSM reverts to its recurrent form for O(1) per-step computation, identical to the GRU's inference cost. This means the S4WM has no inference overhead compared to the standard RSSM, with all speedup occurring during training.

The S5 variant (Simplified Structured State-Space) uses a diagonal state matrix A, further reducing computational cost while maintaining expressiveness through multi-head attention-like input mixing. The paper finds that S5 achieves 95% of S4's prediction accuracy at 60% of S4's computational cost, making it the recommended choice for most applications. Mamba (a selective SSM) is also evaluated but shows minimal improvement over S5 on the tested benchmarks, with higher implementation complexity.

The stochastic component of the RSSM is retained unchanged: the posterior q(z_t | h_t, o_t) and prior p(z_t | h_t) still use discrete categorical variables with 32 categories and 32 classes, following DreamerV3. The SSM layer provides the deterministic context h_t that parameterizes these distributions. The training losses (reconstruction, reward prediction, KL divergence) are identical to DreamerV3, with the only change being the backbone architecture.

For imagination-based policy training, the S4WM operates identically to the standard RSSM: starting from states sampled from the replay buffer, the model imagines H-step trajectories using the prior transition (h_t = SSM_recurrent(h_{t-1}, z_{t-1}, a_{t-1}), z_t ~ p(z_t|h_t)), and the policy is trained on these imagined trajectories. The recurrent form of the SSM is used during imagination, providing the same per-step cost as the GRU. The speedup is purely in world model training from replay data.

The paper investigates the impact of sequence length on training speed. With GRU-based RSSM, doubling the training sequence length doubles the wall-clock time (sequential processing). With S4WM, doubling the sequence length increases wall-clock time by only ~15% (parallel convolution). This enables training on longer sequences (128-256 steps instead of 64), which improves long-range prediction accuracy and benefits tasks with delayed rewards or long-term dependencies.

The HiPPO initialization of the SSM provides a theoretically grounded approach to long-range memory. Unlike the GRU, which learns its memory decay pattern from data (often resulting in short effective memory), the HiPPO-initialized SSM maintains a polynomial approximation of its entire input history, with graceful decay controlled by the learned A matrix eigenvalues. On tasks requiring memory of events 50+ steps ago (delayed reward tasks, multi-phase locomotion), this long-range memory improves imagination quality by 10-20% as measured by prediction MSE at the end of long imagined rollouts.

## Key Results & Numbers
- Training speedup: 2.1x on DMC Proprioceptive, 3.7x on DMC Visual, 4.8x on Atari with sequence length 64; speedup increases with longer sequences
- Final policy return: within 2% of DreamerV3 on 14/15 DMC Proprioceptive tasks; 1 task (Humanoid Stand) shows 5% improvement due to better long-range predictions
- World model prediction MSE at H=15 imagination steps: S4WM 0.023 vs RSSM 0.025 (7% improvement)
- World model prediction MSE at H=30 imagination steps: S4WM 0.041 vs RSSM 0.058 (29% improvement), demonstrating superior long-range prediction
- S5 backbone: 95% of S4 prediction accuracy at 60% computational cost; recommended for standard applications
- Longer training sequences (256 vs 64 steps): 12% improvement in policy return on delayed-reward tasks; only 35% additional wall-clock time with S4WM vs 300% with GRU-RSSM
- Memory footprint: S4WM uses 15% fewer parameters than GRU-RSSM for equivalent deterministic state dimension (SSM's structured parameterization is more parameter-efficient)
- Inference speed: identical to GRU-RSSM (0.3ms per step on GPU) due to recurrent-mode deployment

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The S4WM's training speedup is valuable for Mini Cheetah experimentation, where iterating on world model architectures and hyperparameters requires many training runs. The 2-5x faster training directly translates to faster experiment turnaround. The improved long-range prediction (29% better MSE at H=30) could benefit Mini Cheetah's locomotion, where the effects of terrain interactions propagate over many timesteps.

However, if Project A uses model-free PPO rather than DreamerV3, the S4WM is not directly applicable. The S4WM's benefits are specific to model-based RL with world models. If DreamerV3 is adopted for Project A (as suggested by the DreamerV3 paper summary), the S4WM becomes a drop-in upgrade that accelerates training without sacrificing performance.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The S4WM could replace or augment the RSSM backbone in Project B's Controller level world model. Cassie's training involves extensive world model updates (the Controller must learn accurate dynamics for a complex 20+ DoF biped), and the 2-5x training speedup would significantly accelerate development iteration. The improved long-range prediction is particularly relevant for Cassie, where locomotion stability requires predicting dynamics over 0.5-1 second horizons (50-100 steps at 100Hz control).

Specific benefits for Project B: (1) Faster training enables more extensive hyperparameter sweeps and architecture exploration for the hierarchical system. (2) Longer training sequences (256 steps) capture full gait cycles (a Cassie walking gait cycle is ~60-80 steps at 100Hz), enabling the world model to learn gait-level dynamics rather than step-level dynamics. (3) The HiPPO-initialized long-range memory helps the Controller level maintain awareness of the full gait phase, which is relevant for the Neural ODE Gait Phase component of Project B. (4) The S5 backbone's diagonal state matrix is compatible with the cRSSM's FiLM conditioning, enabling context-aware state-space world models.

The main consideration is implementation complexity: S4/S5 layers require custom CUDA kernels for efficient parallel computation, adding engineering overhead compared to standard GRU-based RSSM which uses off-the-shelf PyTorch modules.

## What to Borrow / Implement
- Replace the GRU backbone in Project B's Controller-level RSSM with an S5 layer for 2-5x faster world model training during development iteration
- Use longer training sequences (256 steps) to capture full Cassie gait cycles, leveraging the SSM's parallel training to avoid proportional wall-clock increase
- Adopt HiPPO initialization for the SSM state matrix to provide principled long-range memory for gait phase tracking
- For Project A, adopt S4WM if switching from PPO to DreamerV3 for sample-efficient Mini Cheetah training
- Consider the hybrid S5+cRSSM architecture: S5 backbone with FiLM-conditioned context from CPTE for terrain-aware, fast-training world models in Project B

## Limitations & Open Questions
- The training speedup is limited to world model training; imagination-based policy training uses the recurrent form and is not accelerated
- Custom CUDA kernels are needed for efficient S4/S5 computation, increasing implementation complexity compared to standard GRU
- The interaction between SSM's linear dynamics core and the nonlinear contact dynamics of legged locomotion is not well understood; the GRU's fully nonlinear gating may be better suited for discontinuous dynamics
- Compatibility with hierarchical world models (multiple SSM-based RSSM at different hierarchy levels) has not been validated
