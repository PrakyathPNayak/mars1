# Decision Transformer: Reinforcement Learning via Sequence Modeling

**Authors:** Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch
**Year:** 2021 | **Venue:** NeurIPS
**Links:** https://arxiv.org/abs/2106.01345

---

## Abstract Summary
Decision Transformer (DT) fundamentally reframes reinforcement learning as a conditional sequence modeling problem. Rather than fitting value functions or computing policy gradients, DT leverages the GPT architecture to autoregressively model trajectories represented as sequences of return-to-go tokens, states, and actions. By conditioning on a desired future return, the model learns to generate actions that achieve the specified cumulative reward, effectively bypassing the need for temporal difference (TD) learning, bootstrapping, or dynamic programming.

The approach is evaluated on the offline RL benchmarks from D4RL, including Atari, OpenAI Gym locomotion tasks (HalfCheetah, Hopper, Walker2d), and Key-to-Door environments. Decision Transformer matches or exceeds the performance of state-of-the-art purpose-built offline RL algorithms such as Conservative Q-Learning (CQL) and Behavior Regularized Actor-Critic (BRAC), while being conceptually simpler and more stable to train.

A particularly noteworthy finding is that DT can generalize across return levels—by conditioning on higher returns at test time, it can extrapolate beyond the average performance seen in the training dataset, effectively "stitching" together high-performing sub-trajectories from sub-optimal demonstrations. This capability makes it a compelling framework for leveraging heterogeneous offline datasets.

## Core Contributions
- Recasts RL as an autoregressive sequence modeling problem, removing the need for TD learning, value function fitting, or dynamic programming
- Introduces return-conditioned policy generation: actions are predicted based on desired future return-to-go, past states, and past actions
- Demonstrates that a minimally modified GPT-2 architecture can serve as an effective RL policy
- Achieves competitive or superior performance to CQL, BRAC, and other offline RL methods on D4RL locomotion and Atari benchmarks
- Shows that conditioning on higher-than-observed returns at test time can lead to extrapolation beyond dataset quality
- Establishes that sequence modeling naturally handles long-horizon credit assignment without explicit reward propagation
- Opens a new research direction connecting large language model architectures to decision-making problems

## Methodology Deep-Dive
The Decision Transformer architecture takes as input a trajectory represented as an interleaved sequence of three token types: return-to-go (R̂_t), state (s_t), and action (a_t). At each timestep t, the return-to-go is defined as R̂_t = Σ_{t'=t}^{T} r_{t'}, representing the sum of future rewards from timestep t onward. The model is trained to predict the action a_t given the subsequence (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_t, s_t).

The architecture is a causal transformer (GPT-2) with a context window of K timesteps. Each modality (return-to-go, state, action) is projected into the transformer's embedding dimension via learned linear layers. Positional embeddings are shared across the three tokens within the same timestep, encoding temporal position rather than token position within the sequence. This design ensures the model understands temporal alignment across modalities.

Training is performed via supervised learning on offline datasets. The loss function is simply the mean-squared error (for continuous actions) or cross-entropy (for discrete actions) between predicted and ground-truth actions. This simplicity is a major advantage over offline RL methods that require careful tuning of pessimism coefficients, conservative regularization, or importance sampling corrections.

At inference time, the desired return-to-go is set to a target value (e.g., the maximum episode return observed in the dataset or higher). As the agent acts in the environment and receives rewards, the return-to-go is decremented accordingly: R̂_{t+1} = R̂_t - r_t. This mechanism naturally guides the policy to pursue the specified cumulative reward.

The context length K is a critical hyperparameter. Longer contexts allow the model to attend to more history, improving performance on tasks requiring long-term memory. The authors find K=20 works well for Gym locomotion tasks, while Atari tasks benefit from longer contexts. The total number of tokens per forward pass is 3K (three tokens per timestep × K timesteps).

## Key Results & Numbers
- HalfCheetah-Medium-v2: DT achieves 42.6 (normalized return), vs. CQL 44.0, BRAC 46.3
- HalfCheetah-Medium-Expert-v2: DT achieves 86.8, outperforming CQL (62.4) and BRAC (41.9)
- Hopper-Medium-v2: DT 67.6 vs. CQL 58.0
- Walker2d-Medium-v2: DT 74.0 vs. CQL 79.2
- On Atari (Breakout, Qbert, Pong, Seaquest), DT outperforms CQL on 3 out of 4 games
- Context length of K=20 is sufficient for locomotion; Atari requires K=30-50
- Training is stable and converges within ~100K gradient steps on locomotion tasks
- No hyperparameter sensitivity to pessimism/conservatism coefficients (which plague CQL/BRAC)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
Decision Transformer provides a viable offline RL framework for learning Mini Cheetah locomotion policies from pre-collected demonstration data. If the project has access to datasets of Mini Cheetah trajectories (e.g., from model-based controllers, human teleoperation, or prior RL policies), DT could be used to train policies without online interaction. The return-conditioning mechanism is attractive for specifying different performance targets (e.g., energy-efficient vs. fast locomotion).

However, the primary limitation is that DT operates purely offline and does not directly support online fine-tuning or sim-to-real adaptation. For Mini Cheetah, where sim-to-real transfer is critical, DT would likely serve as an initialization method rather than the complete training pipeline. The lack of explicit safety constraints is another concern for real hardware deployment.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Decision Transformer's sequence modeling paradigm is directly relevant to the Dual Asymmetric-Context Transformer (DACT) architecture in the Cassie project. DT demonstrates that transformers can effectively model decision-making trajectories, validating the core architectural choice of using attention mechanisms for policy learning. The return-conditioning mechanism could be adapted for Cassie's Planner level, where conditioning on desired locomotion objectives (speed, direction, terrain type) determines the high-level plan.

For the skill library construction in Cassie's Primitives level, DT's offline learning capability is valuable. Pre-collected trajectories of walking, turning, crouching, and recovery behaviors could be distilled into a transformer-based skill library. The ability to condition on different returns enables learning diverse skill variants from the same dataset. The hierarchical extension of DT (see Paper 8) would further enhance applicability to Cassie's 4-level hierarchy.

## What to Borrow / Implement
- Adopt the return-to-go conditioning mechanism for specifying locomotion objectives at Cassie's Planner level
- Use the GPT-2 backbone architecture as a starting point for the DACT transformer design, adapting positional embeddings for proprioceptive state sequences
- Implement the interleaved (R̂, s, a) tokenization scheme for encoding locomotion trajectories in the offline skill library
- Leverage the training stability (simple MSE loss, no conservative regularization) for initial policy pre-training before online fine-tuning
- Explore return extrapolation at test time for pushing beyond demonstration quality on both robots

## Limitations & Open Questions
- Cannot stitch sub-optimal trajectory segments as effectively as value-based methods (TD learning enables credit assignment across trajectories, which DT lacks)
- No mechanism for online fine-tuning or continual learning—purely offline, limiting sim-to-real adaptation
- Return-to-go conditioning assumes reward functions are well-defined and stationary, which may not hold in multi-objective settings (safety + performance)
- Computational cost scales with context length; long-horizon tasks (thousands of timesteps) may require architectural modifications or hierarchical decomposition
