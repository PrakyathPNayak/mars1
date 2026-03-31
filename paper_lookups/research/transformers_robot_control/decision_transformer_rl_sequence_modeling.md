---
## 📂 FOLDER: research/transformers_robot_control/

### 📄 FILE: research/transformers_robot_control/decision_transformer_rl_sequence_modeling.md

**Title:** Decision Transformer: Reinforcement Learning via Sequence Modeling
**Authors:** Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch
**Year:** 2021
**Venue:** NeurIPS 2021
**arXiv / DOI:** arXiv:2106.01345

**Abstract Summary (2–3 sentences):**
Decision Transformer reframes reinforcement learning as a sequence modeling problem, using a GPT-style causal transformer to generate actions conditioned on a desired return-to-go, past states, and past actions. Rather than fitting value functions or computing policy gradients, the model is trained with supervised next-token prediction on offline trajectory data. This approach matches or exceeds state-of-the-art offline RL algorithms (CQL, BCQ) on D4RL benchmarks without any explicit RL training (no Bellman backups, no temporal difference learning).

**Core Contributions (bullet list, 4–7 items):**
- Reframes RL as conditional sequence modeling: given desired return, past states, and actions, predict the next action
- Uses a GPT-2 architecture with causal masking to autoregressively generate actions
- Introduces return-to-go conditioning: the model is conditioned on the remaining cumulative reward, enabling goal-directed behavior at test time
- Matches or exceeds CQL, BCQ, and other offline RL algorithms on D4RL benchmarks using only supervised learning
- Demonstrates that transformers can capture long-horizon credit assignment through attention, replacing the need for Bellman backups
- Shows generalization across tasks — a single model can be conditioned on different return-to-go values to produce policies of varying quality
- Provides a simple, scalable framework that leverages pre-training advances from NLP for RL problems

**Methodology Deep-Dive (3–5 paragraphs):**
Decision Transformer represents a trajectory as an interleaved sequence of return-to-go tokens (R̂_t), state tokens (s_t), and action tokens (a_t): (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T). The return-to-go R̂_t at timestep t is defined as the sum of future rewards from t onward: R̂_t = Σ_{t'=t}^{T} r_{t'}. Each token type has its own embedding layer: states are projected through a linear layer (or CNN for image observations), actions through another linear layer, and return-to-go scalars through a third. Positional encodings are added based on the timestep (not token position within a timestep), so all three tokens at the same timestep share the same positional encoding.

The core architecture is a GPT-2 causal transformer with standard multi-head self-attention and feedforward layers. Causal masking ensures each token can only attend to itself and previous tokens in the sequence, maintaining autoregressive generation. At test time, the model generates actions by feeding in the current state, the desired return-to-go (set by the user), and optionally the history of past states, actions, and returns. The context window length K determines how much history the model considers. The action prediction is made by taking the output at the action token position and passing it through a linear output head. For continuous action spaces, the action is predicted directly; for discrete spaces, a softmax over action logits is used.

Training is performed entirely offline using supervised learning on a dataset of trajectories. Each training sample is a subsequence of length K from a trajectory, with the target being the next action at each timestep. The loss is simply the mean squared error (continuous actions) or cross-entropy (discrete actions) between predicted and ground-truth actions. Critically, no RL-specific training is involved — no value function fitting, no temporal difference learning, no policy gradient computation. The return-to-go provides the signal for what quality of behavior to produce, replacing the role of the reward signal in traditional RL.

A key design choice is how to handle the return-to-go conditioning at test time. The initial return-to-go R̂_1 is set to the desired total episode return (e.g., the maximum observed in the training data for optimal behavior). As the episode progresses, R̂_t is updated by subtracting the received reward: R̂_{t+1} = R̂_t - r_t. This means the model always sees how much reward it still needs to collect, enabling it to adjust its behavior dynamically. Setting a higher initial return-to-go produces more ambitious (expert-level) behavior, while lower values produce more conservative behavior. This provides a natural mechanism for controlling policy quality without retraining.

The experimental evaluation focuses on offline RL benchmarks (D4RL), where the model is trained on fixed datasets of varying quality (random, medium, medium-replay, medium-expert, expert). Decision Transformer is compared against CQL, BCQ, BEAR, and other offline RL algorithms. Results show that Decision Transformer matches or exceeds these algorithms on most tasks, particularly excelling on environments with sparse rewards where long-horizon credit assignment is critical. The advantage comes from the transformer's ability to directly attend to distant timesteps with relevant rewards, bypassing the need to propagate value information through Bellman backups. The authors also demonstrate that Decision Transformer scales well with model size and context length, consistent with scaling laws observed in NLP.

**Key Results & Numbers:**
- Matches or exceeds CQL, BCQ on D4RL benchmarks (HalfCheetah, Hopper, Walker2D) for offline RL
- No explicit RL training needed — purely supervised sequence prediction
- Particularly strong on sparse-reward tasks where long-horizon credit assignment matters
- Performance improves with context length K (up to ~20 timesteps)
- Single model can produce policies of varying quality by adjusting the return-to-go conditioning
- Scales well with model size (up to ~1.5M parameters evaluated)
- Generalizes across multiple tasks within the same environment suite

**Relevance to Project A (Mini Cheetah):** MEDIUM — The sequence modeling paradigm could be applied to Mini Cheetah policy learning, particularly for offline RL from demonstration data or reward-conditioned behavior generation. However, Mini Cheetah uses online PPO training, and Decision Transformer is designed for offline RL. The return-to-go conditioning concept could inspire goal-conditioned policy designs for velocity tracking.

**Relevance to Project B (Cassie HRL):** HIGH — Decision Transformer directly informs the Dual Asymmetric-Context Transformer design in Cassie's HRL system. The concept of conditioning on desired outcomes (return-to-go) maps to the Planner level's goal conditioning for downstream primitives. The sequence modeling framework — processing temporal sequences of states and actions with causal attention — provides the template for the temporal attention mechanism in the Dual Asymmetric-Context Transformer. The asymmetric context idea (different context windows for different information types) extends Decision Transformer's interleaved token design.

**What to Borrow / Implement:**
- Return-to-go conditioning paradigm: condition policies on desired outcomes (velocity, gait quality) rather than only current state — applicable to both the Planner and Controller levels in Cassie's HRL
- Interleaved token design: represent different modalities (state, action, goal, terrain) as separate token types with shared positional encodings per timestep
- Causal attention for temporal history: use causal masking to process history of states and actions for context-dependent decision making
- The insight that transformers handle long-horizon credit assignment through direct attention to distant timesteps — motivates using transformer temporal context in the Planner and Controller
- Offline pre-training with supervised sequence prediction as a warm-start strategy before online PPO fine-tuning

**Limitations & Open Questions:**
- Designed for offline RL — adapting to online/on-policy training (PPO) requires significant modifications
- Return-to-go conditioning assumes access to a distribution of trajectory returns, which may not be available in early training stages
- The model does not explicitly learn dynamics or value functions, potentially limiting its ability to handle out-of-distribution states
- Context window length K limits the temporal horizon; for very long episodes (locomotion over minutes), this may be insufficient
- Computational cost of transformer attention at each timestep may be prohibitive for real-time control (20+ Hz for Mini Cheetah, 40+ Hz for Cassie)
- The paper does not address how to combine Decision Transformer with hierarchical control architectures
---
