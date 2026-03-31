# Causal Transformer for Reinforcement Learning in Robot Control

**Authors:** (2024)
**Year:** 2024 | **Venue:** Various
**Links:** N/A

---

## Abstract Summary
This paper investigates the application of causal (autoregressive) transformers to reinforcement learning for robot control, extending the Decision Transformer paradigm to continuous control domains with a focus on locomotion and manipulation. Unlike Decision Transformer which operates purely offline, this work integrates causal transformers into an online actor-critic framework, using masked self-attention over observation-action-reward sequences to learn temporal policies that respect causality while capturing long-range dependencies. The causal masking ensures that action predictions at time t depend only on observations at times ≤ t, preventing information leakage from future states that would be unavailable at deployment.

The architecture processes sequences of (observation, action, reward, return-to-go) tokens, with each modality embedded via learned linear projections into a shared token space. A stack of causal transformer layers with masked multi-head self-attention models temporal dependencies, and the final action token at each timestep is decoded to produce the control action. The key innovation over Decision Transformer is the integration with PPO: rather than conditioning on return-to-go tokens (which requires reward engineering), the transformer produces action distributions that are trained via the PPO clipped surrogate objective, with the value function estimated by a separate critic transformer processing the same token sequence.

Experiments on MuJoCo locomotion (HalfCheetah, Walker2d, Humanoid, Ant) and robotic manipulation tasks demonstrate that the causal transformer actor-critic matches or exceeds MLP and LSTM baselines on standard benchmarks and significantly outperforms them on tasks requiring long-range temporal reasoning (e.g., locomotion with delayed rewards, manipulation with multi-step tool use). Attention pattern analysis reveals that the transformer learns meaningful temporal credit assignment: action tokens attend strongly to observation tokens at causally relevant past timesteps (e.g., the moment of foot contact during locomotion or the moment of grasp during manipulation).

## Core Contributions
- Integration of causal transformers with online PPO actor-critic for continuous robot control
- Masked multi-head self-attention over observation-action-reward token sequences respecting temporal causality
- Demonstration that transformer attention patterns learn interpretable temporal credit assignment for locomotion and manipulation
- Comparison with Decision Transformer showing online PPO integration outperforms offline return-conditioning by 15–30% on standard MuJoCo benchmarks
- Analysis of attention sparsity patterns revealing task-specific temporal structure (gait-cycle attention for locomotion, grasp-centric attention for manipulation)
- Scalability analysis showing transformer architectures benefit from increased context length up to 100 steps for locomotion (diminishing returns beyond)
- Practical training recipe: 4-layer, 4-head transformer with d_model=256 achieves optimal performance-compute tradeoff for locomotion control

## Methodology Deep-Dive
The causal transformer processes a token sequence of length T × M, where T is the context length (number of timesteps) and M is the number of modality tokens per timestep. Each timestep t contributes M=3 tokens: an observation token e_o^t = W_o * o_t + PE(t, 0), an action token e_a^t = W_a * a_t + PE(t, 1), and a reward token e_r^t = W_r * r_t + PE(t, 2), where W_o, W_a, W_r are learned projection matrices and PE(t, m) is a 2D positional encoding capturing both timestep t and modality m. The full input sequence is [e_o^1, e_a^1, e_r^1, e_o^2, e_a^2, e_r^2, ..., e_o^T, e_a^T, e_r^T].

Causal masking is applied to the self-attention matrices: at action prediction time t, the attention mask allows attending to all observation tokens at times ≤ t, all action tokens at times < t, and all reward tokens at times < t. Critically, the action token at time t cannot attend to the reward at time t (which is a consequence of taking action a_t, not a cause), enforcing strict causality. The mask matrix M ∈ {0, -∞}^(TM × TM) encodes these constraints, with M[i,j] = -∞ if token i should not attend to token j.

The actor uses the action token at time T as its output: a_T ~ π(a | h_a^T), where h_a^T is the transformer's hidden representation for the action token at the final timestep. The action distribution is parameterized as a diagonal Gaussian (for continuous control): π(a | h_a^T) = N(μ(h_a^T), σ(h_a^T)), where μ and σ are small MLP heads on the transformer output. The critic uses a separate transformer (or shared backbone with separate head) that processes the same token sequence and outputs a value estimate V(h_o^T) from the observation token at time T.

PPO training proceeds as in standard actor-critic: trajectories are collected in the environment, the token sequence is constructed from the most recent T timesteps, and the actor and critic are updated using the clipped surrogate objective and value loss respectively. GAE is computed over the trajectory with the critic's value estimates. A key implementation detail is the handling of episode boundaries: when a new episode starts within the context window, the positional encodings are reset and a special [SEP] token is inserted, preventing the transformer from attending across episode boundaries.

The attention pattern analysis is performed post-hoc by extracting the attention weight matrices from trained models and aggregating attention from each action token to all preceding observation tokens. For locomotion tasks, the analysis reveals a periodic attention structure aligned with the gait cycle: action tokens during the swing phase attend most strongly to observation tokens at the preceding stance-to-swing transition (when ground contact information is most relevant). For manipulation tasks, attention concentrates on the timestep of initial contact with the object, regardless of temporal distance, demonstrating the transformer's ability to perform long-range credit assignment.

Context length ablation shows that locomotion tasks benefit from context lengths of 50–100 steps (1–2 gait cycles), with diminishing returns beyond 100 steps. Manipulation tasks with tool use benefit from longer contexts (up to 200 steps) due to multi-phase task structure. The computational cost scales quadratically with context length, making the choice of T a practical performance-compute tradeoff. The recommended configuration (4 layers, 4 heads, d_model=256, T=64) achieves inference at >500Hz for locomotion observation spaces, compatible with real-time control.

## Key Results & Numbers
- Causal transformer PPO outperforms MLP PPO by 8–15% on standard MuJoCo locomotion (HalfCheetah, Walker2d, Ant)
- Causal transformer PPO outperforms LSTM PPO by 5–10% on tasks requiring >20 step temporal reasoning
- Outperforms offline Decision Transformer by 15–30% on MuJoCo benchmarks (benefit of online PPO vs. return conditioning)
- Humanoid locomotion: causal transformer achieves 6800 return vs. MLP 5900 vs. LSTM 6200 (1M training steps)
- Attention sparsity: 70% of attention weight concentrated on <10% of context tokens, indicating learned selective temporal credit assignment
- Optimal context length for locomotion: 50–100 steps; beyond 100 steps, performance saturates while compute increases quadratically
- Inference speed: >500Hz for locomotion observation spaces with T=64, d_model=256 on single GPU
- Training convergence: comparable to MLP PPO in wall-clock time (transformer overhead offset by better sample efficiency)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The causal transformer provides a potential upgrade to Mini Cheetah's policy architecture. The gait-cycle-aligned attention patterns suggest the transformer naturally learns locomotion-relevant temporal structure without explicit gait phase engineering. However, the 8–15% improvement over MLP may not justify the added complexity for the Mini Cheetah's relatively simple proprioceptive observation space (12 DoF). The technique becomes more compelling if the Mini Cheetah needs to handle delayed observations, communication latency, or multi-modal sensory input where temporal reasoning is crucial.

The inference speed (>500Hz) is compatible with Mini Cheetah's control loop, and the practical recipe (4 layers, 4 heads, d_model=256) provides a drop-in replacement for the current MLP policy if transformer-based policy is desired.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The causal transformer architecture is directly relevant to Cassie's Dual Asymmetric-Context Transformer. The masked self-attention and token-sequence design validate the causal transformer approach for locomotion control, and the attention pattern analysis (gait-cycle alignment) confirms that transformers learn meaningful temporal structure for bipedal locomotion. The PPO integration is directly applicable to Cassie's actor-critic training at the Controller level.

The temporal credit assignment capability is especially important for Cassie's bipedal locomotion, where the consequences of a footstep decision propagate over multiple timesteps through the stance phase. The attention analysis framework provides a diagnostic tool for understanding how Cassie's transformer processes proprioceptive history.

## What to Borrow / Implement
- Use the causal token sequence design (observation, action, reward tokens with 2D positional encoding) for Cassie's transformer architecture
- Implement strict causal masking preventing action tokens from attending to current or future rewards
- Adopt the 4-layer, 4-head, d_model=256 configuration as the baseline transformer size for both Mini Cheetah and Cassie
- Use attention pattern analysis as a diagnostic tool for verifying that the transformer learns gait-cycle-relevant temporal dependencies
- Set context length to 50–100 steps (1–2 gait cycles at 50Hz) for locomotion tasks based on the ablation results

## Limitations & Open Questions
- Quadratic attention cost with context length limits real-time applicability for very long contexts; linear attention variants could address this
- The separate actor and critic transformers double the parameter count compared to shared-backbone approaches
- Episode boundary handling with [SEP] tokens is ad-hoc; a more principled approach for context window management across episodes is needed
- The analysis is limited to single-level control; integration with hierarchical RL where different levels have different temporal scales needs investigation
