# Study on LSTM and ConvLSTM Memory-Based Deep Reinforcement Learning

**Authors:** Various
**Year:** 2024 | **Venue:** Springer (LNCS)
**Links:** https://link.springer.com/chapter/10.1007/978-3-031-55326-4_11

---

## Abstract Summary
Comparative study examining LSTM, ConvLSTM, MDN-RNN, and GridLSTM memory modules in deep RL. Evaluates how different recurrent architectures affect policy learning in partially observable environments, with implications for locomotion tasks requiring temporal reasoning. The study provides empirical guidelines for selecting memory architectures based on observation type, task complexity, and computational constraints.

## Core Contributions
- Comprehensive empirical comparison of four recurrent architectures (LSTM, ConvLSTM, MDN-RNN, GridLSTM) in deep RL settings
- Evaluates memory modules across multiple partially observable environments with varying complexity
- Demonstrates that LSTM provides a strong baseline with favorable compute-performance trade-off
- Shows ConvLSTM is superior for tasks with spatial-temporal observation patterns
- Identifies GridLSTM as most effective for tasks requiring complex multi-scale temporal dependencies
- Provides practical guidelines for architecture selection based on task characteristics
- Analyzes memory capacity, training stability, and generalization properties of each architecture

## Methodology Deep-Dive
The study evaluates four recurrent architectures integrated into PPO's actor-critic framework. For each architecture, the recurrent module replaces the standard MLP hidden layers, processing sequential observations to produce a hidden state that captures task-relevant history. The evaluation is conducted across environments that require different types of temporal reasoning: delayed rewards, hidden state estimation, and sequential decision-making under partial observability.

LSTM (Long Short-Term Memory) serves as the baseline architecture. The standard LSTM cell with input, forget, and output gates is used with hidden sizes of 64, 128, and 256. The study confirms LSTM's well-known ability to capture long-range dependencies while maintaining training stability. For locomotion-relevant tasks (requiring proprioceptive history for velocity estimation and terrain inference), LSTM with 128 hidden units provides the best performance-compute trade-off. The gating mechanism naturally learns to retain relevant history (recent contacts, velocity trends) while discarding noise.

ConvLSTM extends LSTM with convolutional operations in the gate computations, making it naturally suited for spatial-temporal data. In the context of robotic RL, ConvLSTM processes structured observations (e.g., heightmaps, joint arrays organized by kinematic topology) while maintaining temporal memory. The study shows ConvLSTM outperforms standard LSTM when observations have spatial structure, achieving 15-25% higher returns on tasks with grid-based or image-based observations. However, for purely vectorized proprioceptive observations, ConvLSTM's advantage is marginal.

MDN-RNN (Mixture Density Network RNN) combines LSTM with a mixture density output that models the distribution of next observations. This architecture is particularly relevant for world models (e.g., in Dreamer/RSSM frameworks) where predicting future observations is a core function. The study finds MDN-RNN provides superior performance in model-based RL settings where the recurrent module serves as both a memory and a forward predictor. However, the added complexity of the mixture density output increases training time by 40-60% and can introduce training instability if not carefully tuned.

GridLSTM organizes multiple LSTM cells in a grid structure, enabling multi-dimensional recurrence. Each grid dimension can process a different aspect of the input (e.g., temporal, spatial, hierarchical). The study shows GridLSTM excels at tasks requiring multi-scale temporal reasoning—where both short-term reactivity and long-term planning are needed. For locomotion, this translates to simultaneously tracking fast dynamics (contact timing, actuator response) and slow dynamics (terrain changes, energy state, gait phase). However, GridLSTM's computational cost is 3-5x that of standard LSTM, limiting its practicality for real-time control.

## Key Results & Numbers
- LSTM (128 hidden): strong baseline, 1x compute cost, robust across all tasks
- ConvLSTM: +15-25% returns on spatial-temporal tasks, +5% on vector observations, 1.5x compute
- MDN-RNN: best for model-based RL / world models, 1.4-1.6x compute, less stable training
- GridLSTM: best for multi-scale temporal tasks, 3-5x compute, highest peak performance
- LSTM matches or exceeds feedforward MLP by 30-50% on POMDP locomotion tasks
- Hidden size 128 optimal for locomotion tasks (64 too small, 256 marginal improvement)
- All recurrent architectures significantly outperform frame-stacking (concatenating last N observations)
- Training with TBPTT (truncated backpropagation through time) over 16-32 timesteps is sufficient for locomotion

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Mini Cheetah's primary policy uses feedforward MLP for PPO, but recurrent architectures become relevant for partial observability scenarios: blind locomotion (no vision), uneven terrain without heightmap access, and estimating ground properties from proprioceptive history. The study's finding that LSTM with 128 hidden units is the best trade-off for proprioceptive locomotion tasks provides a clear starting point if recurrence is needed. The frame-stacking comparison is valuable—it quantifies the improvement from proper recurrence over the common practice of concatenating recent observations. For Mini Cheetah's 500 Hz control loop, the computational overhead of LSTM is negligible compared to the overall policy inference time.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This study is directly relevant to multiple components of Project B's architecture. The Dual Asymmetric-Context Transformer uses attention-based temporal processing, and understanding how it compares to LSTM baselines is critical for justifying the architectural choice. The MDN-RNN findings inform the RSSM/Dreamer world model component, where the recurrent module's ability to predict future states is essential. The MC-GAT (GATv2 on kinematic tree) processes structured observations where ConvLSTM's spatial-temporal capabilities are relevant for comparison. The GridLSTM's multi-scale temporal processing aligns with the hierarchical architecture's need to process information at different timescales (fast controller at 500 Hz, slow planner at 10 Hz). The study provides empirical baselines against which the Transformer-based architecture can be compared, helping quantify the benefit of attention over recurrence.

## What to Borrow / Implement
- Use LSTM (128 hidden) as a baseline recurrent policy for comparison against Transformer architectures in Project B
- Consider ConvLSTM for processing structured kinematic observations in MC-GAT's input preprocessing
- Leverage MDN-RNN insights for the RSSM/Dreamer world model's recurrent backbone design
- Apply the TBPTT guidelines (16-32 steps) for training recurrent policies in both projects
- Use the study's evaluation protocol to compare memory architectures during architecture search
- Implement LSTM-PPO as a blind locomotion baseline for Mini Cheetah before considering more complex architectures

## Limitations & Open Questions
- Evaluation environments may not fully represent the complexity of real-world legged locomotion
- Transformer architectures (which have largely superseded LSTMs) are not included in the comparison
- Computational cost analysis does not account for hardware-specific optimizations (GPU LSTM kernels, etc.)
- The study uses fixed hyperparameters; optimal settings may differ across architectures
- How recurrent architectures interact with domain randomization for sim-to-real transfer is not studied
- Extension to multi-agent or hierarchical settings where multiple recurrent modules interact is unexplored
- Real-world deployment latency analysis for each architecture on embedded hardware is missing
