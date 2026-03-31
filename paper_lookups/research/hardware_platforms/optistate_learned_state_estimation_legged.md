# OptiState: State Estimation for Legged Robots using Gated Networks and Transformers

**Authors:** Alex Schperberg et al.
**Year:** 2024 | **Venue:** ICRA 2024
**Links:** https://github.com/AlexS28/OptiState

---

## Abstract Summary
OptiState is an open-source state estimation framework combining gated neural networks, transformer-based vision modules, and Kalman filtering for robust legged robot state estimation. It handles both proprioceptive and exteroceptive inputs with learned fusion weights, outperforming pure EKF baselines on real-world legged robot platforms.

## Core Contributions
- Combines gated neural networks with transformer-based vision modules for multi-modal state estimation
- Learns optimal sensor fusion weights rather than relying on hand-tuned covariance matrices
- Integrates seamlessly with Kalman filtering as a hybrid learned-classical estimator
- Handles graceful degradation when individual sensor modalities fail or produce noisy data
- Open-source implementation facilitating reproducibility and extension
- Validated on real-world legged robot hardware with extensive benchmarking
- Demonstrates that learned fusion outperforms fixed-weight sensor combination strategies

## Methodology Deep-Dive
OptiState addresses the fundamental challenge of multi-modal sensor fusion for legged robots by learning when and how much to trust each sensor modality. Traditional approaches use fixed covariance matrices in EKF/UKF frameworks, requiring careful hand-tuning that may not generalize across operating conditions. OptiState replaces this manual tuning with learned gating mechanisms.

The architecture consists of three main components. First, a proprioceptive encoder processes IMU data and joint encoder readings through a gated recurrent network that captures temporal dynamics and learns to filter sensor noise. The gating mechanism allows the network to selectively attend to reliable signal components while suppressing noise, adapting in real-time to changing locomotion dynamics.

Second, a transformer-based vision module processes exteroceptive inputs (depth images, point clouds) using self-attention to identify terrain features relevant for state estimation. The transformer architecture is particularly effective at capturing long-range spatial relationships in the visual input, enabling the estimator to anticipate terrain changes before the robot encounters them. The vision module outputs both a terrain representation and a confidence score used in downstream fusion.

Third, a learned fusion layer combines the proprioceptive and exteroceptive estimates using attention-weighted averaging. The fusion weights are conditioned on the estimated reliability of each modality—when vision is degraded (e.g., in darkness or with occlusions), the system automatically upweights proprioceptive estimates, and vice versa. This adaptive fusion is the key advantage over fixed-weight Kalman filter approaches.

The entire pipeline is trained end-to-end using a combination of supervised losses (ground-truth state from motion capture) and self-supervised consistency losses (enforcing kinematic constraints). The Kalman filter component provides a physically grounded prior that regularizes the learned components, preventing overfitting to training distributions.

## Key Results & Numbers
- Outperforms pure EKF baselines across all tested locomotion scenarios
- Open-source implementation available at the linked GitHub repository
- Real-world validation on legged robot hardware with motion capture ground truth
- Handles multi-modal sensor fusion with automatic reliability weighting
- Graceful degradation when individual sensor modalities are occluded or noisy
- Competitive computational cost enabling near-real-time operation

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
OptiState provides a ready-to-use multi-modal state estimation framework for Mini Cheetah deployment. The learned fusion between proprioceptive and exteroceptive inputs is directly applicable when augmenting Mini Cheetah with vision sensors. The gated network's ability to handle sensor degradation is critical for robust field deployment. The open-source nature allows direct integration with the MuJoCo simulation pipeline for training state estimators alongside RL policies.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The transformer-based vision module aligns architecturally with Project B's use of transformers throughout the hierarchy (Dual Asymmetric-Context Transformer). The learned sensor fusion could be integrated with or replace the CPTE (contrastive terrain encoder) for terrain-aware state estimation. The adaptive fusion weights are particularly relevant for Cassie's operation across diverse terrains where sensor reliability varies. The approach of learning to weight different information sources mirrors the attention mechanisms in MC-GAT.

## What to Borrow / Implement
- Adopt the gated proprioceptive encoder for Mini Cheetah's state estimation pipeline
- Integrate the transformer vision module architecture with Cassie's CPTE for improved terrain encoding
- Use learned fusion weights instead of hand-tuned Kalman filter covariances in both projects
- Leverage the open-source codebase as a foundation for project-specific state estimator development
- Apply the self-supervised consistency losses to train state estimators without requiring motion capture ground truth

## Limitations & Open Questions
- Transformer vision module adds computational overhead that may challenge real-time constraints on embedded hardware
- Training requires ground-truth state data (typically from motion capture), limiting training data to lab settings
- Generalization to terrain types and dynamics regimes not seen during training is not guaranteed
- The interaction between the learned estimator and the RL policy (which was trained with different state representations) needs careful handling
- Battery and compute budget on legged robots may require model compression or distillation for deployment
