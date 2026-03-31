# Research Papers: Hierarchical Transformers & Advanced ML for Quadruped Locomotion

A comprehensive survey of 55 papers relevant to advancing the Mini Cheetah locomotion control architecture.

---

## Category 1: Transformer-Based Locomotion Control

### 1. Unified Locomotion Transformer (ULT)
- **Authors**: Dikai Liu, Tianwei Zhang, Jianxiong Yin, Simon See
- **ArXiv**: 2503.08997 (Mar 2025, IROS 2025)
- **Summary**: Proposes a transformer-based framework that unifies teacher-student knowledge transfer and policy optimization in a single network. Eliminates multi-stage distillation by co-training teacher/student with RL, next state-action prediction, and action imitation simultaneously, achieving zero-shot sim-to-real deployment.

### 2. Terrain Transformer (TERT)
- **Authors**: Hang Lai, Weinan Zhang, Xialin He, Chen Yu, Zheng Tian, Yong Yu, Jun Wang
- **ArXiv**: 2212.07740 (Dec 2022, ICRA 2023)
- **Summary**: A high-capacity Transformer model for quadrupedal locomotion on various terrains. Uses a two-stage training framework (offline pretraining + online correction) that integrates Transformer with privileged training. Outperforms baselines on 9 challenging terrains including sand and stairs.

### 3. Masked Sensory-Temporal Attention (MSTA)
- **Authors**: Dikai Liu, Tianwei Zhang, Jianxiong Yin, Simon See
- **ArXiv**: 2409.03332 (Sep 2024, ICRA 2025)
- **Summary**: A transformer-based mechanism with masking for quadruped locomotion that employs direct sensor-level attention to enhance sensory-temporal understanding. Handles different sensor combinations and is robust to missing information, enabling deployment across different physical systems.

### 4. State Estimation Transformers (SET)
- **Authors**: Chen Yu, Yichu Yang, Tianlin Liu, Yangwei You, Mingliang Zhou, Diyun Xiang
- **ArXiv**: 2410.13496 (Oct 2024, IROS 2024)
- **Summary**: Casts state estimation for legged robots as conditional sequence modeling using a causally masked Transformer. Predicts privileged states (body height, velocities) from past observations for agile locomotion including jumping and backflipping.

### 5. Decision Transformers for Quadruped Locomotion
- **Authors**: Orhan Eren Akgün, Néstor Cuevas, Matheus Farias, Daniel Garces
- **ArXiv**: 2402.13201 (Feb 2024)
- **Summary**: Applies Decision Transformer architecture for quadruped locomotion with emphasis on tiny/efficient models suitable for edge deployment in search-and-rescue or swarm robotics applications.

### 6. Fourier Controller Networks
- **Authors**: Hengkai Tan, Songming Liu, Kai Ma et al.
- **ArXiv**: 2405.19885 (May 2024)
- **Summary**: Proposes frequency-domain representation as an alternative to Transformers for real-time embodied decision-making. Addresses low data efficiency and high inference latency issues of Transformers for locomotion control.

---

## Category 2: Attention Mechanisms for Locomotion

### 7. AME-2: Attention-Based Neural Map Encoding
- **Authors**: Chong Zhang, Victor Klemm, Fan Yang, Marco Hutter
- **ArXiv**: 2601.08485 (Jan 2026)
- **Summary**: A unified RL framework for agile and generalized locomotion using a novel attention-based map encoder. Extracts local and global mapping features with attention mechanisms for interpretable, generalized terrain encoding. Validated on both quadruped and biped robots.

### 8. PhysGraph: Graph-Transformer Policies
- **Authors**: Runfa Blark Li et al.
- **ArXiv**: 2603.01436 (Mar 2026)
- **Summary**: Uses graph-transformer architecture for dexterous manipulation by modeling articulated hand structure as a graph with topological information, rather than naive flattened state vectors.

### 9. Embedding Morphology into Transformers
- **Authors**: Kei Suzuki, Jing Liu, Ye Wang et al.
- **ArXiv**: 2603.00182 (Feb 2026)
- **Summary**: Enables cross-robot policy learning by embedding robot morphological information directly into transformer representations, allowing knowledge transfer between different robot embodiments.

### 10. GET-Zero: Graph Embodiment Transformer
- **Authors**: Austin Patel, Shuran Song
- **ArXiv**: 2407.15002 (Jul 2024)
- **Summary**: Model architecture for learning embodiment-aware control policy that generalizes across different robot morphologies using graph-based representation of robot body structure.

### 11. CroSTAta: Cross-State Transition Attention Transformer
- **Authors**: Giovanni Minelli et al.
- **ArXiv**: 2510.00726 (Oct 2025)
- **Summary**: Introduces cross-state transition attention for robotic manipulation, capturing temporal dependencies across state transitions for improved policy learning.

### 12. HEIGHT: Heterogeneous Interaction Graph Transformer
- **Authors**: Shuijing Liu et al.
- **ArXiv**: 2411.12150 (Nov 2024, T-ASE)
- **Summary**: Uses heterogeneous graph transformers for robot navigation in crowded/constrained environments, modeling all types of spatial and temporal interactions among agents and obstacles.

---

## Category 3: Mixture of Experts & Multi-Task Learning

### 13. MoE-Loco: Mixture of Experts for Multitask Locomotion
- **Authors**: Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao
- **ArXiv**: 2503.08564 (Mar 2025)
- **Summary**: MoE framework enabling a single policy to handle diverse terrains (bars, pits, stairs, slopes) while supporting multiple gaits. Different experts naturally specialize in distinct locomotion behaviors, mitigating gradient conflicts in multitask RL.

### 14. Efficient Diffusion Transformer Policies with MoE Denoisers
- **Authors**: Moritz Reuss et al.
- **ArXiv**: 2412.12953 (Dec 2024)
- **Summary**: Combines diffusion policies with Mixture of Expert denoisers for efficient multitask learning in robot control, scaling to many tasks without proportional compute increase.

### 15. Parkour in the Wild: Multi-expert Distillation
- **Authors**: Nikita Rudin, Junzhe He, Joshua Aurand, Marco Hutter
- **ArXiv**: 2505.11164 (May 2025)
- **Summary**: Novel framework for agile locomotion using multi-expert distillation and RL fine-tuning. Trains terrain-specific experts then distills into a single general policy for unstructured environments including parkour scenarios.

### 16. Towards Adaptive Humanoid Control via Multi-Behavior Distillation
- **Authors**: Yingnan Zhao et al.
- **ArXiv**: 2511.06371 (Nov 2025)
- **Summary**: Learns diverse human-like behaviors through multi-behavior distillation and reinforced fine-tuning, addressing the challenge of combining multiple skills into a single controller.

---

## Category 4: Hierarchical & Curriculum Learning

### 17. TDGC: Task-level Decision to Gait Control
- **Authors**: Sijia Li, Haoyu Wang, Shenghai Yuan et al.
- **ArXiv**: 2603.05783 (Mar 2026, submitted to IROS 2026)
- **Summary**: A hierarchical policy architecture for quadrupedal navigation where a high-level policy makes task-level decisions and a low-level policy handles gait control. Addresses sim-to-real instabilities under out-of-distribution conditions.

### 18. Scaling Rough Terrain Locomotion with Automatic Curriculum RL (LP-ACRL)
- **Authors**: Ziming Li, Chenhao Li, Marco Hutter
- **ArXiv**: 2601.17428 (Jan 2026)
- **Summary**: Learning Progress-based Automatic Curriculum RL that estimates online learning progress to adaptively adjust task-sampling distribution. Achieves 2.5 m/s on diverse terrains (stairs, slopes, gravel) without prior knowledge of difficulty.

### 19. CurricuLLM: LLM-Guided Task Curricula
- **Authors**: Kanghyun Ryu et al.
- **ArXiv**: 2409.18382 (Sep 2024, ICRA 2025)
- **Summary**: Uses large language models to automatically design task curricula for learning complex robot skills, eliminating manual curriculum design.

### 20. SPECI: Skill Prompts Hierarchical Continual Imitation Learning
- **Authors**: Jingkai Xu, Xiangli Nie
- **ArXiv**: 2504.15561 (Apr 2025)
- **Summary**: Hierarchical continual imitation learning framework for robot manipulation in dynamic environments requiring lifelong adaptability to evolving objects and tasks.

### 21. Distillation-PPO for Humanoid Robot Perceptive Locomotion
- **Authors**: Qiang Zhang et al.
- **ArXiv**: 2503.08299 (Mar 2025)
- **Summary**: Two-stage RL framework combining knowledge distillation with PPO for humanoid perceptive locomotion on complex terrains.

### 22. PUMA: Perception-driven Unified Foothold Prior
- **Authors**: Liang Wang et al.
- **ArXiv**: 2601.15995 (Jan 2026)
- **Summary**: Hierarchical control with perception-driven foothold selection for quadruped parkour, where the robot reasons about environmental features to select footholds.

---

## Category 5: Sim-to-Real Transfer & Domain Adaptation

### 23. DreamWaQ++: Obstacle-Aware Quadrupedal Locomotion
- **Authors**: I Made Aswin Nahrendra et al.
- **ArXiv**: 2409.19709 (Sep 2024, T-RO 2026)
- **Summary**: Fuses proprioception and exteroception via resilient multi-modal RL. Achieves agile locomotion over rough terrains, steep slopes, and high-rise stairs while being robust to out-of-distribution situations. Extension of the DreamWaQ framework.

### 24. DrEureka: LLM-Guided Sim-To-Real Transfer
- **Authors**: Yecheng Jason Ma et al.
- **ArXiv**: 2406.01967 (Jun 2024, RSS 2024)
- **Summary**: Uses language models to guide sim-to-real transfer by automatically tuning domain randomization parameters and reward functions for zero-shot deployment.

### 25. Contrastive Representation Learning for Robust Sim-to-Real Transfer
- **Authors**: Yidan Lu et al.
- **ArXiv**: 2509.12858 (Sep 2025)
- **Summary**: Uses contrastive representation learning to bridge the sim-to-real gap for adaptive humanoid locomotion on complex terrains.

### 26. Impact of Static Friction on Sim2Real in Robotic RL
- **Authors**: Xiaoyi Hu et al.
- **ArXiv**: 2503.01255 (Mar 2025)
- **Summary**: Analyzes the critical role of static friction modeling in successful sim-to-real transfer for robotic RL, identifying key parameters.

### 27. Impedance Matching for Sim-to-Real Quadruped Jumping
- **Authors**: Neil Guan et al.
- **ArXiv**: 2404.15096 (Apr 2024)
- **Summary**: Addresses the sim-to-real gap for dynamic quadruped jumping by matching impedance parameters between simulation and real hardware.

### 28. OptiState: Gated Networks with Transformer-based Vision and Kalman Filtering
- **Authors**: Alexander Schperberg et al.
- **ArXiv**: 2401.16719 (Jan 2024, ICRA 2024)
- **Summary**: Combines Transformer-based visual processing with Kalman filtering for robust state estimation in legged robots, bridging proprioception and vision.

---

## Category 6: World Models & Model-Based Learning

### 29. Denoising World Model Learning (DWL)
- **Authors**: Xinyang Gu et al.
- **ArXiv**: 2408.14472 (Aug 2024, RSS 2024 Best Paper Finalist)
- **Summary**: End-to-end RL framework with denoising world model for humanoid locomotion on challenging terrains (snow, stairs, uneven terrain). Zero-shot sim-to-real with a single neural network.

### 30. GenRL: Multimodal Foundation World Models
- **Authors**: Pietro Mazzaglia et al.
- **ArXiv**: 2406.18043 (Jun 2024, NeurIPS 2024)
- **Summary**: Multimodal foundation world models for generalization in embodied agents across multiple tasks and domains.

### 31. Residual MPC: Blending RL with GPU-Parallelized MPC
- **Authors**: Se Hwan Jeon et al.
- **ArXiv**: 2510.12717 (Oct 2025)
- **Summary**: Combines RL residual policies with Model Predictive Control for locomotion, leveraging physical model accuracy while compensating for model mismatch with learned residuals.

### 32. Data-Driven Physics Embedded Dynamics with RL for Quadrupeds
- **Authors**: Prakrut Kotecha et al.
- **ArXiv**: 2603.14333 (Mar 2026)
- **Summary**: Integrates physics-embedded dynamics models with MPC and RL for quadruped locomotion, using learned dynamics that respect physical laws.

---

## Category 7: Diffusion-Based Policies

### 33. DiffuseLoco: Real-Time Legged Locomotion with Diffusion
- **Authors**: Xiaoyu Huang et al.
- **ArXiv**: 2404.19264 (Apr 2024)
- **Summary**: Framework for multi-skill diffusion-based policies from offline datasets. Uses receding horizon control and delayed inputs for real-time deployment. Demonstrates free transitions between locomotion skills and robustness to environmental variations.

### 34. CDP: Causal Diffusion for Visuomotor Policy
- **Authors**: Jiahua Ma et al.
- **ArXiv**: 2506.14769 (Jun 2025)
- **Summary**: Autoregressive visuomotor policy learning via causal diffusion for robust policy generation with temporal causality.

### 35. MaIL: Improving Imitation Learning with Mamba
- **Authors**: Xiaogang Jia et al.
- **ArXiv**: 2406.08234 (Jun 2024)
- **Summary**: Applies Mamba (selective state-space model) architecture to imitation learning as an alternative to transformers, offering linear complexity for long sequences.

---

## Category 8: Morphology-Aware & GNN-Based Control

### 36. MS-PPO: Morphological-Symmetry-Equivariant PPO
- **Authors**: Sizhe Wei et al.
- **ArXiv**: 2512.00727 (Dec 2025)
- **Summary**: Encodes morphological symmetry directly into the PPO policy architecture for legged robot locomotion, improving sample efficiency and policy quality.

### 37. Evolving Embodied Intelligence: GNN-Driven Co-Design
- **Authors**: Jianqiang Wang et al.
- **ArXiv**: 2603.19582 (Mar 2026)
- **Summary**: Uses graph neural networks for co-optimizing morphology and control in soft robotics, demonstrating that GNNs can capture structural relationships for better control policies.

### 38. Morphology-Aware Graph RL for Tensegrity Robots
- **Authors**: Chi Zhang et al.
- **ArXiv**: 2510.26067 (Oct 2025)
- **Summary**: Graph RL approach that encodes morphological structure for tensegrity robot locomotion, demonstrating value of structure-aware representations.

### 39. HeteroMorpheus: Universal Control via Morphological Heterogeneity
- **Authors**: YiFan Hao et al.
- **ArXiv**: 2408.01230 (Aug 2024)
- **Summary**: Universal control based on modeling morphological heterogeneity, enabling cross-embodiment transfer between different robot types.

### 40. COMPOSER: Scalable Modular Policies for Snake Robots
- **Authors**: Yuyou Zhang et al.
- **ArXiv**: 2310.00871 (Oct 2023)
- **Summary**: Modular policy architecture with attention-based message passing between body segments for hyper-redundant robot control.

---

## Category 9: Terrain-Adaptive Locomotion

### 41. KiRAS: Keyframe Guided Self-Imitation
- **Authors**: Xiaoyi Wei et al.
- **ArXiv**: 2603.15179 (Mar 2026, ICRA 2026)
- **Summary**: Self-imitation framework with keyframe guidance for robust and adaptive skill learning in quadruped robots on challenging terrains.

### 42. PGTT: Phase-Guided Terrain Traversal
- **Authors**: Alexandros Ntagkas et al.
- **ArXiv**: 2510.18348 (Oct 2025)
- **Summary**: Phase-guided approach to perceptive legged locomotion that structures terrain traversal into distinct gait phases for improved control.

### 43. Learning Terrain-Specialized Policies
- **Authors**: Matheus P. Angarola et al.
- **ArXiv**: 2509.20635 (Sep 2025, ICAR 2025)
- **Summary**: Trains multiple terrain-specialized policies and selects the appropriate policy based on current terrain conditions for adaptive locomotion.

### 44. DPL: Depth-only Perceptive Humanoid Locomotion
- **Authors**: Jingkai Sun et al.
- **ArXiv**: 2510.07152 (Oct 2025)
- **Summary**: Uses depth-only perception with cross-attention terrain reconstruction for humanoid locomotion on diverse terrains.

### 45. Motion Priors Reimagined: Adapting Flat-Terrain Skills
- **Authors**: Zewei Zhang et al.
- **ArXiv**: 2505.16084 (May 2025, CoRL)
- **Summary**: Adapts motion priors learned on flat terrain for complex mobility tasks, reducing the need for direct training on challenging terrains.

### 46. GenTe: Generative Real-world Terrains
- **Authors**: Hanwen Wan et al.
- **ArXiv**: 2504.09997 (Apr 2025)
- **Summary**: Framework for generating physically realistic terrains for training legged locomotion controllers, improving generalization to real-world conditions.

---

## Category 10: Multi-Embodiment & General Policy Learning

### 47. One Policy to Run Them All: Multi-Embodiment Locomotion
- **Authors**: Nico Bohlinger et al.
- **ArXiv**: 2409.06366 (Sep 2024)
- **Summary**: End-to-end learning approach for multi-embodiment locomotion that trains a single policy working across different robot morphologies.

### 48. GPO: Growing Policy Optimization
- **Authors**: Shuhao Liao et al.
- **ArXiv**: 2601.20668 (Jan 2026)
- **Summary**: Progressive policy growing for legged robot locomotion and whole-body control, automatically scaling policy complexity as task difficulty increases.

### 49. Articulated-Body Dynamics Network
- **Authors**: Sangwoo Shin et al.
- **ArXiv**: 2603.19078 (Mar 2026)
- **Summary**: Dynamics-grounded prior network for robot learning that embeds articulated-body dynamics knowledge into the neural network architecture for improved physics-aware control.

### 50. Universal Morphology Control via Contextual Modulation
- **Authors**: Zheng Xiong et al.
- **ArXiv**: 2302.11070 (Feb 2023, ICML 2023)
- **Summary**: Universal policy across different robot morphologies using contextual modulation of neural network weights based on robot structure.

---

## Category 11: Reward Design & Safety

### 51. Entropy-Controlled Intrinsic Motivation for Quadruped Locomotion
- **Authors**: Wanru Gong et al.
- **ArXiv**: 2512.06486 (Dec 2025)
- **Summary**: Extends PPO with entropy-controlled intrinsic motivation to improve exploration and locomotion on complex terrains.

### 52. Risk-Aware RL with Bandit-Based Adaptation
- **Authors**: Yuanhong Zeng, Anushri Dixit
- **ArXiv**: 2510.14338 (Oct 2025)
- **Summary**: Risk-aware reinforcement learning for quadruped locomotion that adapts behavior based on estimated risk using multi-armed bandit techniques.

### 53. VOCALoco: Viability-Optimized Cost-Aware Adaptive Locomotion
- **Authors**: Stanley Wu et al.
- **ArXiv**: 2510.23997 (Oct 2025, RAL)
- **Summary**: Adaptive locomotion framework that optimizes viability and cost-awareness for traversing increasingly complex legged robot terrains.

### 54. FACET: Force-Adaptive Control via Impedance Reference Tracking
- **Authors**: Botian Xu et al.
- **ArXiv**: 2505.06883 (May 2025)
- **Summary**: Force-adaptive control with learned impedance reference tracking for legged robots, enabling better ground reaction force control.

### 55. Symmetry-Guided Memory Augmentation
- **Authors**: Kaixi Bao et al.
- **ArXiv**: 2502.01521 (Feb 2025)
- **Summary**: Uses morphological symmetry and memory augmentation to improve locomotion learning efficiency, reducing required training samples and improving policy quality.

---

## Key Architectural Insights for Implementation

Based on this survey, the following architectural innovations are most applicable to our Mini Cheetah project:

1. **Hierarchical Transformer Architecture** (from ULT, TERT, SET): Replace MLP with multi-level transformer that processes temporal sequences of observations
2. **Masked Sensory-Temporal Attention** (from MSTA): Structured attention over different sensor modalities
3. **Mixture of Experts** (from MoE-Loco): Multiple expert sub-networks with gating for terrain-specific behaviors
4. **Adaptive Curriculum** (from LP-ACRL): Learning progress-based curriculum instead of fixed threshold progression
5. **Denoising World Model** (from DWL): Auxiliary world model prediction head for better representation learning
6. **Morphological Symmetry** (from MS-PPO): Encode quadruped leg symmetry to reduce parameter space
7. **Contrastive Adaptation** (from contrastive sim-to-real): Contrastive learning for robust domain transfer
