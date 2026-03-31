# NavThinker: Action-Conditioned World Models for Coupled Prediction and Planning in Social Navigation

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [AlphaXiv](https://www.alphaxiv.org/overview/2603.15359v2)

---

## Abstract Summary
NavThinker proposes a world-model-based planning framework for quadruped robot social navigation in human-populated environments. The core idea is to learn an action-conditioned world model that predicts future states of both the robot and surrounding humans given a sequence of planned actions, then use this model for look-ahead planning that accounts for human-robot interactions. Unlike reactive policies that map current observations to actions, NavThinker reasons about the consequences of actions multiple steps into the future, enabling proactive navigation that avoids conflicts before they arise.

The world model is structured as a latent dynamics model that takes the current state embedding and a candidate action sequence, then autoregressively predicts future latent states. These predicted states are decoded to estimate future robot poses, human trajectories, and collision probabilities. The planning module optimizes action sequences by evaluating them through the world model, selecting the sequence that maximizes navigation progress while minimizing predicted social costs (collision risk, personal space violations, path disruption to humans).

The framework is deployed and validated on a real Unitree Go2 quadruped robot navigating in indoor environments with multiple moving humans. Experimental results show that NavThinker achieves significantly smoother and more socially compliant navigation compared to reactive baselines, with fewer near-collisions, less path disruption to pedestrians, and faster overall navigation times in crowded scenarios.

## Core Contributions
- Action-conditioned world model that jointly predicts robot state evolution and human trajectory responses to robot actions
- Coupled prediction and planning: world model predictions directly feed into action optimization for socially-aware navigation
- Latent dynamics model architecture with separate encoders for robot proprioception, human observations, and scene context
- Model-predictive planning that evaluates candidate action sequences through multi-step world model rollouts
- Real-world deployment on Unitree Go2 quadruped in indoor environments with multiple pedestrians
- Social cost functions that penalize personal space violations, path disruption, and unexpected robot behavior
- Demonstration that world-model-based planning outperforms reactive policies in crowded, dynamic environments

## Methodology Deep-Dive
The world model consists of three components: an encoder, a latent dynamics model, and a decoder. The encoder processes multi-modal observations—robot proprioception (joint states, IMU, odometry), human detections from an onboard camera (bounding boxes, estimated positions and velocities), and scene context (local occupancy grid)—into a compact latent state vector z_t ∈ ℝ^256. The latent dynamics model is a recurrent network (GRU-based) that predicts z_{t+1} = f(z_t, a_t), where a_t is the robot's action (commanded velocity). The decoder reconstructs observable quantities from latent states: predicted robot pose, predicted human positions, and predicted collision probability.

The world model is trained on data collected from both simulation and real-world operation. The simulation component uses Isaac Sim with procedurally generated indoor environments and ORCA-based pedestrian simulation. Real-world data is collected during teleoperated navigation sessions in office environments. The training objective combines reconstruction losses (MSE on predicted robot pose and human positions) with a contrastive loss that shapes the latent space to cluster similar interaction scenarios.

Planning uses the Model-Predictive Path Integral (MPPI) algorithm. At each planning step, MPPI samples N = 1000 candidate action sequences of horizon H = 20 steps (corresponding to 4 seconds at 5 Hz planning frequency). Each candidate sequence is evaluated by rolling it out through the world model, producing a sequence of predicted latent states. The cost function evaluates each rollout based on: (1) navigation progress—distance reduction to the goal; (2) collision cost—predicted collision probabilities summed over the horizon; (3) social cost—predicted personal space violations weighted by severity; (4) smoothness cost—penalizes abrupt velocity changes; (5) energy cost—penalizes high velocities. MPPI combines costs across candidates using a softmax weighting to produce the optimal action.

The world model is recurrently updated at runtime: as new observations arrive, the latent state is refined through the encoder, and the model's predictions are compared against actual outcomes to maintain an online calibration of prediction uncertainty. High uncertainty triggers more conservative planning (larger social distances, slower speeds).

The locomotion layer beneath NavThinker uses a pre-trained PPO policy that tracks velocity commands, similar to the standard quadruped RL pipeline. NavThinker operates at the navigation/planning level, generating velocity commands that the locomotion policy executes. This separation allows the locomotion policy to be trained independently with standard methods while NavThinker handles the higher-level social navigation challenge.

The system architecture on the Go2 hardware consists of: (1) a perception module running on the Jetson GPU that processes camera images for human detection and tracking; (2) the NavThinker planning module running at 5 Hz that generates velocity commands; (3) the locomotion PPO policy running at 50 Hz that executes commands. All components communicate via ROS2.

## Key Results & Numbers
- 40% fewer near-collision events compared to reactive DRL navigation baselines in crowded scenarios (>5 humans)
- 25% reduction in path disruption to pedestrians (measured by human trajectory deviation caused by robot presence)
- 15% faster navigation time in medium-density environments (3–5 humans) due to proactive path planning
- World model prediction accuracy: <0.3m robot position error at 2-second horizon, <0.5m at 4-second horizon
- Human trajectory prediction accuracy: <0.4m at 2-second horizon (comparable to dedicated trajectory prediction models)
- Planning frequency: 5 Hz on Jetson Orin, with 1000 MPPI samples per planning step
- End-to-end latency (perception → planning → locomotion command): <50ms

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
NavThinker demonstrates a complete world-model-based planning stack deployed on a real quadruped robot, providing a reference architecture for extending the Mini Cheetah beyond pure locomotion to navigation tasks. The two-level architecture (NavThinker planner → PPO locomotion policy) could be adopted for Mini Cheetah, where the locomotion policy is the primary project deliverable and NavThinker-style planning could be a future extension.

The world model training methodology—combining simulation data with real-world data—is relevant for Mini Cheetah's sim-to-real pipeline. Even for pure locomotion, learning a dynamics world model could serve as an auxiliary training signal that improves the quality of the PPO locomotion policy.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
NavThinker's world model architecture is directly relevant to the RSSM-based Planner at the top of Cassie's 4-level hierarchy. The action-conditioned latent dynamics model (GRU-based, with encoder/dynamics/decoder structure) closely resembles RSSM (Recurrent State-Space Model), which Cassie's Planner uses for imagined rollouts and long-horizon planning. The MPPI-based planning over world model rollouts provides a concrete algorithm for Cassie's Planner to generate high-level goals.

The hierarchical separation (NavThinker planner → locomotion policy) mirrors the separation between Cassie's Planner and lower levels (Primitives → Controller → Safety). The real-world deployment at 5 Hz planning / 50 Hz locomotion frequency provides a validated frequency separation that could inform Cassie's hierarchy timing design.

## What to Borrow / Implement
- GRU-based latent dynamics model architecture as a reference for Cassie's RSSM Planner implementation
- MPPI planning algorithm with N=1000 samples and H=20 horizon as a baseline planning strategy
- Two-level frequency separation (5 Hz planner, 50 Hz controller) as a design template for hierarchical control timing
- Online world model calibration via prediction-vs-reality comparison for adaptive planning conservatism
- Combined simulation + real-world data training for world model robustness

## Limitations & Open Questions
- World model trained primarily for social navigation; generalization to other planning domains (terrain traversal, obstacle avoidance) untested
- GRU-based dynamics model may have limited capacity for very long horizons (>5 seconds); Cassie's RSSM with stochastic latent states may handle uncertainty better
- MPPI planning is computationally expensive; scaling to higher-dimensional action spaces (e.g., Cassie's full skill-selection + locomotion) may require more efficient planning algorithms
- The locomotion policy beneath NavThinker is a standard flat MLP-PPO; no investigation of how GNN-based or hierarchical locomotion policies interact with world-model planning
