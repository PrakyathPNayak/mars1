# ALARM: Adaptive Learning with Asymmetric RL for Safe Robust Locomotion

**Authors:** (SUSTech Robotics Group)
**Year:** 2024 | **Venue:** arXiv / Conference
**Links:** [Project Page](https://sucro-legged.github.io/ALARM/)

---

## Abstract Summary
ALARM (Adaptive Learning with Asymmetric RL for Manipulation and locomotion) presents a framework combining imitation learning with reinforcement learning for training robust, proprioception-driven locomotion controllers that transfer efficiently from simulation to real hardware. The core architectural innovation is an asymmetric training setup where the teacher policy has access to privileged information (ground-truth terrain maps, exact contact forces, full state) during training, while the student policy receives only proprioceptive observations (joint encoders and IMU) available on real hardware. The student learns to match the teacher's behavior through a combination of behavior cloning and RL fine-tuning.

This asymmetric paradigm addresses a fundamental challenge in sim-to-real transfer for locomotion: in simulation, perfect state information enables training of highly capable policies, but this information is unavailable on real robots. Rather than constraining the training policy to only use real-world-available sensors (which limits performance), ALARM trains with full information and then distills the knowledge into a deployable student policy. The distillation process preserves the teacher's robustness while adapting to the observation constraints of the real system.

The framework achieves safe and robust locomotion across diverse terrains (stairs, slopes, rubble, deformable surfaces) without any exteroceptive sensing (cameras, LiDAR). The proprioceptive-only student policy demonstrates remarkable terrain adaptation capabilities, having implicitly learned to infer terrain properties from the dynamic response of the robot's legs. Real-world deployment is demonstrated on a quadruped platform, showing successful sim-to-real transfer with minimal performance degradation.

## Core Contributions
- Asymmetric teacher-student framework with privileged teacher training and proprioceptive-only student deployment
- Hybrid imitation learning + RL fine-tuning for student policy training, combining behavior cloning stability with RL adaptability
- Safe locomotion across diverse terrains using only proprioceptive sensing (no cameras, LiDAR, or depth sensors)
- Efficient sim-to-real transfer with minimal performance gap between simulation and real hardware
- Implicit terrain estimation through proprioceptive feedback, enabling adaptation without explicit terrain mapping
- Robust disturbance rejection demonstrated under external pushes and payload variations
- Comprehensive real-world deployment results across indoor and outdoor environments

## Methodology Deep-Dive
The ALARM framework consists of three training phases: privileged teacher training, behavior cloning initialization, and RL fine-tuning of the student.

**Phase 1: Privileged Teacher Training.** A teacher policy π_teacher(a|s_priv) is trained using PPO in simulation with access to a privileged observation vector s_priv that includes: (1) robot proprioception (joint angles, velocities, body orientation and angular velocity), (2) ground-truth linear velocity, (3) terrain height map (a local 10×10 grid of terrain heights centered on the robot, sampled at 5cm resolution), (4) exact contact forces on each foot, and (5) friction coefficient at each contact point. This rich observation enables the teacher to learn terrain-adaptive locomotion strategies that explicitly reason about upcoming obstacles and surface properties. The teacher is trained across a diverse terrain curriculum including flat ground, random rough terrain, stairs (up and down), slopes (±20°), and discrete obstacles.

**Phase 2: Behavior Cloning Initialization.** The student policy π_student(a|s_prop) receives only proprioceptive observations: joint angles, joint velocities, body orientation (rotation matrix from IMU), body angular velocity, previous action, and a history of the last 50 proprioceptive observations. The observation history is critical—it allows the student to implicitly estimate terrain properties from the dynamic response patterns. The student is initialized via supervised learning (behavior cloning) on the teacher's rollout data, minimizing ||π_student(s_prop) - π_teacher(s_priv)||² over a dataset of (s_prop, s_priv, a_teacher) tuples collected from teacher rollouts. This provides a strong initialization before RL fine-tuning.

**Phase 3: RL Fine-Tuning.** The initialized student is fine-tuned using PPO with the original task reward plus an imitation reward component: r_total = r_task + β · r_imitation, where r_imitation = -||π_student(s_prop) - a_teacher||². The imitation reward keeps the student close to the teacher's behavior while allowing RL to discover adaptations specific to the limited observation space. The mixing coefficient β decays over training, gradually releasing the student from teacher imitation and allowing it to develop its own proprioceptive strategies.

A key architectural detail is the observation encoder: the 50-step proprioceptive history is processed through a 1D temporal convolution network (TCN) with 3 layers, producing a fixed-size latent embedding that captures temporal patterns (gait phase, velocity trends, terrain roughness indicators). This embedding is concatenated with the current proprioceptive state and fed into the policy MLP. The TCN architecture was chosen over RNNs for its parallelizable training and ability to capture patterns at multiple temporal scales.

For safety, the training includes a constraint cost function that penalizes: (1) joint torques exceeding 80% of hardware limits, (2) joint velocities exceeding safe thresholds, (3) body roll or pitch exceeding 45°, and (4) foot slip velocity exceeding a threshold. The constraint cost is incorporated via Lagrangian relaxation, adding a penalty term to the reward with an adaptively tuned multiplier. This ensures the student policy remains within safe operating bounds even when exploring novel proprioceptive strategies during RL fine-tuning.

Domain randomization is applied extensively: mass (±20%), friction (0.3–1.5), motor strength (±15%), observation noise (Gaussian σ = 0.05), and control latency (0–20ms uniform). The training uses 8192 parallel environments for 1B total time steps, completing in approximately 24 hours on 4 GPUs.

## Key Results & Numbers
- Sim-to-real success rate: 92% gait stability across real-world terrains (stairs, gravel, grass, carpet)
- Student vs. teacher performance gap: <8% in average velocity tracking error, <5% in stability metrics
- Behavior cloning alone: 65% sim-to-real success rate; BC + RL fine-tuning: 92% (27 percentage point improvement)
- Recovery from 150N lateral push: successful in 95% of trials (real hardware)
- Payload variation: maintains stable locomotion with ±30% body mass change
- Observation history ablation: 50-step history optimal; 10-step reduces terrain adaptation by 30%; 100-step adds marginal benefit
- Training time: 24 hours on 4 GPUs (8192 parallel environments, 1B steps)
- Real-world velocity tracking: 0.08 m/s RMS error on flat ground, 0.15 m/s on stairs

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
ALARM's asymmetric teacher-student framework is directly applicable to Mini Cheetah sim-to-real transfer. Training a privileged teacher with terrain height maps and exact contact forces in MuJoCo, then distilling into a proprioceptive student, is a proven approach for quadruped deployment. The hybrid BC + RL fine-tuning methodology is more sample-efficient than training the student from scratch with RL alone, reducing the sim-to-real iteration cycle.

The proprioceptive-only deployment is particularly attractive for Mini Cheetah, which has limited onboard compute and may not support real-time depth processing. The implicit terrain estimation through observation history is well-suited for Mini Cheetah's sensor suite. The safety constraints via Lagrangian relaxation could be adopted to protect Mini Cheetah's hardware from policy-induced damage during early deployment trials.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
ALARM's asymmetric training paradigm directly aligns with the Dual Asymmetric-Context Transformer in Project B's architecture. The concept of a privileged context (terrain maps, contact forces) available during training but not deployment is the same architectural principle. The key difference is that Project B uses a transformer-based encoder for the asymmetric context rather than a TCN, but the underlying methodology transfers directly.

The safety constraints via Lagrangian relaxation are relevant to the Safety level (LCBF) of the Cassie hierarchy. ALARM's approach could be used to pre-train the Safety module's constraint awareness before integrating it with the LCBF framework. The observation history mechanism (50-step buffer processed by TCN) is analogous to the temporal encoding in the Dual Asymmetric-Context Transformer and could inform the design of the history encoder architecture. The 3-phase training process (privileged teacher → BC initialization → RL fine-tuning) could be applied to each level of the Cassie hierarchy, with higher levels receiving more abstract privileged information.

## What to Borrow / Implement
- Adopt the 3-phase training pipeline (privileged teacher → BC → RL fine-tuning) for Mini Cheetah sim-to-real transfer
- Use the TCN-based observation history encoder (50 steps) for proprioceptive-only policies in both projects
- Implement Lagrangian safety constraints during training to prevent hardware-damaging behaviors
- Apply the asymmetric teacher-student concept across all levels of Cassie's hierarchy, with level-specific privileged information
- Use the β-decay schedule for imitation reward to smoothly transition from teacher imitation to independent RL exploration

## Limitations & Open Questions
- The teacher-student gap (8% in velocity tracking) represents information irrecoverably lost when moving from privileged to proprioceptive observations; tighter gaps may require architectural innovations
- The 50-step observation history adds significant input dimensionality; scaling to more complex robots or longer histories may require attention-based architectures
- Safety constraints are enforced during training but not guaranteed at deployment; adversarial disturbances exceeding training distribution may violate constraints
- Real-world results are demonstrated on a single quadruped platform; transfer to bipedal (Cassie-like) robots is not validated
