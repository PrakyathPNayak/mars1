# Domain-Aware Behavior Cloning for Bridging the Sim-to-Real Gap of Legged Robots

**Authors:** Zhou et al.
**Year:** 2025 | **Venue:** Science China Information Sciences (Springer)
**Links:** [Springer](https://link.springer.com/article/10.1007/s11432-024-4755-6)

---

## Abstract Summary
This paper introduces a domain-aware behavior cloning (DABC) framework that addresses the sim-to-real gap by combining adaptive reinforcement learning with behavior cloning in a domain-conditioned manner. Unlike standard behavior cloning which naively imitates expert demonstrations regardless of domain shift, DABC explicitly conditions the student policy on domain features extracted from observation history, enabling the cloned policy to adapt its behavior based on detected environmental characteristics.

The core architecture consists of three components: (1) a domain encoder that maps observation history to a compact domain embedding capturing physical parameters like friction, mass, and actuator properties, (2) a teacher RL policy trained with privileged domain information in simulation, and (3) a student BC policy that learns to replicate teacher behavior conditioned on the inferred domain embedding. The domain encoder is trained via a contrastive objective that ensures physically similar environments cluster in embedding space while dissimilar environments are separated.

The framework demonstrates robust sim-to-real transfer on quadruped locomotion tasks, handling unseen terrain types and payload conditions that were not present during training. The online adaptation capability—where the domain encoder continuously updates its estimate from recent observations—allows the policy to handle dynamic environmental changes such as transitioning from hard floor to soft ground or sudden payload additions.

## Core Contributions
- Domain-aware behavior cloning framework that explicitly conditions policy on inferred domain features
- Contrastive domain encoder trained to produce physically meaningful embeddings from observation history
- Teacher-student training paradigm where the teacher uses privileged simulation information and the student uses only real-world observable quantities
- Online domain adaptation through continuous domain embedding updates from rolling observation windows
- Robust state estimation module that filters noisy sensor readings for reliable domain inference
- Demonstrated transfer to unseen terrain types and dynamic environmental changes
- Complementary approach to physics-based sim-to-real methods, providing a learning-based alternative

## Methodology Deep-Dive
The teacher policy is trained in simulation using PPO with access to privileged information: ground-truth friction coefficients, terrain height maps, exact body mass, and actuator parameters. This privileged policy achieves near-optimal performance because it can perfectly adapt to any domain configuration. The observation space includes proprioceptive measurements (joint positions q, joint velocities q̇, body orientation R, angular velocity ω) plus privileged domain parameters d_priv that are not available on the real robot.

The domain encoder f_φ processes a history window of T timesteps of proprioceptive observations {o_{t-T}, ..., o_t} through a temporal convolutional network (TCN) to produce a domain embedding z_d ∈ R^k (typically k=16-32). The TCN architecture uses causal convolutions to maintain temporal ordering, with kernel sizes of 3-5 and 4-6 layers. The encoder is trained with a contrastive loss: trajectories collected under similar domain parameters should produce similar embeddings (positive pairs), while trajectories from different domains should produce dissimilar embeddings (negative pairs). The similarity metric uses cosine distance, and the loss follows the InfoNCE formulation with a temperature parameter τ=0.1.

The student policy π_θ(a|o_t, z_d) takes the current observation o_t concatenated with the domain embedding z_d and outputs actions. It is trained via behavior cloning on a dataset of (observation, domain_embedding, teacher_action) tuples collected by rolling out the teacher policy across diverse domain configurations. The BC loss is the standard MSE between student and teacher actions: L_BC = E[||π_θ(o_t, z_d) - π_teacher(o_t, d_priv)||²]. Crucially, the domain embedding z_d acts as a surrogate for the privileged information d_priv, enabling the student to approximate teacher-level adaptation using only observable quantities.

Online adaptation works by maintaining a sliding window of the most recent T observations (typically T=50-100 at 50 Hz control frequency, representing 1-2 seconds of history). At each timestep, the domain encoder processes this window to produce an updated z_d, which the policy uses for the current action. This creates a feedback loop where the policy's behavior influences future observations, which in turn influence domain estimation. Stability is ensured by using exponential moving average smoothing on z_d with decay factor α=0.95.

The robust state estimation module preprocesses raw sensor data before feeding it to the domain encoder. It implements an Extended Kalman Filter (EKF) that fuses IMU accelerometer/gyroscope data with joint encoder readings to produce clean estimates of body velocity, orientation, and foot contact states. This filtering is critical for domain estimation because noisy observations can cause the domain encoder to produce unstable embeddings, leading to erratic policy behavior.

## Key Results & Numbers
- 85-92% success rate on unseen terrain types (gravel, foam, wet surfaces) vs. 60-70% for standard BC
- Domain embedding clustering accuracy: 89% correct domain identification from observation history alone
- Online adaptation latency: ~0.5 seconds to converge on new domain embedding after environmental change
- Tracking error (joint RMSE): 0.05 rad with DABC vs. 0.12 rad with standard BC on real robot
- Teacher policy achieves 95% success with privileged info; student achieves 88% using domain encoder
- Works with observation history windows of 50-100 timesteps (1-2 seconds)
- Contrastive encoder produces 16-32 dimensional domain embeddings

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

The DABC framework provides a compelling alternative to pure RL-based sim-to-real transfer for the Mini Cheetah. The teacher-student paradigm is particularly attractive: train a privileged teacher in MuJoCo with access to ground-truth physics parameters, then distill into a deployable student that uses only onboard sensors. The domain encoder's ability to infer environmental conditions from proprioceptive history alone is valuable for the Mini Cheetah, which may operate across diverse surfaces without exteroceptive sensing.

The contrastive domain encoder architecture can be directly integrated into the Mini Cheetah's PPO training pipeline. During simulation, collect trajectories across randomized domains, train the encoder with contrastive loss, then use the domain embedding as additional policy input. This approach subsumes some of the benefits of domain randomization (robustness) while adding explicit adaptation (performance).

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchical system, DABC's domain encoder concept maps naturally to the Controller level, where accurate domain estimation is critical for the CBF-QP safety filter. The domain embedding can provide the safety layer with real-time estimates of friction and surface properties, enabling adaptive constraint tightening—using wider safety margins on slippery surfaces and tighter margins on high-friction surfaces.

The teacher-student paradigm also applies at the Primitives level: train privileged primitive policies with DIAYN/DADS using ground-truth domain information, then distill into deployable primitives conditioned on inferred domain embeddings. This ensures that skill selection and execution account for current environmental conditions. The online adaptation capability (0.5s convergence) is fast enough for Cassie's real-time operation.

## What to Borrow / Implement
- Implement a contrastive domain encoder (TCN, 16-32 dim embedding) trained on diverse MuJoCo domain configurations
- Use the teacher-student framework: privileged PPO teacher → domain-conditioned BC student for Mini Cheetah
- Integrate domain embeddings into Cassie's CBF-QP safety filter for adaptive constraint tightening
- Adopt the EKF-based state estimation preprocessing to clean sensor inputs before domain inference
- Apply exponential moving average smoothing (α=0.95) on domain embeddings for stable online adaptation

## Limitations & Open Questions
- Behavior cloning is fundamentally limited by the quality of the teacher policy; errors compound in out-of-distribution states
- Contrastive domain encoder may fail to distinguish between domains that produce similar proprioceptive signatures (e.g., low friction vs. high damping)
- Online adaptation convergence time (0.5s) may be too slow for sudden catastrophic changes (e.g., stepping on ice)
- The approach requires extensive simulation data collection across many domain configurations for encoder training
