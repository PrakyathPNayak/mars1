---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/real_world_humanoid_locomotion_reinforcement_learning.md

**Title:** Real-World Humanoid Locomotion with Reinforcement Learning
**Authors:** Ilija Radosavovic, Tete Xiao, Bike Zhang, Trevor Darrell, Jitendra Malik, Koushil Sreenath
**Year:** 2024
**Venue:** Science Robotics
**arXiv / DOI:** Available via Berkeley Hybrid Robotics

**Abstract Summary (2-3 sentences):**
This paper introduces a causal transformer policy trained with reinforcement learning for humanoid locomotion, achieving zero-shot real-world deployment over challenging indoor and outdoor terrains on the Digit humanoid robot. The transformer-based policy processes sequential proprioceptive observations with causal attention, enabling the model to implicitly perform system identification and adapt to varying terrain conditions through its context window. This represents the first successful deployment of a transformer-based locomotion policy on a full-sized humanoid in unstructured real-world environments.

**Core Contributions (bullet list, 4-7 items):**
- First transformer-based locomotion policy deployed on a real-world humanoid robot
- Causal transformer architecture processing sequential observations for implicit system identification
- Zero-shot deployment across diverse real-world terrains including grass, gravel, concrete, and indoor surfaces
- Demonstration of robustness to unseen disturbances such as external pushes and unmodeled terrain features
- Comprehensive comparison with MLP and LSTM baselines showing transformer superiority in generalization
- Analysis of attention patterns revealing that the transformer learns to attend to dynamically relevant history
- Scaling study showing improved performance with larger transformer models and longer context windows

**Methodology Deep-Dive (3-5 paragraphs):**
The core architecture is a causal (decoder-only) transformer that receives a sequence of proprioceptive observation tokens and outputs action tokens at each timestep. Each observation token consists of joint positions, joint velocities, pelvis orientation (quaternion), pelvis angular velocity, and the commanded velocity target, all concatenated and projected through a linear embedding layer. The causal attention mask ensures that each output action depends only on current and past observations, maintaining the autoregressive property necessary for real-time control. The transformer uses 4-8 attention layers with 4 attention heads and an embedding dimension of 128-256, with the context window spanning 50-200 timesteps of history at 50 Hz control frequency.

Training follows a two-phase approach combining privileged learning with RL fine-tuning. In the first phase, a teacher policy with access to privileged information (ground truth terrain height map, exact contact states, true body dynamics parameters) is trained using PPO in Isaac Gym. This teacher achieves strong locomotion performance by leveraging information unavailable during deployment. In the second phase, the causal transformer student is trained to match the teacher's behavior using observation histories only, through a combination of behavior cloning on teacher rollouts and subsequent PPO fine-tuning. The fine-tuning phase is critical because pure behavior cloning does not adequately capture the teacher's robustness to distribution shifts. The RL fine-tuning uses the same reward function as the teacher but operates on the transformer's observation-only input.

The reward function for training combines velocity tracking (forward, lateral, and yaw rate), orientation stability (penalizing deviations from upright posture), action smoothness (penalizing large changes between consecutive actions), energy efficiency (penalizing total joint power consumption), and contact regularity (encouraging periodic foot contact patterns consistent with walking gaits). Domain randomization during training covers ground friction (0.3-2.0), body mass (plus or minus 20%), center of mass offsets, motor strength scaling, observation delays (0-3 timesteps), and terrain variations including slopes (up to 15 degrees), stairs, and random rough terrain. The terrain curriculum progressively increases difficulty as the policy improves, ensuring the policy is exposed to challenging conditions without destabilizing early training.

The transformer's attention mechanism provides a principled way to handle the varying relevance of historical observations. Analysis of the learned attention patterns reveals that the transformer selectively attends to observations at specific temporal offsets corresponding to the gait cycle period, ground contact events, and recent perturbation events. This stands in contrast to RNN-based approaches where the temporal filtering is implicit and not easily interpretable. The attention analysis also shows that different layers specialize in different temporal scales: early layers attend to very recent observations for reactive control, while deeper layers attend to longer-range patterns for terrain and dynamics estimation.

Real-world deployment runs the transformer policy at 50 Hz on Digit's onboard NVIDIA Jetson computing module. The context window is maintained as a rolling buffer of the most recent observations, with older entries discarded as new observations arrive. Inference time is approximately 3-5 ms per step for the largest transformer variant tested, leaving substantial margin within the 20 ms control cycle. The authors evaluate on multiple real-world surfaces including laboratory tile, outdoor concrete, grass, gravel paths, and foam mats, demonstrating consistent walking performance across all conditions without any terrain-specific adaptation.

**Key Results & Numbers:**
- Zero-shot deployment on Digit across 6+ distinct real-world terrain types
- Robust to unseen perturbations including lateral pushes up to 80 N
- Transformer policy outperforms MLP baseline by 35% on terrain generalization metrics
- Transformer outperforms LSTM baseline by 15% on perturbation robustness
- Inference latency of 3-5 ms on NVIDIA Jetson (well within 20 ms control cycle)
- Context window of 50-200 timesteps provides optimal performance
- Scaling from 4 to 8 transformer layers improves success rate by 12% on rough terrain
- Walking speeds up to 0.8 m/s on flat ground, 0.4 m/s on challenging terrain

**Relevance to Project A (Mini Cheetah):** MEDIUM — The transformer policy architecture is directly transferable to quadruped locomotion. The causal attention mechanism, context window design, and teacher-student training pipeline could improve the Mini Cheetah's adaptation capability. However, the specific terrain challenges and balance requirements differ for quadrupeds.

**Relevance to Project B (Cassie HRL):** HIGH — Directly informs the Dual Asymmetric-Context Transformer design for the Cassie hierarchical controller. The causal transformer architecture, attention pattern analysis, and scaling properties provide concrete design guidance. The teacher-student training with privileged information is directly applicable to the privileged learning component of the hierarchy.

**What to Borrow / Implement:**
- Causal transformer architecture as the backbone for the Dual Asymmetric-Context Transformer
- Two-phase training pipeline (privileged teacher then transformer student with RL fine-tuning)
- Attention pattern analysis methodology for understanding what the transformer learns
- Context window length tuning results (50-200 steps at 50 Hz) as starting points for the asymmetric windows
- Terrain curriculum design for progressive difficulty scaling
- Inference latency benchmarks as targets for the Cassie onboard deployment

**Limitations & Open Questions:**
- Requires significant compute for transformer training (more than MLP or LSTM baselines)
- Limited to locomotion only; no manipulation or interaction tasks
- Inference latency of 3-5 ms, while acceptable, is higher than sub-1 ms MLP inference
- Context window length is a fixed hyperparameter that may need terrain-specific tuning
- Scaling benefits plateau beyond 8 layers, suggesting architectural limits
- No explicit safety guarantees or constraint enforcement during deployment
- The teacher-student paradigm introduces complexity and potential information loss in distillation
---
