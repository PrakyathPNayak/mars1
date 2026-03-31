---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/genloco_generalized_locomotion_controllers.md

**Title:** GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots
**Authors:** Gilbert Feng, Hongbo Zhang, Zhongyu Li, Xue Bin Peng, Beren Millidge, Zhitao Song, Koushil Sreenath
**Year:** 2022
**Venue:** CoRL 2022 (Conference on Robot Learning)
**arXiv / DOI:** arXiv:2209.05309

**Abstract Summary (2–3 sentences):**
GenLoco presents a framework for training generalized locomotion controllers that transfer zero-shot across quadrupedal robots with different morphologies. By randomizing robot morphology parameters during training, a single policy can control diverse commercial quadrupeds (Unitree A1, Go1, ANYmal-B, ANYmal-C, Spot, Mini Cheetah, and others) without robot-specific retraining.

**Core Contributions (bullet list, 4–7 items):**
- Morphology randomization during training for cross-robot generalization
- Single policy controlling 10+ different quadruped platforms zero-shot
- Morphology-conditioned policy architecture with robot descriptor input
- Successful real-world deployment on Spot and A1 without per-robot fine-tuning
- Demonstration of multiple gaits (pacing, spinning, trotting) across diverse robots
- Open-source code and pretrained policies
- Significant reduction in engineering effort for deploying on new platforms

**Methodology Deep-Dive (3–5 paragraphs):**
GenLoco addresses the impractical reality that most RL locomotion controllers are robot-specific and require complete retraining for each new platform. The key idea is morphology randomization: during training in simulation, the system procedurally generates a population of "virtual" quadrupeds with varied limb lengths, masses, joint limits, and body proportions. This forces the policy to learn locomotion strategies that are robust to morphological variation.

The policy architecture takes as input the robot's proprioceptive state concatenated with a "morphology descriptor" — a compact vector encoding the robot's physical parameters (leg lengths, masses, joint ranges). This descriptor provides the policy with explicit knowledge of the current robot's body, allowing it to adapt its behavior accordingly. Training uses RL with reference motion imitation, where the policy learns to track reference locomotion trajectories while maintaining balance and stability.

The training pipeline uses Isaac Gym for massively parallel simulation. During each episode, morphology parameters are randomly sampled from a distribution designed to cover the range of commercially available quadrupeds. The reward function combines motion imitation (matching reference joint trajectories), velocity tracking, and stability penalties. Domain randomization additionally covers terrain, friction, and sensor noise.

At deployment, the morphology descriptor for a specific robot is computed from its URDF specification and provided as a constant input to the policy. This allows the same neural network weights to control any quadruped within the training distribution without retraining or fine-tuning.

**Key Results & Numbers:**
- Zero-shot deployment on 10 commercially available quadruped robots
- Real-world testing on Boston Dynamics Spot and Unitree A1 confirmed successful transfer
- Cross-robot generalization maintained >85% of robot-specific policy performance
- Training time: ~2 hours on a single GPU for morphology-randomized policies
- Supported gaits: pacing, trotting, spinning across all tested platforms

**Relevance to Project A (Mini Cheetah):** HIGH — Mini Cheetah is one of the test platforms; the morphology-conditioned approach could enable rapid policy deployment and comparison across quadruped variants.
**Relevance to Project B (Cassie HRL):** LOW — Focused on quadruped morphology generalization, less directly applicable to bipedal locomotion.

**What to Borrow / Implement:**
- Use morphology randomization during training to improve robustness to parameter uncertainty in Mini Cheetah
- The morphology descriptor concept could be adapted for parameterizing different Mini Cheetah configurations (e.g., with/without payloads)
- Leverage the open-source codebase as a reference for multi-robot training pipelines

**Limitations & Open Questions:**
- Performance degrades for robots whose morphology falls outside the training distribution
- Does not address robots with different kinematic structures (e.g., hexapods, bipeds)
- Reference motion imitation requires pre-existing locomotion trajectories for each gait
---
