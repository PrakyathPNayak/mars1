# Imitation Learning for Legged Robot Locomotion: A Survey

**Authors:** Various
**Year:** 2025 | **Venue:** Frontiers in Robotics and AI
**Links:** https://www.frontiersin.org/articles/10.3389/frobt.2025.1678567

---

## Abstract Summary
This comprehensive survey covers the landscape of imitation learning (IL) techniques applied to legged robot locomotion. It reviews behavioral cloning (BC), DAgger, Generative Adversarial Imitation Learning (GAIL), adversarial motion priors (AMP), diffusion models for multi-modal policy generation, and hybrid RL-IL approaches. The paper examines integration of multiple motion skills, motion retargeting across morphologies, and real-world transfer on quadruped and humanoid platforms.

## Core Contributions
- Provides a unified taxonomy of imitation learning methods for legged locomotion (BC, DAgger, GAIL, AMP, diffusion-based)
- Surveys hybrid RL-IL approaches that combine reward shaping with demonstration data
- Reviews multi-skill integration methods for composing locomotion behaviors
- Analyzes motion retargeting pipelines from human to robot morphologies
- Identifies diffusion-based multi-modal policy generation as an emerging frontier
- Discusses real-world transfer challenges and solutions across quadruped and humanoid platforms
- Highlights open problems in sample efficiency, multi-terrain generalization, and long-horizon skill sequencing

## Methodology Deep-Dive
The survey organizes IL methods along several axes. At the simplest level, behavioral cloning (BC) treats locomotion as supervised learning from expert state-action pairs. The paper discusses BC's distribution shift problem and how DAgger addresses it through interactive data aggregation, where the expert labels states visited by the learner during rollout. The authors provide a thorough comparison of convergence guarantees and practical failure modes for each approach.

Adversarial methods receive significant attention. GAIL trains a discriminator to distinguish expert from policy trajectories, using the discriminator's output as a reward signal. AMP extends this by matching style features from reference motions, allowing natural-looking gaits without hand-crafted rewards. The survey traces the evolution from GAIL to AMP and discusses how these methods handle the reward sparsity problem inherent in locomotion.

Diffusion models represent the newest paradigm covered. By framing policy output as a conditional denoising process, diffusion policies can represent multi-modal action distributions—critical for locomotion where multiple valid gaits may exist for the same observation. The survey compares DDPM and score-based formulations and their trade-offs in inference speed vs. action quality.

The hybrid RL-IL section discusses methods that use demonstrations to warm-start RL, shape reward functions, or constrain the policy search space. These approaches aim to combine the sample efficiency of IL with the performance optimization of RL. The paper reviews residual policy learning, demo-augmented replay buffers, and reward shaping from demonstrations.

Finally, the survey covers motion retargeting—translating human motion capture data to robot joint commands despite morphological differences. Methods range from kinematic optimization to learned latent-space approaches, with discussion of how retargeting quality affects downstream policy performance.

## Key Results & Numbers
- Taxonomy covers 5 major IL paradigms: BC, DAgger, GAIL/AMP, diffusion, hybrid RL-IL
- AMP-based methods show the most natural gaits across surveyed works
- Hybrid approaches consistently outperform pure IL in robustness and task performance
- Diffusion policies enable multi-modal gait generation but suffer from inference latency (10–100x slower than feed-forward)
- Motion retargeting quality directly correlates with downstream policy naturalness
- Identifies >50 papers in the legged locomotion IL space from 2019–2025

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This survey is a key reference for selecting IL techniques to augment PPO training on the Mini Cheetah. The comparison of AMP vs. BC vs. hybrid methods directly informs whether to incorporate demonstration data into the MuJoCo sim training pipeline. The discussion of motion retargeting is relevant if using animal or human reference motions for natural gait priors. The hybrid RL-IL approaches could supplement the existing PPO + domain randomization pipeline by adding demonstration-based reward shaping, potentially accelerating convergence and improving gait quality during curriculum learning. The survey's analysis of real-world transfer challenges maps directly to the sim-to-real transfer pipeline for the 12 DoF Mini Cheetah with PD control at 500 Hz.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The survey directly informs multiple levels of Cassie's 4-level hierarchy. AMP-based adversarial imitation is foundational to the adversarial curriculum component, where natural bipedal gait priors from human motion capture can drive realistic walking styles. The multi-skill integration discussion maps to the Primitives level, where DIAYN/DADS-discovered skills could be refined with IL. Diffusion models are relevant to the Planner level for multi-modal trajectory generation. The hybrid RL-IL approaches can be combined with PPO and Option-Critic training throughout the hierarchy. Motion retargeting from human to Cassie morphology is essential for leveraging human motion datasets.

## What to Borrow / Implement
- Adopt AMP-style adversarial reward for natural gait generation in both projects
- Implement hybrid RL-IL by adding demonstration replay buffer to PPO training
- Use the survey's taxonomy to select the best IL method for each hierarchy level in Project B
- Consider diffusion-based policies at the Planner level for multi-modal trajectory proposals
- Evaluate DAgger-based fine-tuning for post-deployment policy refinement

## Limitations & Open Questions
- Survey breadth may sacrifice depth on any single method
- Diffusion policy inference latency may be prohibitive for 500 Hz control loops
- Motion retargeting quality metrics are inconsistent across surveyed works
- Limited coverage of proprioceptive-only IL (most surveyed works use vision)
- Open question: How to best combine multiple IL paradigms in a single hierarchical system?
