# Learning Latent Dynamic Robust Representations for World Models (Hybrid RSSM)

**Authors:** Various
**Year:** 2024 | **Venue:** ICML 2024
**Links:** https://arxiv.org/abs/2405.06263

---

## Abstract Summary
Hybrid RSSM (HRSSM) uses spatio-temporal masking, bisimulation principles, and latent reconstruction to extract task-relevant information and filter out exogenous noise. It strengthens latent state representations for visually cluttered robotic control tasks. The approach addresses a key weakness of standard RSSM — susceptibility to irrelevant visual distractors that pollute the latent space.

## Core Contributions
- Introduced spatio-temporal masking in latent space to filter out task-irrelevant information and focus on controllable dynamics
- Applied bisimulation principles to the RSSM framework, encouraging states with similar dynamics and rewards to have similar latent representations
- Developed latent reconstruction objectives that improve robustness to visual distractors without requiring clean/distractor-free data
- Demonstrated significant improvements in policy learning under visual noise compared to vanilla RSSM and DreamerV3
- Provided theoretical grounding for why standard RSSM representations degrade under exogenous noise
- Showed that robust latent representations transfer better across visual conditions, improving generalization

## Methodology Deep-Dive
Standard RSSM-based world models learn latent representations by reconstructing observations, which means they must encode everything in the observation — including task-irrelevant visual noise like moving backgrounds, lighting changes, or occluding objects. This leads to latent states that waste capacity on exogenous factors, degrading policy performance. HRSSM addresses this with three complementary techniques.

Spatio-temporal masking applies learned attention masks in both spatial (what parts of the observation matter) and temporal (which timesteps are informative) dimensions. The spatial mask highlights task-relevant regions of the observation (e.g., the robot and nearby terrain, not distant backgrounds), while the temporal mask identifies informative transitions versus redundant frames. These masks are learned end-to-end via gradient descent, with sparsity regularization encouraging the model to focus on minimal sufficient information.

Bisimulation-based regularization adds a loss term that encourages latent states to be close (in representation space) when they lead to similar dynamics and rewards, regardless of their visual appearance. This is inspired by bisimulation metrics from control theory: two states are bisimilar if they produce the same reward and transition to bisimilar successor states. In practice, this is implemented as a contrastive loss that pulls together representations of dynamically equivalent states and pushes apart representations of dynamically different states.

Latent reconstruction replaces or augments pixel reconstruction with reconstruction in the latent space itself. Instead of decoding back to high-dimensional pixel observations, the model reconstructs masked latent features from unmasked ones, or predicts future latent states from current ones. This self-supervised objective focuses the representation on dynamically relevant features without requiring the model to perfectly reconstruct irrelevant visual details.

The combination of these three techniques creates a robust latent space that captures only the controllable, task-relevant aspects of the environment. The policy trained on these representations is less sensitive to visual perturbations and generalizes better to novel visual conditions. The additional computational cost is modest — mainly the masking operations and bisimulation contrastive loss.

## Key Results & Numbers
- Improved robustness to visual distractors compared to vanilla RSSM and DreamerV3 on DMControl with distractors
- Better policy learning in noisy environments, with 20-40% higher returns under heavy visual noise
- Outperforms vanilla RSSM on tasks with moving backgrounds, lighting changes, and occluding objects
- Latent representations show cleaner clustering of dynamically similar states
- Transfer performance: policies trained with HRSSM generalize better to unseen visual conditions
- Minimal computational overhead (~10-15%) compared to standard RSSM training

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Robustness to visual noise is useful for real-world deployment with onboard cameras. If Mini Cheetah uses visual observations for terrain-aware locomotion, HRSSM would help filter out irrelevant visual clutter (people walking by, lighting changes, camera shake). The bisimulation principle is also applicable to proprioceptive observations: encouraging similar latent representations for states with similar dynamics regardless of irrelevant sensor noise (e.g., slight IMU drift). The spatio-temporal masking could help identify which proprioceptive features are most relevant at each phase of the gait cycle.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Noise-robust RSSM variants directly improve the planner's world model, especially when Cassie's terrain observations are noisy. The CPTE (Contrastive Terrain Encoder) generates terrain embeddings that may contain exogenous noise from visual sensors, and HRSSM's masking and bisimulation techniques would clean these representations before they enter the planner. The bisimulation principle is especially relevant for the hierarchical architecture: at the planner level, states that lead to the same optimal locomotion primitive should have similar representations, regardless of visual differences. Spatio-temporal masking could help the planner focus on terrain features relevant to the current locomotion mode while ignoring distant or irrelevant observations.

## What to Borrow / Implement
- Implement bisimulation-based contrastive regularization in the RSSM world model to encourage dynamically meaningful representations
- Apply spatial masking to terrain observations to focus the planner on immediately relevant terrain features
- Use temporal masking to identify and weight the most informative timesteps during world model training
- Replace pixel reconstruction with latent reconstruction objectives to reduce computational cost and improve representation quality
- Test HRSSM's robustness improvements by training with synthetic visual distractors and evaluating transfer
- Combine bisimulation principles with the CPTE to create terrain representations that are invariant to irrelevant visual features

## Limitations & Open Questions
- Bisimulation contrastive loss requires careful tuning of temperature and margin parameters
- Spatio-temporal masking may inadvertently mask relevant information if regularization is too aggressive
- Not directly validated on locomotion tasks — primarily tested on manipulation and DMControl
- Interaction with domain randomization is unclear — does HRSSM complement or conflict with existing randomization strategies?
- May be less relevant for proprioceptive-only policies where there is minimal exogenous noise
- The learned masks are task-specific and may not transfer across different locomotion behaviors
- Computational overhead of masking and bisimulation may accumulate in the already complex hierarchical architecture
