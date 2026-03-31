---
## 📂 FOLDER: research/curriculum_learning/

### 📄 FILE: research/curriculum_learning/dc_gait_decoupled_gait_generation_adaptation.md

**Title:** Dc-Gait: Efficient Locomotion Learning by Decoupled Gait Generation and Adaptation
**Authors:** Chen Feng, et al.
**Year:** 2025
**Venue:** Proceedings of the Institution of Mechanical Engineers
**arXiv / DOI:** 10.1177/09544062251386659

**Abstract Summary (2–3 sentences):**
Dc-Gait proposes a decoupled two-stage pipeline for legged robot locomotion: Stage 1 generates reference gaits using RL under ideal (flat terrain, no disturbances) conditions, and Stage 2 separately trains adaptation policies to imitate and robustify these gaits under environmental disturbances and challenging terrains. This modular separation improves training efficiency by 2x over end-to-end RL and produces reusable gait generators that can be paired with different adaptation modules for different deployment scenarios.

**Core Contributions (bullet list, 4–7 items):**
- Proposes a decoupled two-stage pipeline separating gait generation from gait adaptation
- Achieves 2x faster training convergence compared to end-to-end RL
- Produces modular, reusable gait generators that transfer across deployment scenarios
- Demonstrates better terrain robustness than monolithic end-to-end approaches
- Stage 1 gait generation focuses purely on locomotion quality under ideal conditions
- Stage 2 adaptation learns to handle disturbances while tracking Stage 1 reference gaits
- Validates on quadruped robots across flat, rough, and sloped terrain conditions

**Methodology Deep-Dive (3–5 paragraphs):**
The first stage of Dc-Gait trains a gait generation policy using PPO under idealized simulation conditions: flat terrain, no external disturbances, nominal robot dynamics parameters, and perfect state estimation. The reward function in this stage focuses exclusively on locomotion quality metrics: smooth periodic joint trajectories, symmetric gait patterns, appropriate foot contact timing, energy efficiency, and velocity tracking. By removing all environmental complexity, the gait generator can focus entirely on producing high-quality reference motions. The trained Stage 1 policy outputs reference joint position trajectories as a function of gait phase and commanded velocity.

The second stage trains an adaptation policy that takes the Stage 1 reference trajectories as input along with the robot's current proprioceptive state, and outputs corrective joint position adjustments. The adaptation policy is formulated as a residual controller: the final joint command is u = u_ref + Δu, where u_ref comes from the Stage 1 gait generator and Δu comes from the Stage 2 adaptation network. This residual formulation is key — it means the adaptation policy only needs to learn the correction required to handle environmental disturbances, not the entire locomotion behavior from scratch. The Stage 1 policy is frozen during Stage 2 training.

Stage 2 training introduces the full spectrum of environmental challenges: rough terrain with random height variations, slopes up to ±15 degrees, varying friction coefficients, external push disturbances, and domain-randomized dynamics parameters (mass ±15%, friction ±30%, motor strength ±20%). The reward function combines reference tracking (penalizing large Δu to stay close to the reference gait) with task performance (velocity tracking, stability) and robustness rewards (surviving perturbations). A curriculum progressively increases the difficulty of terrain and disturbances during Stage 2, starting with mild perturbations and advancing to extreme conditions.

The decoupled architecture provides several practical advantages beyond training speed. The Stage 1 gait generator is a reusable module — once trained, it can be paired with different Stage 2 adaptation policies for different deployment scenarios (indoor vs outdoor, flat vs rough terrain) without retraining. This modularity also simplifies debugging and iterative improvement: if locomotion quality is poor, the gait generator is refined; if robustness is lacking, the adaptation policy is retrained. The separation also enables transfer learning — a gait generator trained for one robot morphology can sometimes be adapted to similar robots with only Stage 2 retraining.

Evaluation compares Dc-Gait against end-to-end PPO trained from scratch on the same environmental conditions. Dc-Gait achieves equivalent final performance with approximately half the total training samples (combining Stage 1 and Stage 2). On challenging terrains (steep slopes, very rough ground), Dc-Gait shows 10–20% better survival rates, attributed to the clean reference gaits providing a stable behavioral foundation that the adaptation policy can deviate from when needed but return to when conditions improve.

**Key Results & Numbers:**
- 2x faster total training time compared to end-to-end RL
- 10–20% better survival rates on challenging terrains
- Modular and reusable gait generation across deployment scenarios
- Residual adaptation corrections are typically <15% of the reference joint command magnitude
- Validated on quadruped robots across flat, rough (±5 cm), and sloped (±15°) terrains
- Stage 1 training converges in ~1/3 the time of full end-to-end training
- Stage 2 adaptation training converges in ~1/2 the time of full end-to-end training

**Relevance to Project A (Mini Cheetah):** HIGH — The decoupled approach is directly applicable to the Mini Cheetah training pipeline. Stage 1 can generate clean reference gaits in ideal MuJoCo conditions, and Stage 2 can train robust adaptation for outdoor deployment. The 2x training speedup is valuable for iterative development.
**Relevance to Project B (Cassie HRL):** MEDIUM — The gait generation concept is relevant to the primitives level, and the residual adaptation approach could inform how the controller level corrects reference trajectories. However, the Cassie hierarchy already provides more sophisticated separation of concerns than the two-stage pipeline.

**What to Borrow / Implement:**
- Adopt the two-stage pipeline for Mini Cheetah: train gaits in ideal conditions, then adapt for robustness
- Use the residual controller formulation (u = u_ref + Δu) for the Mini Cheetah adaptation stage
- Apply the gait quality-focused reward in Stage 1 for cleaner reference motions
- Implement the progressive difficulty curriculum in Stage 2 for domain randomization
- Leverage the modularity for iterative development — refine gait quality and robustness independently
- Consider the reusable gait generator concept for Cassie's primitives level

**Limitations & Open Questions:**
- Two-stage training requires careful interface design between stages
- Residual corrections are limited — very large disturbances may exceed the adaptation capacity
- Stage 1 gaits under ideal conditions may not be the best references for extreme terrain
- No automatic mechanism to determine when Stage 1 training is sufficient
- Limited to periodic gaits in Stage 1; aperiodic behaviors require different treatment
- Transfer between robot morphologies is empirically demonstrated but lacks theoretical justification
---
