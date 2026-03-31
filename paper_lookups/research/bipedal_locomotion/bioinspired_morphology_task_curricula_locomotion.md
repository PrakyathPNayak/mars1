---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/bioinspired_morphology_task_curricula_locomotion.md

**Title:** Bioinspired Morphology and Task Curricula for Learning Locomotion in Musculoskeletal Bipeds
**Authors:** Filip Toric, Dimitar Stanev, Auke Ijspeert
**Year:** 2025
**Venue:** Communications Engineering (Nature)
**arXiv / DOI:** 10.1038/s44172-025-00443-0

**Abstract Summary (2-3 sentences):**
This paper investigates dual curriculum strategies for training bipedal locomotion in musculoskeletal models, combining a morphological growth curriculum (simulating age-related developmental changes in body proportions, muscle strength, and mass distribution) with a task difficulty curriculum (progressively increasing terrain complexity and speed demands). The bioinspired developmental approach demonstrates 2-3x faster training convergence compared to training directly on the adult morphology, while also improving generalization to novel terrains and speeds not seen during training. The emergent gaits exhibit biologically plausible characteristics including natural toe-off, heel-strike patterns, and energy-efficient arm swing.

**Core Contributions (bullet list, 4-7 items):**
- Dual curriculum framework combining morphological growth and task difficulty progression for bipedal locomotion
- Demonstration of 2-3x faster RL training convergence through developmental curriculum compared to direct adult training
- Improved terrain and speed generalization from curriculum-trained policies
- Emergence of biologically plausible gait characteristics without explicit biomechanical reward terms
- Comprehensive analysis of how morphological parameters (limb proportions, muscle strength, mass distribution) affect learning dynamics
- Comparison of curriculum orderings and schedules revealing optimal developmental progression strategies
- Open-source musculoskeletal bipedal model and curriculum implementation

**Methodology Deep-Dive (3-5 paragraphs):**
The morphological growth curriculum models the developmental trajectory of a bipedal organism from infant-like proportions to adult morphology. The musculoskeletal model features 20+ muscle-tendon actuators per leg with biologically realistic force-length-velocity relationships, a segmented torso with flexible spine, and articulated feet with toe joints. During the morphological curriculum, body parameters change gradually over training: limb segment lengths grow according to allometric scaling laws derived from human developmental data, muscle maximum isometric forces increase following strength development curves, and the mass distribution shifts as the center of mass moves from relatively high (infant proportions with large head-to-body ratio) to lower (adult proportions). The key insight is that infant-like morphologies are inherently easier to balance and control due to their lower center of mass, shorter limb lengths, and proportionally stronger muscles relative to body weight.

The task curriculum operates orthogonally to the morphological curriculum, progressively increasing the difficulty of the locomotion task. The progression moves through several stages: (1) standing balance on flat ground, (2) walking at a single slow speed on flat ground, (3) walking at variable speeds including start-stop transitions, (4) walking on mildly uneven terrain with small bumps and slopes, (5) walking on challenging terrain with larger obstacles, slopes up to 15 degrees, and gaps, and (6) running at higher speeds on flat and uneven terrain. The task curriculum is advancement-gated: the policy must achieve a minimum performance threshold at the current difficulty level before the curriculum advances. The authors explore both synchronized dual curricula (morphology and task advance together) and asynchronous schedules (morphology advances faster, then task difficulty catches up).

Training uses PPO with the musculoskeletal simulation running in a custom physics engine optimized for muscle-tendon dynamics. The observation space includes joint angles and angular velocities for all body segments, muscle activation levels, ground contact forces at discrete foot sensor points, pelvis IMU-equivalent readings, and the current morphological parameters (allowing the policy to be conditioned on the current developmental stage). The action space specifies desired muscle activation levels (0 to 1) for each muscle-tendon unit, which are converted to forces through the Hill-type muscle model. The reward function focuses on task performance (velocity tracking, heading maintenance), stability (upright posture, smooth center-of-mass trajectory), and efficiency (metabolic cost of transport computed from muscle activations and mechanical work). Notably, no explicit gait style rewards are used; the biologically plausible gaits emerge from the interaction between the musculoskeletal dynamics and the efficiency-focused reward.

The dual curriculum produces consistent training benefits across multiple random seeds and hyperparameter settings. The morphological curriculum alone (without task curriculum) provides approximately 1.5x speedup, the task curriculum alone provides approximately 1.8x speedup, and the combination provides 2-3x speedup, demonstrating synergistic effects. The authors analyze the learning dynamics and find that the morphological curriculum helps the policy discover stable periodic gaits early in training (when the morphology is easy to control), and these gait patterns serve as a foundation that is progressively refined as the body grows to adult proportions. Without the morphological curriculum, the policy often gets stuck in local optima corresponding to static standing or shuffling gaits from which it struggles to escape.

Generalization experiments evaluate the trained policies on terrains and speeds not encountered during training. Curriculum-trained policies successfully walk on terrain types 25-40% more challenging than the maximum training difficulty, while directly trained policies fail at or slightly above the training difficulty level. Similarly, curriculum-trained policies generalize to speeds 20-30% beyond the training range, suggesting that the curriculum produces more robust internal representations. The authors attribute this to the curriculum forcing the policy to continuously adapt its control strategy as the morphology and task change, preventing overfitting to any specific configuration.

**Key Results & Numbers:**
- 2-3x faster training convergence with dual curriculum compared to direct adult training
- Morphological curriculum alone: 1.5x speedup; task curriculum alone: 1.8x speedup
- Improved terrain generalization: policies handle terrains 25-40% more challenging than training maximum
- Speed generalization: 20-30% beyond training speed range
- Biologically plausible gait patterns emerge without explicit biomechanical rewards
- Natural heel-strike and toe-off patterns observed in emergent gaits
- Metabolic cost of transport within 15% of biological values for equivalent body size
- 20+ muscle-tendon actuators per leg in the musculoskeletal model

**Relevance to Project A (Mini Cheetah):** MEDIUM — The curriculum learning concept is broadly applicable to quadruped RL training. Progressive difficulty scaling for terrain and speed could accelerate Mini Cheetah policy training. However, the musculoskeletal model is specific to biological systems and does not directly apply to rigid-body robots.

**Relevance to Project B (Cassie HRL):** HIGH — The dual curriculum approach is directly relevant to the adversarial curriculum component of the hierarchical controller. The task curriculum design provides a template for progressively training different hierarchy levels. The morphological adaptation concept, while not directly applicable (Cassie's morphology is fixed), could inspire domain randomization curricula where physics parameters are gradually shifted during training to improve robustness.

**What to Borrow / Implement:**
- Dual curriculum design philosophy: combine environmental difficulty progression with another orthogonal curriculum dimension
- Advancement-gated curriculum scheduling to prevent premature difficulty increases
- The insight that starting with easier configurations and progressively increasing difficulty produces more robust policies than direct training on the target difficulty
- Curriculum synchronization analysis: optimal scheduling between multiple curriculum dimensions
- Generalization evaluation protocol: testing beyond the training distribution to assess robustness

**Limitations & Open Questions:**
- Musculoskeletal model differs fundamentally from rigid-body robots like Cassie; muscle dynamics and biological scaling laws do not directly transfer
- Simulation-only results with no real-robot validation
- Computational cost of musculoskeletal simulation is significantly higher than rigid-body simulation
- The morphological curriculum is specific to developmental biology; the analogy to robotics requires creative adaptation
- Fixed curriculum schedules may not be optimal; adaptive curriculum methods could improve results
- Limited exploration of how curriculum parameters interact with RL hyperparameters
---
