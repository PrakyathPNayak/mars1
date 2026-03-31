# Diverse Skill Discovery for Quadruped Robots via Unsupervised Learning

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** arXiv (2025)

---

## Abstract Summary
This paper applies DIAYN-based unsupervised skill discovery specifically to quadruped robots, systematically analyzing the challenges and solutions unique to four-legged locomotion. While DIAYN has been widely demonstrated on simplified MuJoCo environments (Half-Cheetah, Ant), transferring the framework to realistic quadruped models with high-dimensional joint spaces, contact dynamics, and stability constraints reveals significant practical challenges. The paper identifies and addresses issues including skill degeneracy, contact-mode collapse, and gait instability.

The authors propose targeted modifications to the DIAYN framework for quadrupeds: contact-aware discriminator conditioning, gait regularization terms, and a curriculum over skill complexity. The discovered skill repertoire includes trotting at various speeds, bounding, pronking, lateral shuffling, turning gaits, and hybrid locomotion patterns. Each skill is evaluated on diversity metrics (trajectory mutual information, state-space coverage) and quality metrics (energy efficiency, stability margin, gait periodicity).

The work bridges the gap between theoretical DIAYN and practical quadruped deployment, providing detailed ablation studies showing which modifications are essential for high-quality skill discovery on realistic quadruped models. The resulting skill libraries are demonstrated both in simulation and on real quadruped hardware, with analysis of which discovered skills transfer successfully and which require fine-tuning.

## Core Contributions
- Systematic evaluation of DIAYN on realistic quadruped models, identifying failure modes specific to legged locomotion (contact-mode collapse, gait asymmetry, energy explosion)
- Contact-aware discriminator that conditions on foot contact patterns alongside proprioceptive state, encouraging skills with distinct contact sequences (trot vs. bound vs. pronk)
- Gait regularization curriculum that initially biases toward periodic, stable gaits before relaxing constraints to allow more exotic skills
- Comprehensive diversity and quality metrics for evaluating discovered quadruped locomotion skills
- Demonstration of discovered skill deployment on real quadruped hardware with analysis of sim-to-real transfer success rates per skill type
- Ablation studies showing the individual contribution of each proposed modification to skill quality and diversity
- Open-source implementation and skill library for community benchmarking

## Methodology Deep-Dive
The base framework follows DIAYN with a discrete skill space z ∈ {1, ..., K} and SAC-based policy optimization. The discriminator q_φ(z|s) is modified to receive an augmented state representation that includes standard proprioception (joint angles, angular velocities, base orientation, base velocity) concatenated with binary foot contact indicators and a contact phase variable encoding the current gait cycle position. This contact-aware input prevents the failure mode where all skills converge to static standing (the easiest distinguishable behavior) by explicitly making the discriminator attend to locomotion-specific features.

The gait regularization curriculum operates in three phases. Phase 1 (0–1M steps): strong periodicity and symmetry constraints enforce stable, regular gaits. The policy receives auxiliary rewards for maintaining periodic joint trajectories (measured by autocorrelation peaks) and bilateral symmetry (left-right leg phase relationships). Phase 2 (1M–3M steps): constraints are gradually relaxed, allowing skills to deviate from strict periodicity while maintaining stability. The stability constraint is a center-of-mass projection criterion ensuring the CoM projection stays within the support polygon. Phase 3 (3M+ steps): only minimal safety constraints remain (no self-collision, bounded joint velocities), allowing the emergence of agile and unconventional skills.

The skill diversity is promoted through an additional coverage bonus: r_coverage = −log(ρ(s)), where ρ(s) is a kernel density estimate of the state visitation across all skills. This encourages skills to explore under-visited state regions, complementing the discriminator-based MI reward. The density estimate is updated periodically using a replay buffer of recent trajectories from all skills.

Training is conducted in Isaac Gym with 2048 parallel environments, each running a different skill index. The policy architecture is an LSTM-based network with 2 layers of 256 hidden units, chosen to capture temporal gait patterns. The discriminator uses a temporal convolutional network (TCN) operating on windows of 50 timesteps to classify skills based on trajectory segments rather than individual states, improving discrimination of dynamic behaviors.

Evaluation metrics include: (1) Trajectory MI — estimated mutual information between skills and trajectory distributions; (2) State-Space Coverage — fraction of a discretized state space visited across all skills; (3) Gait Quality Index — composite score of periodicity, symmetry, energy efficiency, and stability; (4) Skill Uniqueness — minimum pairwise trajectory distance across all skill pairs.

## Key Results & Numbers
- Discovered 16 distinct locomotion skills on a simulated Unitree Go1 quadruped with K=20 skill slots (4 degenerate)
- Identified skill types: 3 trotting variants (slow, medium, fast), 2 bounding gaits, 1 pronking gait, 3 turning skills (left, right, pivot), 2 lateral shuffle skills, 3 hybrid/transitional gaits, 2 backward locomotion skills
- Contact-aware discriminator increased skill diversity by 45% compared to standard DIAYN (measured by trajectory MI)
- Gait regularization curriculum reduced unstable skill ratio from 35% to 8%
- 12 of 16 skills transferred to real Unitree Go1 hardware without fine-tuning; remaining 4 required 10 minutes of online adaptation
- Energy efficiency: discovered trotting skills within 20% of hand-tuned trotting controller
- State-space coverage: 3.2× improvement over vanilla DIAYN, 1.4× improvement over DADS
- Training time: 6 hours on NVIDIA RTX 4090 with Isaac Gym

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to Mini Cheetah gait repertoire discovery. The contact-aware discriminator and gait regularization curriculum address the exact challenges expected when applying DIAYN to Mini Cheetah's 12-DoF system. The discovered skill types (trotting, bounding, pronking, turning) map precisely to the gait library needed for agile Mini Cheetah locomotion. The LSTM-based policy architecture and TCN-based discriminator can be adopted directly. The ablation studies provide a clear roadmap for which modifications are essential vs. optional, saving significant development time. The sim-to-real transfer analysis (12/16 skills transferring zero-shot) sets realistic expectations for Mini Cheetah deployment.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
While focused on quadrupeds, several insights transfer to Cassie's bipedal setting. The contact-aware discriminator concept applies directly — conditioning on Cassie's foot contact states to encourage skills with different stance/swing patterns (walking, running, hopping). The gait regularization curriculum is even more important for bipedal stability where the margin of stability is narrower. The trajectory-based (TCN) discriminator over time windows rather than single states is valuable for distinguishing dynamic bipedal gaits that differ in temporal patterns. However, the quadruped-specific contact patterns and gait types don't directly map to bipedal locomotion.

## What to Borrow / Implement
- Implement contact-aware discriminator for both Mini Cheetah (4 contact points) and Cassie (2 contact points) by concatenating binary foot contacts and gait phase to discriminator input
- Adopt the three-phase gait regularization curriculum: strict periodicity → relaxed constraints → minimal safety only
- Use TCN-based discriminator operating on 50-step trajectory windows instead of single-state classification for better dynamic gait discrimination
- Apply the coverage bonus r_coverage to complement MI reward and prevent state-space collapse in high-dimensional robot settings
- Benchmark discovered skills using the proposed metrics: trajectory MI, state coverage, gait quality index, and skill uniqueness

## Limitations & Open Questions
- K=20 skill slots are somewhat arbitrary; no principled method for selecting the optimal number of skills for a given robot morphology
- The gait regularization curriculum schedule (phase transition points at 1M and 3M steps) may need re-tuning for different robots and simulators
- Degenerate skills (4 out of 20) suggest the method doesn't fully solve the skill collapse problem, only mitigates it
- Real-robot transfer of agile skills (bounding, pronking) remains challenging, with 4 of 16 skills requiring online fine-tuning
