---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/deep_rl_robotic_bipedal_locomotion_survey.md

**Title:** Deep Reinforcement Learning for Robotic Bipedal Locomotion: A Brief Survey
**Authors:** Lingfan Bao, Joseph Sifakis, Toshiaki Aoki, Takuya Kobayashi
**Year:** 2025
**Venue:** Applied Intelligence (Springer)
**arXiv / DOI:** 10.1007/s10462-025-11451-z

**Abstract Summary (2-3 sentences):**
This comprehensive survey categorizes deep reinforcement learning frameworks for bipedal locomotion published between 2023 and 2025, covering both end-to-end and hierarchical approaches across multiple humanoid and bipedal platforms. The paper reviews sim-to-real transfer techniques, reward design philosophies, and evaluation methodologies, identifying that PPO and SAC are the dominant algorithms in the field. The survey provides a structured taxonomy of approaches and highlights open challenges including sample efficiency, safety guarantees, and generalization across platforms.

**Core Contributions (bullet list, 4-7 items):**
- Comprehensive taxonomy of DRL approaches for bipedal locomotion organized into 6 major categories
- Systematic comparison of 50+ papers across multiple dimensions (algorithm, architecture, platform, transfer method)
- Identification of PPO and SAC as the dominant algorithms for bipedal locomotion RL
- Analysis of sim-to-real transfer techniques categorized by strategy (randomization, adaptation, system ID)
- Review of reward design philosophies from hand-crafted to learned reward functions
- Evaluation methodology comparison across the field, highlighting inconsistencies and proposing standardization
- Identification of key open challenges: sample efficiency, safety, generalization, and multi-skill learning

**Methodology Deep-Dive (3-5 paragraphs):**
The survey organizes the bipedal locomotion DRL literature into six major categories based on the architectural approach: (1) end-to-end flat policies that directly map observations to actions, (2) hierarchical policies with explicit high-level and low-level decomposition, (3) reference-guided policies that track motion capture or optimization-generated trajectories, (4) model-based RL approaches that learn or leverage dynamics models, (5) multi-agent or multi-policy frameworks that combine multiple specialists, and (6) hybrid approaches that integrate classical control elements with learned components. For each category, the survey identifies representative works, analyzes their strengths and weaknesses, and discusses the types of locomotion behaviors they have achieved. The hierarchical category is particularly relevant to modern approaches, with the survey noting a trend toward increasingly deep hierarchies (2-4 levels) that decompose locomotion into planning, primitive selection, and low-level execution.

The sim-to-real transfer analysis is one of the most detailed sections, breaking down transfer strategies into four primary classes. Domain randomization approaches randomize physics parameters during training to produce robust policies; the survey catalogs the specific parameters randomized by different groups and their ranges, noting that ground friction, body mass, and actuator dynamics are universally randomized while joint stiffness and damping receive less attention. System identification approaches use real-world data to calibrate simulator parameters; the survey notes that these are less common in recent work due to the difficulty of identifying complex contact dynamics. Domain adaptation approaches use techniques like privileged learning (teacher-student frameworks) or adversarial training to align simulated and real distributions; this category has grown significantly, with privileged learning becoming the dominant paradigm. Hybrid approaches combine elements of multiple strategies, with the survey identifying domain randomization plus privileged learning as the most successful combination overall.

The reward design analysis reveals a spectrum from fully hand-crafted multi-term rewards (10-20 weighted terms) to learned reward functions using inverse RL or adversarial methods. The survey finds that hand-crafted rewards still dominate for practical deployment, with the most common terms being velocity tracking, orientation stability, action smoothness, energy efficiency, and contact regularity. However, the relative weighting of these terms varies dramatically across papers, and the survey notes that reward tuning remains one of the most time-consuming aspects of bipedal RL development. Learned rewards show promise for producing more natural gaits but introduce training instability and are less common in deployed systems. The survey also discusses the role of reward curricula, where reward terms or their weights change during training to guide the policy through progressive skill acquisition.

The platform analysis covers a wide range of bipedal and humanoid robots including Cassie, Digit, Atlas, HRP series, NAO, and various custom platforms, as well as simulation-only platforms like the MuJoCo humanoid and NVIDIA Isaac humanoid models. The survey notes significant variation in evaluation practices: some works report simulation-only results, others demonstrate real-world deployment, and the metrics used (success rate, velocity tracking error, energy efficiency, robustness to perturbations) are not standardized. This inconsistency makes direct comparison between approaches difficult, and the survey calls for community-agreed evaluation benchmarks similar to those available for manipulation tasks.

**Key Results & Numbers:**
- Catalogs and categorizes 50+ papers on DRL for bipedal locomotion (2023-2025)
- Identifies 6 major categories of DRL approaches for bipedal locomotion
- PPO used in approximately 65% of surveyed papers, SAC in approximately 25%
- Domain randomization combined with privileged learning identified as most successful sim-to-real strategy
- Hierarchical approaches trending toward 2-4 level decompositions
- Hand-crafted rewards with 10-20 terms remain dominant in deployed systems
- Significant gap between simulation-only and real-world deployed results across the field

**Relevance to Project A (Mini Cheetah):** MEDIUM — Provides a broad landscape of DRL techniques, many of which (reward design, sim-to-real transfer, algorithm selection) are applicable to quadruped RL. The taxonomy of approaches helps position the Mini Cheetah work within the broader legged locomotion field.

**Relevance to Project B (Cassie HRL):** HIGH — Essential reference for positioning the Cassie HRL system within the bipedal RL literature. The hierarchical policy category directly encompasses the proposed approach, and the sim-to-real transfer analysis provides critical context for the privileged learning and domain randomization strategies. The survey's identification of open challenges aligns with the motivations for the 4-level hierarchical design.

**What to Borrow / Implement:**
- Taxonomy of approaches for structuring the related work section and positioning the HRL contribution
- Sim-to-real best practices: domain randomization + privileged learning as the recommended combination
- Reward design patterns: common terms and their typical weight ranges for bipedal locomotion
- Evaluation metrics and protocols used across the field for consistent comparison
- Identification of open challenges that the hierarchical controller aims to address

**Limitations & Open Questions:**
- Survey paper that does not introduce any new methods or algorithms
- Coverage period (2023-2025) may not include the very latest 2025 developments
- Comparison across papers is inherently limited by different evaluation protocols and platforms
- The survey does not deeply analyze computational costs or training efficiency of different approaches
- Some categorizations are subjective, and papers may span multiple categories
- Rapid evolution of the field means the survey may become dated within 1-2 years
---
