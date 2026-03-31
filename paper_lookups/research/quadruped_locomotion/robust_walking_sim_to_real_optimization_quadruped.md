---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/robust_walking_sim_to_real_optimization_quadruped.md

**Title:** Robust Walking and Sim-to-Real Optimization for Quadruped Robots via Reinforcement Learning
**Authors:** Zhongwen Zhang, Mingdong Zhang, Fei Meng, Xueshan Gao
**Year:** 2025
**Venue:** Journal of Bionic Engineering
**arXiv / DOI:** DOI: 10.1007/s42235-024-00618-z

**Abstract Summary (2–3 sentences):**
This paper proposes a two-phase learning approach for robust quadruped locomotion: first pretraining a general locomotion policy, then specializing it with terrain-specific information for stair climbing and challenging terrain traversal. The method demonstrates improved adaptability through systematic sim-to-real optimization, with successful deployment on real hardware across diverse challenging scenarios.

**Core Contributions (bullet list, 4–7 items):**
- Two-phase training: general policy pretraining followed by terrain-specific specialization
- Systematic sim-to-real optimization pipeline for closing the reality gap
- Robust stair climbing policy achieving reliable ascent and descent on standard stairs
- Terrain-specific reward shaping for challenging locomotion scenarios
- Real-robot deployment with demonstrated generalization to unseen environments
- Comprehensive analysis of sim-to-real gap factors and mitigation strategies
- Comparison of domain randomization strategies for different terrain types

**Methodology Deep-Dive (3–5 paragraphs):**
The approach addresses the observation that training a single policy for all terrain types often leads to compromise performance — the policy becomes mediocre at everything rather than excellent at specific challenges. Instead, the authors propose a two-phase strategy: Phase 1 pretrains a general locomotion policy using PPO with standard flat-terrain rewards and broad domain randomization. This creates a policy with robust basic locomotion capabilities.

Phase 2 specializes the pretrained policy for specific terrain challenges (primarily stair climbing and rough terrain navigation). Starting from the Phase 1 checkpoint, the policy is fine-tuned with terrain-specific rewards that encourage appropriate foot clearance, body pitch adjustment, and contact timing for stair traversal. The terrain-specific reward includes terms for maintaining adequate foot height during swing, penalizing toe-stub contacts, and rewarding successful stair completion.

The sim-to-real optimization component systematically identifies and addresses sources of the reality gap. The authors conduct ablation studies on key factors: motor response delay, ground contact modeling, sensor noise, and mass distribution uncertainty. Each factor is quantified in terms of its contribution to the sim-to-real gap, and appropriate randomization ranges are derived from real hardware measurements.

Domain randomization is applied hierarchically: broad randomization during Phase 1 for general robustness, and targeted randomization during Phase 2 focused on terrain-specific parameters (stair dimensions, friction, edge geometry). This hierarchical approach achieves better transfer than uniform randomization across all parameters.

**Key Results & Numbers:**
- Stair climbing success rate: 92% on standard indoor stairs (17 cm rise) in real-world tests
- General locomotion speed: up to 1.5 m/s on flat terrain with stable gait
- Phase 2 specialization improved stair success rate by 40% over Phase 1 alone
- Sim-to-real optimization reduced performance gap from ~35% to ~8%
- Deployed on Unitree quadruped robot with onboard computation

**Relevance to Project A (Mini Cheetah):** HIGH — The two-phase training strategy and systematic sim-to-real optimization are directly applicable to the Mini Cheetah MuJoCo-to-real pipeline.
**Relevance to Project B (Cassie HRL):** MEDIUM — The terrain specialization concept aligns with the hierarchical controller's terrain-adaptive components, though the quadruped-specific details differ.

**What to Borrow / Implement:**
- Adopt the two-phase (pretrain general + specialize) training strategy for terrain-specific policies
- Use the systematic sim-to-real gap analysis methodology to identify and mitigate key transfer factors
- Implement hierarchical domain randomization (broad → terrain-specific) for improved transfer

**Limitations & Open Questions:**
- Each terrain type requires a separate Phase 2 specialization, limiting flexibility
- No mechanism for automatic terrain detection and policy switching at deployment
- The sim-to-real optimization methodology is hardware-specific and may not generalize across platforms
---
