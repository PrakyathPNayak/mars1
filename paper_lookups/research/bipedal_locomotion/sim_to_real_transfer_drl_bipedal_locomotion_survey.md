---
## 📂 FOLDER: research/bipedal_locomotion/

### 📄 FILE: research/bipedal_locomotion/sim_to_real_transfer_drl_bipedal_locomotion_survey.md

**Title:** Sim-to-Real Transfer in Deep Reinforcement Learning for Bipedal Locomotion: A Survey
**Authors:** Various (comprehensive arXiv survey)
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2511.06465

**Abstract Summary (2-3 sentences):**
This comprehensive survey focuses specifically on sim-to-real transfer methods for bipedal locomotion reinforcement learning, cataloging over 80 papers and categorizing approaches into domain randomization, system identification, domain adaptation, and hybrid methods. The survey reviews transfer success rates across multiple platforms including Cassie, Digit, Atlas, and various humanoids, identifying domain randomization combined with privileged learning as the most successful combination. Key remaining gaps are identified in contact dynamics modeling, actuator simulation fidelity, and standardized evaluation protocols for sim-to-real transfer quality.

**Core Contributions (bullet list, 4-7 items):**
- Comprehensive catalog of 80+ papers on sim-to-real transfer for bipedal locomotion RL
- Taxonomy of transfer methods: domain randomization, system identification, domain adaptation, hybrid approaches
- Cross-platform comparison of transfer success rates across Cassie, Digit, Atlas, HRP, NAO, and custom robots
- Identification of domain randomization + privileged learning as the most successful transfer combination
- Detailed analysis of remaining gaps in contact dynamics and actuator modeling for sim-to-real transfer
- Proposed evaluation framework for standardizing sim-to-real transfer quality assessment
- Timeline analysis showing evolution of transfer techniques from 2018 to 2025

**Methodology Deep-Dive (3-5 paragraphs):**
The survey structures its analysis around four primary categories of sim-to-real transfer methods. Domain randomization (DR) approaches train policies across a distribution of simulator parameters, with the goal of producing policies that are robust to any parameter setting within the distribution, including the real world. The survey catalogs the specific parameters randomized in bipedal locomotion research: physics parameters (mass, inertia, friction, restitution, damping), actuator parameters (motor strength, backlash, delay, current-torque mapping), sensor parameters (noise, bias, delay, dropout), and environmental parameters (terrain geometry, wind forces, external disturbances). A key finding is that the choice of randomization distribution (uniform, Gaussian, log-uniform) and its bounds significantly affects transfer quality, with overly broad distributions leading to conservative policies and overly narrow distributions failing to cover the real system.

System identification (SysID) approaches use real-world data to calibrate simulator parameters, aiming to minimize the sim-to-real gap at its source. The survey identifies several SysID paradigms used in bipedal locomotion: offline SysID (collecting real-world data once and optimizing simulator parameters to match), online SysID (continuously updating simulator parameters as new real-world data becomes available), and learned SysID (training neural networks to predict simulator parameters from observed robot behavior). The survey notes that pure SysID has become less common in recent bipedal locomotion work, primarily because contact dynamics and actuator nonlinearities are difficult to identify accurately. However, SysID remains valuable as a preprocessing step that reduces the required range of domain randomization.

Domain adaptation approaches aim to align the distributions of simulated and real-world experience without explicitly modifying simulator parameters. The survey covers three primary adaptation paradigms: privileged learning (training a teacher with access to ground truth simulator parameters, then distilling to a student that must infer these from observation history), adversarial domain adaptation (using discriminators to align simulated and real observation distributions), and meta-learning (training policies that can rapidly adapt to new dynamics with few real-world samples). Privileged learning has emerged as the dominant paradigm, used in the majority of successful real-world bipedal locomotion deployments since 2022. The survey provides a detailed comparison of privileged learning implementations, noting that the choice of privileged information (terrain height maps, contact states, dynamics parameters, or combinations) significantly affects the quality of the distilled student policy.

Hybrid methods combining elements of multiple categories represent the current state of the art. The most common and successful hybrid is DR + privileged learning, where the teacher is trained with domain randomization to be robust across parameter variations, and the student learns to infer the implicit dynamics parameters from observation history to achieve similarly adaptive behavior. The survey identifies several emerging hybrid combinations: DR + online SysID (narrowing the randomization range with real-world data), privileged learning + meta-learning (distilling a teacher that can also rapidly adapt), and curriculum-based DR (progressively expanding randomization ranges as training progresses). The survey also discusses the role of simulation fidelity, noting that higher-fidelity simulators (e.g., MuJoCo with improved contact models) can reduce but not eliminate the need for transfer techniques.

The cross-platform analysis reveals significant variation in transfer success rates. Cassie-based systems show the highest overall success rate (approximately 85% of published sim-to-real results report successful walking), attributed to the extensive community experience with Cassie and the availability of calibrated simulation models. Digit-based systems show approximately 75% success rate, with the primary challenge being the additional complexity of the upper body and arms. Full-sized humanoids (HRP, Atlas-like) show lower success rates (approximately 60%), primarily due to the higher consequences of failure and the difficulty of modeling large-scale actuator dynamics. The survey identifies contact dynamics modeling as the single largest remaining gap across all platforms, with ground contact, foot-terrain interaction, and impact dynamics being poorly captured by current simulators.

**Key Results & Numbers:**
- Catalogs 80+ papers on bipedal sim-to-real transfer
- Domain randomization + privileged learning identified as most successful combination
- Cassie sim-to-real success rate approximately 85% in published results
- Digit sim-to-real success rate approximately 75%
- Full-sized humanoid sim-to-real success rate approximately 60%
- Contact dynamics modeling identified as the largest remaining sim-to-real gap
- Privileged learning used in the majority of successful deployments since 2022
- Evolution from pure DR (2018-2020) to DR + privileged learning (2021-2025) clearly traced

**Relevance to Project A (Mini Cheetah):** MEDIUM — Sim-to-real transfer survey techniques are broadly applicable across legged platforms. The domain randomization parameter ranges, privileged learning frameworks, and evaluation protocols are transferable to quadruped sim-to-real for the Mini Cheetah, though the specific challenges of bipedal balance do not directly apply.

**Relevance to Project B (Cassie HRL):** HIGH — Essential reference for designing the Cassie sim-to-real pipeline. The survey's identification of best practices (DR + privileged learning), Cassie-specific transfer success rates, and remaining gaps (contact dynamics, actuator modeling) directly inform the CPTE (Contact-Phase Terrain Embedding) and Neural ODE components of the hierarchical controller. The evaluation framework provides a template for benchmarking the HRL system's transfer quality.

**What to Borrow / Implement:**
- DR + privileged learning as the baseline transfer strategy for the Cassie hierarchical controller
- Cassie-specific domain randomization parameter ranges and distributions from the literature catalog
- Contact dynamics gap analysis to inform the CPTE design and contact modeling improvements
- Standardized evaluation protocol for sim-to-real transfer quality assessment
- Privileged information selection guidelines (what to give the teacher vs. what the student must infer)
- Timeline of transfer technique evolution for contextualizing the proposed approach in related work

**Limitations & Open Questions:**
- Survey paper that does not introduce new methods; relies on reported results from other papers
- Rapidly evolving field where new techniques appear monthly, potentially making some analysis outdated quickly
- Published results may have selection bias toward successful transfers, inflating reported success rates
- Cross-platform comparison is inherently limited by different evaluation protocols across labs
- The survey does not deeply analyze computational costs or real-world data requirements for different transfer methods
- Contact dynamics gap identified but no concrete solutions proposed
---
