# MARG: MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping

**Authors:** (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** arXiv (2024)

---

## Abstract Summary
MARG presents a risk-informed deep reinforcement learning framework for legged robot locomotion across gap terrains — one of the most challenging terrain types where a single misstep can result in catastrophic failure. The system integrates onboard elevation mapping with proprioceptive sensing to enable safe gap negotiation. A key innovation is the selective privileged information training methodology, where risk-relevant terrain features (gap width, depth, edge locations) are provided as privileged information during training but must be inferred from the elevation map during deployment.

The framework addresses a critical limitation of prior terrain-adaptive locomotion methods: most approaches treat all terrain types uniformly, without distinguishing between terrains where failure is recoverable (rough ground, moderate slopes) and terrains where failure is catastrophic (gaps, cliffs, thin bridges). MARG introduces a risk assessment module that explicitly estimates the consequence of missteps and adjusts the locomotion strategy accordingly. On low-risk terrain, the policy prioritizes speed and energy efficiency; on high-risk terrain, it shifts to conservative, precise foot placement with reduced velocity.

Real-time terrain maps are constructed from a single LiDAR sensor, with a novel drift compensation mechanism that reduces mapping errors from odometry drift. The system demonstrates successful gap traversal on gaps up to 35cm width on a quadruped robot, with safe fallback behavior when gaps exceed the robot's capabilities.

## Core Contributions
- **Risk-informed RL objective** that incorporates terrain risk assessment into the reward function, balancing locomotion performance with failure avoidance
- **Selective privileged information training** where only risk-relevant terrain features are privileged, reducing the distillation gap compared to full-terrain privileged training
- **Online risk assessment module** that estimates traversal risk from elevation map features, triggering conservative locomotion strategies near dangerous terrain
- **LiDAR-based elevation mapping** with drift compensation enabling real-time terrain reconstruction from minimal exteroception
- **Gap-specific locomotion skills** including approach adjustment, straddling, and leap-across strategies selected based on gap width estimation
- **Safe fallback behavior** that halts forward progress and initiates retreat when gap risk exceeds a learned threshold

## Methodology Deep-Dive
The MARG framework consists of three interconnected modules: the elevation mapping module, the risk assessment module, and the locomotion policy. The elevation mapping module processes point clouds from a single forward-facing LiDAR (Livox MID-360) to construct a local 2.5D heightmap in the robot's body frame. A key challenge is odometry drift, which causes map distortion over time. MARG addresses this with a height consistency check: when the robot revisits previously mapped regions (as it moves forward), inconsistencies between the stored and newly observed heights trigger a drift correction that adjusts the entire map. This reduces mapping RMSE from 4.2cm to 1.8cm over 10-meter traversals.

The risk assessment module takes the local elevation map as input and produces a risk score for each grid cell. The risk is defined based on terrain traversability: cells with sufficient support area are low risk, cells near gap edges are medium risk, and cells over gaps or cliffs are high risk. During training, the risk is computed from ground-truth terrain geometry. During deployment, a lightweight CNN (3 convolutional layers + 2 FC layers) predicts risk from the elevation map. The risk network is trained with supervised learning on simulation data and fine-tuned with real-world elevation maps.

The selective privileged information approach is a refinement of standard asymmetric training. Rather than providing the student with all terrain information, only risk-critical features are privileged: gap width, gap depth, distance to nearest gap edge, and surface friction at foot contact points. Other terrain features (general roughness, mild slopes) are left for the student to infer from the elevation map and proprioception. This selective approach reduces the information gap between teacher and student, leading to better distillation. The teacher is trained with PPO using an augmented reward: the standard locomotion reward (velocity tracking, energy penalty, smoothness) plus a risk-weighted survival bonus that heavily penalizes falling into gaps.

The locomotion policy operates at three levels of gap negotiation: (1) far from gap — normal locomotion with risk-informed speed reduction, (2) near gap edge — precise foot placement with approach angle adjustment to maximize gap clearance, and (3) at gap — select between straddling (for narrow gaps <15cm) or leaping (for gaps 15-35cm). The strategy selection is implicit in the policy, learned through the risk-conditioned reward rather than explicit mode switching.

Training uses a terrain curriculum focused on gaps of increasing difficulty: starting with narrow cracks (5cm), progressing through moderate gaps (15-25cm), and ending with challenging gaps (30-35cm). Environmental randomization includes gap width, gap depth (from shallow ditches to infinite depth), gap edge irregularity, surrounding terrain roughness, and surface friction.

## Key Results & Numbers
- **Gap traversal success rate**: 90% on 25cm gaps, 78% on 30cm gaps, 55% on 35cm gaps (approaching physical limits of the robot)
- **Risk assessment accuracy**: 94% precision and 91% recall for identifying high-risk terrain cells
- **Mapping drift reduction**: RMSE from 4.2cm to 1.8cm with the height consistency drift compensation over 10m traversals
- **Selective privileged training** improves student performance by **12%** over full-privileged baseline (distillation gap reduced)
- **Safe fallback trigger rate**: 95% true positive rate for detecting impassable gaps, with <3% false positive rate
- **Velocity modulation**: average speed reduction of 40% when risk score exceeds threshold, with no missed gaps in the approach phase
- **Real-world demonstration** on Unitree Go1 traversing outdoor gaps (sidewalk cracks, drainage channels, construction gaps)
- **Computation**: full pipeline (mapping + risk + policy) runs at **30Hz** on Jetson Orin NX

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
MARG is directly relevant to the Mini Cheetah project as it addresses gap terrain navigation, which is one of the most critical failure modes for quadruped locomotion. The risk-informed RL framework could be integrated into the Mini Cheetah's training pipeline to develop safe locomotion policies. The LiDAR-based elevation mapping with drift compensation provides a practical perception solution that could be adapted for the Mini Cheetah's sensor suite. The selective privileged information approach could improve the efficiency of Mini Cheetah's teacher-student training by reducing the distillation gap.

The gap negotiation strategies (approach, straddle, leap) are directly applicable to the Mini Cheetah's locomotion repertoire, especially given the similar quadruped morphology. The terrain curriculum focused on progressive gap difficulty aligns with the project's curriculum learning approach.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
MARG's risk assessment module is directly applicable to Cassie's Safety level in the 4-level hierarchy. The concept of terrain risk scoring and risk-conditioned policy behavior maps to the CBF-QP (Control Barrier Function - Quadratic Programming) safety framework planned for Cassie. The risk assessment network could serve as the learned barrier function input, providing terrain-based safety constraints to the QP solver.

The selective privileged information training is relevant to Cassie's CPTE training, where risk-relevant terrain features should be prioritized in the contrastive embedding. The safe fallback behavior (halt and retreat) provides a concrete example of the Safety level intervention that Cassie's LCBF (Learned CBF) needs to implement. The concept of risk-weighted survival bonus in the reward function could be adapted for Cassie's push recovery and balance maintenance training.

## What to Borrow / Implement
- **Risk assessment module** — implement a terrain risk scorer that feeds into Cassie's CBF-QP safety layer, using elevation map features to predict traversal risk
- **Selective privileged information** — apply the selective privileged approach to CPTE training, privileging only risk-critical terrain features to reduce distillation gap
- **Risk-weighted RL reward** — augment PPO reward function with risk-conditioned survival bonus for both Mini Cheetah and Cassie training
- **Elevation map drift compensation** — implement the height consistency check for real-time terrain mapping on physical robots
- **Safe fallback behavior learning** — train explicit fallback policies (halt, retreat, seek alternative path) triggered by high-risk terrain detection

## Limitations & Open Questions
- **Gap-specific focus** — the risk framework is heavily tailored to gap terrains; generalization to other risk types (thin ice, crumbling edges, moving surfaces) requires additional work
- **LiDAR dependency** — the mapping module assumes reliable LiDAR data; failure modes under adverse conditions (rain, dust, high reflectivity) are not addressed
- **Binary risk paradigm** — the risk assessment classifies terrain as safe/unsafe with limited gradation; a continuous risk spectrum with probabilistic uncertainty could enable more nuanced behavior
- **No multi-robot or dynamic environment considerations** — the system assumes a static environment; dynamic obstacles near gaps could significantly complicate the risk assessment
