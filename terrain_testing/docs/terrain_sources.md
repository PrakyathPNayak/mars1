# Terrain Sources — Paper Citations

All terrain implementations in this suite are derived from or inspired by the
following papers. Each entry lists: terrain name in registry, paper, key idea used.

---

## Paper-Derived Terrains

### 1. Isaac Gym / legged_gym Terrain Suite
**Paper:** Rudin et al. 2022 — "Learning to Walk in Minutes Using Massively
Parallel Deep Reinforcement Learning"
**Source:** https://arxiv.org/abs/2109.11978
**Terrains used:** `pyramid_stairs`, `discrete_obstacles`, `wave`, `sloped_terrain`
**Key idea:** Curriculum over terrain difficulty; heightfield resolution ~20 cm.
Mixed terrain with slope + stairs combo for curriculum progression.

### 2. RMA — Rapid Motor Adaptation
**Paper:** Kumar et al. 2021 — "RMA: Rapid Motor Adaptation for Legged Robots"
**Source:** https://arxiv.org/abs/2107.04559
**Terrains used:** `rma_rough`, `rma_stepping_stones`
**Key idea:** Extrinsics vector encodes terrain from proprioception. Rough terrain
σ = 0.02 m, stepping stones with 20–40 cm spacing tested on A1 hardware.

### 3. DreamWaQ
**Paper:** Nahrendra et al. 2023 — "DreamWaQ: Learning Robust Quadrupedal Locomotion
with World Models"
**Source:** https://arxiv.org/abs/2301.03652
**Terrains used:** `dreamwaq_rough`, `dreamwaq_mixed`
**Key idea:** Blind locomotion over rough terrain; proprioceptive history captures
terrain from foot contact patterns. Terrain σ up to 5 cm.

### 4. Walk These Ways
**Paper:** Margolis & Agrawal 2023 — "Walk These Ways: Tuning Robot Walking for
Generalisation with Numerous Costs"
**Source:** https://arxiv.org/abs/2212.03238
**Terrains used:** `walk_these_ways_rough`, `walk_these_ways_stairs`
**Key idea:** Multi-gait training over rough terrain. Stair height 10–20 cm.

### 5. PGTT — Progressive Generalization with Terrain Types
**Paper:** Yang et al. 2023
**Terrains used:** `pgtt_progressive`
**Key idea:** Curriculum starts flat → rough → slopes → stairs → gaps. Difficulty
gates: advance only when 80% success rate reached.

### 6. Anymal Hwangbo 2019
**Paper:** Hwangbo et al. 2019 — "Learning Agile and Dynamic Motor Skills for
Legged Robots"
**Source:** https://arxiv.org/abs/1901.08652
**Terrains used:** `anymal_rough`, `anymal_steps`
**Key idea:** 2 cm rough terrain + 15 cm step obstacles. Domain randomization over
terrain friction (0.4–1.0).

### 7. Blind Stair Climbing
**Paper:** Zhuang et al. 2024 — "Robot Parkour Learning"
**Terrains used:** `parkour_wall`, `parkour_hurdle`, `parkour_gap`
**Key idea:** High obstacles (30–80 cm), gaps (20–60 cm), walls requiring jumping.
Uses privileged teacher for initial training.

### 8. CHRL — Comprehensive Hierarchical RL Survey
**Paper:** Nature Scientific Reports 2024
**Terrains used:** `chrl_mixed`, `chrl_challenge`
**Key idea:** Combination terrain: rough + stairs + gaps in sequence. Push
perturbations 50–80 N every 5–10 s.

---

## Custom Terrains (your friend's inventions)

### C1. `crater_field`
Random bowl-shaped depressions + ridges. Tests lateral stability.

### C2. `tunnel_exit`
Flat approach → narrow elevated corridor → drop. Tests height adaptation.

### C3. `sand_dunes`
Smooth sinusoidal terrain with varying frequency. Tests energy efficiency.

### C4. `rubble_field`
Dense random blocks of varying height (3–15 cm). Densest obstacle environment.

### C5. `asymmetric_slope`
Left side slopes up, right side slopes down simultaneously. Tests roll compensation.

### C6. `frozen_lake`
Very low friction (0.1) flat terrain. Tests pure friction robustness.

### C7. `trench_crossing`
Parallel trenches perpendicular to motion. Tests stride length adaptation.

---

## Notes on Difficulty Scaling

All terrains accept `difficulty` in [0.0, 1.0]:
- 0.0 → easiest variant (nearly flat / wide gaps / shallow stairs)
- 0.5 → paper-reported nominal difficulty
- 1.0 → hardest variant (maximum obstacle height / narrowest gaps)

Recommended starting difficulty for curriculum: 0.1, advance by 0.1 per 500K steps.
