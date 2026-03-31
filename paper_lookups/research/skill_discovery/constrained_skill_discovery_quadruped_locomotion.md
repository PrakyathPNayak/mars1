---
## 📂 FOLDER: research/skill_discovery/

### 📄 FILE: research/skill_discovery/constrained_skill_discovery_quadruped_locomotion.md

**Title:** Constrained Skill Discovery: Quadruped Locomotion with Unsupervised Reinforcement Learning
**Authors:** Christoph Heindl, Thomas Pöll, Jurgen Hochreiter
**Year:** 2024
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2410.07877

**Abstract Summary (2–3 sentences):**
This paper extends unsupervised skill discovery (building on DIAYN and DADS) by introducing distance constraints in the latent skill space to produce more stable and practically usable locomotion behaviors for quadruped robots. The constraint prevents skills from collapsing into similar behaviors or diverging into unstable motions by enforcing that skill embeddings maintain a minimum separation while remaining within a bounded region. Applied to ANYmal quadruped locomotion, the method discovers a wider range of stable gaits than unconstrained DIAYN, including walking, trotting, and bounding primitives suitable for hierarchical control.

**Core Contributions (bullet list, 4–7 items):**
- Introduces distance constraints in the latent skill space to improve stability and diversity of discovered locomotion behaviors
- Proposes a constrained optimization formulation for skill discovery that extends DIAYN's mutual information objective with Lagrangian penalty terms
- Demonstrates more stable and usable locomotion skills than unconstrained DIAYN on a realistic quadruped (ANYmal) simulation
- Discovers diverse locomotion gaits (walking, trotting, bounding, turning) without any extrinsic reward signal
- Provides analysis showing that constrained skills have lower variance in execution quality, making them more reliable for downstream hierarchical RL
- Evaluates the quality of discovered skills for downstream task completion, showing improved performance over unconstrained baselines
- Addresses a key practical limitation of DIAYN: many discovered skills are unstable, redundant, or task-irrelevant

**Methodology Deep-Dive (3–5 paragraphs):**
The paper starts from the DIAYN framework where skills z are sampled from a prior p(z) and trained to maximize mutual information I(z; s) with visited states. The key observation is that unconstrained DIAYN often discovers skills that are: (1) unstable — the robot falls or enters degenerate configurations, (2) redundant — multiple skill codes produce nearly identical behaviors, or (3) impractical — behaviors that are distinguishable in state space but useless for locomotion (e.g., spinning in place). These issues arise because the DIAYN objective only cares about skill distinguishability, not skill quality or usability.

The authors introduce two types of constraints on the skill latent space. The first is a minimum distance constraint: skill embeddings in the latent space must maintain a minimum pairwise distance, preventing skill collapse (where multiple z values produce similar behaviors). This is implemented as a repulsive penalty in the loss function, computed over sampled skill pairs. The second is a maximum distance (boundedness) constraint: skill embeddings must remain within a bounded region, preventing divergence into extreme or unstable behaviors. Together, these constraints create a well-structured skill space where skills are spread out (diverse) but bounded (stable). The constraints are incorporated via Lagrangian multipliers, turning the problem into a constrained optimization solved with primal-dual gradient descent.

The training procedure alternates between: (1) policy update — maximize the DIAYN intrinsic reward r(s,z) = log q(z|s) - log p(z) plus constraint satisfaction penalties, (2) discriminator update — train q(z|s) to classify skills from states, and (3) Lagrange multiplier update — adjust constraint weights based on constraint violation. The policy is trained with SAC (Soft Actor-Critic). The constraint penalties are soft — they allow temporary violation during training but drive the system toward satisfaction as the Lagrange multipliers increase. An annealing schedule gradually increases the constraint strictness over training, allowing initial exploration followed by convergence to a structured skill space.

For the quadruped locomotion application, the authors construct a detailed simulation of the ANYmal robot (12 DoF, similar to Mini Cheetah) in Isaac Gym. The observation space includes joint angles, angular velocities, body orientation, and foot contact information. The skill space is continuous with z ∈ R^2, enabling visualization of the skill landscape. After training, the 2D skill space organizes into regions corresponding to different locomotion modes: forward walking, lateral walking, turning, trotting, and bounding. The organization is smooth — neighboring points in skill space produce similar but gradually varying behaviors, enabling interpolation.

The evaluation compares constrained skill discovery against vanilla DIAYN and DADS on three metrics: (1) skill diversity — how many semantically distinct locomotion modes are discovered, (2) skill stability — what fraction of discovered skills produce stable locomotion without falling, and (3) downstream task performance — how well a meta-controller can compose skills for velocity tracking. Constrained skill discovery outperforms on all three metrics. In particular, the stability improvement is dramatic: ~90% of constrained skills produce stable locomotion vs ~50% for unconstrained DIAYN. The downstream velocity tracking task shows ~30% improvement when using constrained skills as primitives.

**Key Results & Numbers:**
- ~90% of constrained skills produce stable locomotion vs ~50% for unconstrained DIAYN
- Discovers 6–8 semantically distinct locomotion gaits (walking, trotting, bounding, turning variants)
- ~30% improvement in downstream velocity tracking when using constrained vs unconstrained skills as primitives
- Demonstrated on ANYmal quadruped (12 DoF) in Isaac Gym simulation
- 2D continuous skill space shows smooth, interpretable organization of locomotion modes
- Lower execution variance: constrained skills have ~40% lower return variance per skill
- No extrinsic reward used during skill discovery phase

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable to discovering locomotion primitives for Mini Cheetah. ANYmal and Mini Cheetah have similar morphologies (12 DoF quadruped), so the results should transfer well. The constrained skill discovery could provide a set of stable, diverse locomotion gaits (walk, trot, bound, gallop) that a higher-level controller selects from based on terrain and speed commands. The stability guarantee (~90% stable skills) addresses a key practical concern for real-robot deployment.

**Relevance to Project B (Cassie HRL):** HIGH — Directly relevant to the Primitives level of Cassie's 4-level HRL hierarchy. The improved stability and diversity of discovered skills addresses the main practical weakness of DIAYN for locomotion: many DIAYN skills are unusable. Constrained skill discovery would provide a more reliable set of primitives for the Planner to compose. The smooth 2D skill space enables continuous interpolation between gaits, which maps to the Planner's continuous skill selection interface. The Lagrangian constraint enforcement methodology also parallels the LCBF safety constraint approach used at the Safety level.

**What to Borrow / Implement:**
- Constrained skill discovery for both Mini Cheetah and Cassie Primitives: add distance constraints to prevent skill collapse and instability
- Lagrangian penalty formulation for constraint enforcement during skill training — the same primal-dual approach used in the LCBF Safety level
- Continuous 2D skill space with smooth interpolation — use as the Planner→Primitives interface, where the Planner outputs continuous skill parameters
- Stability-aware skill evaluation: after discovery, filter or rank skills by stability before exposing them to the meta-controller
- The constraint annealing schedule (loose early, strict late) to allow exploration before convergence — applicable to training curriculum design

**Limitations & Open Questions:**
- Only demonstrated in simulation (Isaac Gym) — no real-robot validation
- The 2D skill space may be too low-dimensional to capture the full diversity of locomotion behaviors needed for complex terrains
- Distance constraints are defined in the latent embedding space, not in behavior space — latent distance may not perfectly correlate with behavioral difference
- The method does not incorporate terrain awareness into skill discovery — skills may not be suitable for all terrains
- Integration with safety constraints (LCBF) during skill discovery is not explored — how to discover safe-by-construction skills is an open question
- Scaling to higher-dimensional skill spaces (needed for bipedal locomotion with more behavioral dimensions) is not evaluated
---
