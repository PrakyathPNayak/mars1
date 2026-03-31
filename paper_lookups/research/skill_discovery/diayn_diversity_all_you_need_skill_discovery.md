---
## 📂 FOLDER: research/skill_discovery/

### 📄 FILE: research/skill_discovery/diayn_diversity_all_you_need_skill_discovery.md

**Title:** Diversity is All You Need: Learning Skills without a Reward Function
**Authors:** Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine
**Year:** 2019
**Venue:** ICLR 2019
**arXiv / DOI:** arXiv:1802.06070

**Abstract Summary (2–3 sentences):**
DIAYN (Diversity Is All You Need) proposes an unsupervised skill discovery framework that learns a diverse set of skills without any task-specific reward function by maximizing the mutual information between skill latent codes and visited states. The method uses an information-theoretic objective that encourages skills to be distinguishable (each skill visits distinct states) while being as random as possible in their actions given the skill identity, preventing degenerate solutions. Applied to locomotion in MuJoCo, DIAYN discovers diverse behaviors including running, jumping, and turning, which can be composed for downstream hierarchical RL tasks.

**Core Contributions (bullet list, 4–7 items):**
- Proposes an unsupervised skill discovery objective based on maximizing mutual information I(z; s) between skill code z and visited states s
- Decomposes the objective into a discriminability reward (can we identify the skill from the visited states?) minus a conditional entropy penalty
- The discriminator q(z|s) provides the intrinsic reward: r(s, z) = log q(z|s) - log p(z), requiring no extrinsic reward engineering
- Discovers diverse locomotion behaviors (running, jumping, turning, flipping) in MuJoCo environments without any task reward
- Demonstrates that discovered skills form useful primitives for hierarchical RL — a meta-controller can compose skills for downstream tasks
- Shows that DIAYN skills can be used for imitation learning and exploration in sparse-reward environments
- Provides theoretical connections to variational inference and the information bottleneck

**Methodology Deep-Dive (3–5 paragraphs):**
DIAYN's core objective is to maximize the mutual information I(z; s) between a categorical skill variable z ~ p(z) (sampled uniformly from a fixed set of K skills) and the states s visited by the policy π(a|s, z). Expanding the mutual information, I(z; s) = H(z) - H(z|s), DIAYN aims to: (1) maximize H(z) — use all skills equally (ensured by sampling z uniformly), and (2) minimize H(z|s) — make states visited by different skills distinguishable. Minimizing H(z|s) is equivalent to maximizing the discriminability of skills from states. The framework also adds a term to minimize I(z; a|s), encouraging skills to be distinguished by what states they visit, not by the specific actions taken, promoting diversity in state-space rather than action-space.

The practical implementation uses a variational lower bound on I(z; s). A learned discriminator network q_φ(z|s) approximates the posterior p(z|s), and the intrinsic reward for skill z is defined as r(s, z) = log q_φ(z|s) - log p(z). The first term rewards the agent for visiting states where the discriminator can confidently identify the current skill, while the second term (a constant for uniform p(z)) provides a baseline. The discriminator is trained simultaneously with the policy using cross-entropy loss to classify skills from states. The policy π_θ(a|s, z) is trained with any standard RL algorithm (the authors use SAC) to maximize this intrinsic reward.

During training, an episode proceeds as follows: (1) sample a skill z ~ Uniform(1, K), (2) run the policy π(a|s, z) for the full episode, collecting states, (3) compute intrinsic rewards using the discriminator, (4) update the policy to maximize intrinsic rewards, and (5) update the discriminator to better classify skills from visited states. This alternating optimization drives the policy to discover increasingly diverse and distinguishable behaviors. As the discriminator improves, the policy must find more distinct state visitation patterns to maintain high reward, creating a curriculum of increasing diversity.

For hierarchical RL applications, DIAYN is used in a two-phase approach. First, skills are discovered through unsupervised pre-training (phase 1). Then, a meta-controller (higher-level policy) is trained to select among the discovered skills to solve downstream tasks with sparse or complex rewards (phase 2). The meta-controller operates at a lower frequency, selecting a skill z every H timesteps, while the skill policy π(a|s, z) executes at the environment frequency. This hierarchical decomposition enables the meta-controller to plan over a reduced action space (K discrete skills instead of continuous torques), significantly simplifying the downstream learning problem.

The authors evaluate DIAYN on several MuJoCo environments including HalfCheetah, Ant, and Humanoid. Qualitative analysis shows that discovered skills correspond to semantically meaningful behaviors: forward/backward running at different speeds, turning left/right, jumping, and crawling. Quantitatively, the authors show that DIAYN skills, when composed by a meta-controller, can solve downstream tasks (reaching specific locations, following trajectories) more efficiently than training from scratch with the same RL algorithm. They also demonstrate that DIAYN-discovered skills provide better exploration in sparse-reward environments compared to random exploration.

**Key Results & Numbers:**
- Discovers diverse locomotion skills (running, jumping, turning, flipping) in MuJoCo without task-specific rewards
- Typically discovers 8–20 semantically distinct skills with K=50 total skill codes
- Skills useful for downstream hierarchical RL: 2–5× faster learning on goal-reaching tasks when using DIAYN skills vs learning from scratch
- Demonstrated on HalfCheetah, Ant, Humanoid, and Mountain Car environments
- Skills provide effective exploration in sparse-reward environments
- Discriminator accuracy typically reaches 80–95% for distinguishing between 20+ skills

**Relevance to Project A (Mini Cheetah):** MEDIUM — DIAYN could be used to pre-discover diverse locomotion primitives for Mini Cheetah (different gaits, speeds, turning behaviors) without hand-engineering rewards for each behavior. These could serve as a behavior repertoire that a higher-level controller selects from. However, Mini Cheetah's current approach uses a single PPO policy with explicit reward shaping, so DIAYN would represent a paradigm shift.

**Relevance to Project B (Cassie HRL):** HIGH — DIAYN is directly used (or directly inspires) the skill/primitive discovery at the Primitives level of Cassie's 4-level HRL system. The Planner selects among primitives, and DIAYN provides the mechanism for discovering these primitives without manually engineering a reward for each gait type. The mutual information objective naturally discovers locomotion modes (walking, running, turning) that the Planner can compose. The hierarchical composition framework (meta-controller over skills) maps directly to the Planner→Primitives interface.

**What to Borrow / Implement:**
- Use DIAYN for unsupervised primitive discovery in Cassie's Primitives level: pre-train diverse locomotion skills, then let the Planner select among them
- The discriminator r(s, z) = log q(z|s) - log p(z) as the intrinsic reward for primitive training — no manual reward engineering per primitive
- Two-phase training: (1) unsupervised DIAYN pre-training to discover primitives, (2) supervised/RL training of the Planner to compose primitives for target tasks
- State-based (not action-based) skill diversity ensures primitives are distinguished by their locomotion outcomes, not low-level motor patterns
- For Mini Cheetah: DIAYN could discover a behavior repertoire for multi-task deployment without per-task reward engineering

**Limitations & Open Questions:**
- Discovered skills may not align with task-relevant behaviors — DIAYN finds diverse skills, but not necessarily useful ones for a specific downstream task
- The number of skill codes K is a hyperparameter; too few limits diversity, too many leads to redundant or degenerate skills
- Skills are discovered in state space, but locomotion tasks often require skills defined in terms of velocity, heading, or other derived quantities
- No mechanism for skill composability — skills are atomic and cannot be smoothly blended or interpolated
- The discriminator may find spurious distinctions (e.g., different initial conditions) rather than meaningful behavioral differences
- Does not account for robot safety constraints during skill discovery — discovered skills may include dangerous behaviors
---
