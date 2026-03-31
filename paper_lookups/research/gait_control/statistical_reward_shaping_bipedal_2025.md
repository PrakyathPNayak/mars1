# Statistical Reward Shaping for Reinforcement Learning in Bipedal Locomotion

**Authors:** (MDPI Electronics 2025)
**Year:** 2025 | **Venue:** MDPI Electronics
**Links:** https://www.mdpi.com/2079-9292/15/6/1203

---

## Abstract Summary
This paper presents a data-driven statistical approach to reward shaping for bipedal locomotion reinforcement learning. Instead of relying on domain expertise and manual tuning to design reward functions, the authors propose a systematic methodology that uses regression analysis, correlation analysis, and statistical hypothesis testing to identify the most impactful reward components and their optimal weights. The approach treats reward function design as a statistical modeling problem: given a dataset of training runs with varying reward configurations, statistical methods identify which reward terms most strongly predict desired outcomes (forward distance, fall rate, posture quality).

The methodology proceeds in three phases: (1) an exploration phase where diverse reward configurations are sampled and evaluated via short training runs, generating a dataset of (reward_config → performance_metrics) pairs; (2) a statistical analysis phase using multiple regression, Pearson/Spearman correlation, and ANOVA to identify high-impact reward terms and their interactions; and (3) a refinement phase where the statistically-optimized reward is used for full training. This approach replaces ad-hoc reward tuning with a principled, reproducible optimization process.

Applied to bipedal locomotion tasks (Cassie-like and Humanoid environments in MuJoCo), the statistical reward shaping method achieves significant improvements in gait learning speed (40% faster convergence), walking stability (35% fewer falls), and posture quality compared to manually-tuned baselines. The methodology is environment-agnostic and can be applied to any RL task where reward function design is challenging.

## Core Contributions
- A systematic statistical methodology for reward function design that replaces manual tuning with data-driven analysis using regression, correlation, and ANOVA
- Three-phase pipeline: exploration (diverse reward sampling), analysis (statistical identification of high-impact terms), and refinement (optimized reward training)
- Identification of critical reward term interactions that are difficult to discover through manual tuning (e.g., synergistic effects between foot clearance and hip torque penalties)
- 40% faster convergence and 35% fewer falls compared to manually-tuned reward baselines on bipedal locomotion tasks
- Statistical evidence that reward term weights follow non-linear interaction patterns, challenging the common assumption of additive reward decomposition
- Reproducible reward optimization protocol that reduces the expertise barrier for reward function design
- Comprehensive analysis on both Cassie-like (10-DoF) and Humanoid (21-DoF) bipedal environments, demonstrating generality

## Methodology Deep-Dive
The exploration phase uses Latin Hypercube Sampling (LHS) to efficiently cover the reward weight space. Given N reward terms (typically 8–12 for bipedal locomotion), LHS generates M=50–100 reward configurations that uniformly sample the weight space. Each configuration is evaluated via a short training run (500 PPO iterations with 1024 parallel environments, ~30 minutes each) producing performance metrics: average forward distance, fall rate, average episode length, posture deviation (roll/pitch RMSE), energy consumption, and velocity tracking error.

The statistical analysis phase applies three complementary methods. First, multiple linear regression fits the model: performance_metric = β_0 + Σ β_i · w_i + Σ β_ij · w_i · w_j + ε, where w_i are reward weights and β_ij capture pairwise interactions. The regression coefficients β_i rank each reward term's impact on each performance metric. Second, Pearson and Spearman correlation analysis identifies monotonic and non-monotonic relationships between individual reward weights and outcomes, flagging reward terms with high correlation to desired outcomes. Third, one-way ANOVA with post-hoc Tukey tests identifies reward weight ranges that produce statistically distinct performance clusters.

Critical findings from the statistical analysis on bipedal locomotion include: (1) the foot clearance reward has the highest single-term impact on fall rate (β = -0.42, p < 0.001), (2) a strong synergistic interaction exists between hip torque penalty and forward velocity reward (β_interaction = 0.31, p < 0.01), meaning both must be present for effective gait learning, (3) the energy penalty weight has a non-monotonic relationship with forward distance—moderate penalties improve efficiency without sacrificing speed, but excessive penalties cause the robot to barely move, and (4) action smoothness penalties beyond a threshold produce diminishing returns and can even harm performance.

The refinement phase takes the statistically-optimized reward weights and runs full-scale training (5000 PPO iterations with 4096 parallel environments). Optionally, a second round of exploration around the statistical optimum can further fine-tune weights, though the authors find the first-round analysis sufficient in most cases.

The PPO training configuration uses standard hyperparameters: clip ε=0.2, γ=0.99, λ_GAE=0.95, learning rate 3×10⁻⁴ with linear decay. The policy network is a 2-layer MLP (256×256) with ELU activations. Domain randomization includes friction (0.3–1.5), payload (0–5 kg), joint damping (±15%), motor strength (±10%), and terrain noise (±3 cm for Cassie, ±5 cm for Humanoid).

## Key Results & Numbers
- Convergence speed: 40% fewer PPO iterations to reach 90% of final performance compared to manually-tuned rewards
- Fall rate: 35% reduction (from 28% to 18% per episode) with statistically-optimized rewards on Cassie-like environment
- Forward velocity tracking RMSE: 0.08 m/s (statistical) vs. 0.13 m/s (manual) on Cassie-like environment
- Posture quality (roll/pitch RMSE): 3.2° (statistical) vs. 5.1° (manual), 37% improvement
- Energy consumption: 15% reduction with statistically-optimized energy penalty weight
- Statistical analysis time: ~50 hours for 100 exploration runs (parallelizable to ~5 hours on 10 GPUs)
- Interaction effect: hip torque × velocity reward interaction contributes 18% of explained variance in forward distance
- Transferability: reward weights optimized for Cassie-like environment achieve 80% of optimal performance on Humanoid without re-tuning

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The statistical reward shaping methodology is applicable to Mini Cheetah's reward function design, though it was developed for bipedal environments. The three-phase pipeline (exploration, analysis, refinement) can be applied to Mini Cheetah's locomotion reward to systematically identify optimal weights for velocity tracking, energy penalty, gait quality, and stability terms. The Latin Hypercube Sampling approach for reward weight exploration is more efficient than the grid search or random search typically used for reward tuning.

The discovery of non-linear reward term interactions is particularly relevant—Mini Cheetah's reward likely has similar synergistic effects between terms that manual tuning misses. The 40% faster convergence would accelerate Mini Cheetah's already-long training pipeline, and the reproducibility of the statistical approach improves the scientific rigor of reward design.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is critically relevant to Cassie's project. The statistical reward shaping was specifically validated on Cassie-like bipedal environments, making the results directly transferable. The methodology addresses one of the central challenges in Cassie's hierarchical architecture: designing effective reward functions for each level of the hierarchy. The statistical approach can be applied independently to each level—optimizing Controller-level reward weights, Primitives-level diversity rewards, and Planner-level goal rewards.

The interaction analysis revealing synergistic effects between hip torque and velocity rewards provides direct design insight for Cassie's Controller-level reward function. The 35% fall rate reduction and 37% posture improvement would significantly enhance Cassie's walking stability and quality. The methodology's reproducibility is especially valuable for Cassie's complex multi-level reward specification, where manual tuning across four hierarchy levels is impractical.

## What to Borrow / Implement
- Apply the three-phase statistical reward optimization pipeline to both Mini Cheetah and Cassie reward function design
- Use Latin Hypercube Sampling for efficient exploration of reward weight spaces, especially for Cassie's multi-level rewards
- Implement the regression + correlation + ANOVA analysis to identify high-impact reward terms and their interactions for each project
- Leverage the discovered bipedal reward interactions (foot clearance, hip torque × velocity synergy) directly in Cassie's Controller-level reward
- Adopt the non-monotonic energy penalty finding to calibrate energy regularization weights for both platforms

## Limitations & Open Questions
- The exploration phase requires 50–100 short training runs, which is computationally expensive (~50 GPU-hours) even though each run is short; this cost scales with the number of reward terms
- Statistical analysis assumes the relationship between reward weights and performance can be captured by regression with pairwise interactions; higher-order interactions or non-parametric relationships may be missed
- The method optimizes reward weights for a fixed set of reward terms; it does not discover new reward terms, which limits its utility when the reward function structure itself is suboptimal
- Transfer of statistically-optimized weights across significantly different environments (e.g., from Cassie to Mini Cheetah) shows degradation, suggesting the method must be rerun for each target platform
