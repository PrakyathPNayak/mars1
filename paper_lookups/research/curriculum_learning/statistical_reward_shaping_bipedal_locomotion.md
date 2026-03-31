# Statistical Reward Shaping for Reinforcement Learning in Bipedal Locomotion

**Authors:** Various
**Year:** 2025 | **Venue:** MDPI Electronics
**Links:** https://www.mdpi.com/2079-9292/15/6/1203

---

## Abstract Summary
This paper uses data-driven statistical methods including parameter sweeps, regression analysis, and correlation studies to systematically design and optimize reward functions for bipedal humanoid robot locomotion. The approach identifies how individual reward terms affect gait stability and performance metrics. The methodology reduces fall rates from 75% to 2% through principled reward optimization rather than manual trial-and-error tuning.

## Core Contributions
- Data-driven methodology for systematic reward function optimization in bipedal locomotion
- Statistical analysis (regression, correlation) revealing individual reward term impacts on performance
- Reduction of fall rates from 75% to 2% through principled reward design
- Parameter sweep framework for exploring reward weight hyperparameter space
- Quantitative analysis of reward term interactions and their effects on gait quality
- Practical guidelines for reward function design based on statistical evidence
- Validated on bipedal humanoid platform in simulation

## Methodology Deep-Dive
The paper takes an empirical, data-driven approach to the reward function design problem that plagues RL for locomotion. Rather than hand-tuning reward weights through intuition and trial-and-error, the authors conduct systematic parameter sweeps over the space of reward coefficients and analyze the resulting policy behaviors using statistical methods.

The experimental setup involves training a bipedal humanoid locomotion policy with PPO across hundreds of reward configurations. Each configuration varies the weights of standard reward terms: velocity tracking, energy penalty, joint acceleration penalty, foot contact reward, orientation penalty, symmetry reward, and others. For each configuration, the resulting policy is evaluated on metrics including fall rate, forward velocity, gait smoothness, energy consumption, and gait symmetry.

Regression analysis is then applied to the dataset of (reward weights → performance metrics) to identify which reward terms most strongly influence each performance metric. For example, the analysis might reveal that the orientation penalty weight has a strong negative correlation with fall rate (higher weight → fewer falls) while the velocity tracking weight has a strong positive correlation with forward speed but a weaker positive correlation with fall rate. Interaction effects between reward terms are also analyzed—e.g., the energy penalty is only effective when the foot contact reward is sufficiently high.

Correlation analysis further reveals redundancies and conflicts between reward terms. Some terms are highly correlated in their effect (and thus one is redundant), while others have opposing effects that require careful balancing. This statistical evidence replaces subjective intuition with quantitative guidance for reward design.

The optimized reward function, derived from the statistical analysis, dramatically improves performance. The fall rate reduction from 75% to 2% is achieved not by adding new reward terms but by optimally weighting existing ones based on statistical evidence. The paper provides a practical recipe: start with the statistically-identified important terms, set weights according to the regression model, and fine-tune with small sweeps around the predicted optimum.

## Key Results & Numbers
- Fall rate reduced from 75% to 2% through data-driven reward optimization
- Hundreds of reward configurations tested systematically via parameter sweeps
- Regression analysis identifies top 3-4 most impactful reward terms out of 10+
- Interaction effects between reward terms quantified (e.g., energy penalty × contact reward)
- Optimal reward weight regions identified with confidence intervals
- Gait smoothness improved by 40% with optimized reward weights
- Energy consumption reduced by 25% while maintaining locomotion quality
- Methodology reproducible and applicable to any reward-based RL locomotion setup

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The systematic reward analysis methodology is applicable to Mini Cheetah's PPO training pipeline. Rather than manually tuning reward weights for the 12-DoF quadruped, the parameter sweep approach could identify optimal reward configurations more efficiently. The statistical insights about which reward terms matter most could reduce the dimensionality of the reward tuning problem. However, the specific results (which terms matter, optimal weights) are for bipedal humanoids and won't directly transfer to quadrupeds—the methodology transfers, not the specific findings. Running parameter sweeps in MuJoCo for Mini Cheetah would be computationally feasible with the existing infrastructure.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
This paper is directly applicable to Project B's reward function design for Cassie. The statistical methodology can be applied to each level of the 4-level hierarchy to optimize rewards independently. The dramatic fall rate reduction (75% → 2%) suggests significant room for improvement through principled reward optimization. The regression analysis approach can identify which reward terms are most important at the Controller level, where Cassie's stability is most directly affected. The interaction effects analysis is particularly valuable for the multi-objective optimization across hierarchy levels—understanding how rewards at different levels interact is critical for the hierarchical training. The methodology complements the constraint-based approach (Paper 4) by providing statistical evidence for which objectives should be constraints vs. rewards.

## What to Borrow / Implement
- Implement the parameter sweep framework for reward weight optimization in both projects
- Apply regression analysis to identify the most impactful reward terms for Mini Cheetah and Cassie
- Use correlation analysis to detect redundant reward terms and simplify reward functions
- Develop an automated reward optimization pipeline: sweep → analyze → predict optimal weights
- Combine with constraint-based formulation: use statistics to decide which terms become constraints
- Apply the methodology at each level of Project B's hierarchy to optimize level-specific rewards

## Limitations & Open Questions
- Parameter sweeps are computationally expensive (hundreds of training runs required)
- Linear regression may miss complex non-linear interactions between reward terms
- Results are specific to the training environment; reward optima may shift with domain randomization
- The approach assumes a fixed set of reward terms; it doesn't discover new useful terms
- How to extend the methodology to hierarchical reward structures with inter-level dependencies?
- Scaling to very large reward term sets (20+) makes parameter sweeps intractable
