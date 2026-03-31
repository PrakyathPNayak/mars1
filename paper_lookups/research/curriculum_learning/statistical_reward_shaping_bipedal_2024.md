# Statistical Reward Shaping for Reinforcement Learning in Bipedal Locomotion

**Authors:** (MDPI Electronics Contributors)
**Year:** 2024 | **Venue:** MDPI Electronics
**Links:** [DOI: 10.3390/electronics15061203](https://www.mdpi.com/2079-9292/15/6/1203)

---

## Abstract Summary
This paper presents a statistical approach to reward function design for bipedal locomotion reinforcement learning. Rather than relying on intuition-driven manual tuning of reward weights, the authors apply correlation analysis, regression modeling, and systematic parameter optimization to refine reward function components. The method identifies which reward terms most strongly predict desired locomotion outcomes (forward progress, stability, energy efficiency) and adjusts their relative weights accordingly using data-driven techniques.

The results are striking: the statistically optimized reward function nearly doubles locomotion efficiency (measured as distance traveled per unit energy) compared to baseline hand-tuned rewards, while reducing fall rates by over 60%. The approach is demonstrated on a simulated bipedal walker in MuJoCo-like environments, with systematic ablation studies showing the contribution of each optimization step. The key insight is that reward function components interact non-linearly, making their optimal weighting impossible to determine by tuning one parameter at a time.

The methodology is general-purpose and can be applied to any RL locomotion task where multiple reward components are combined. The statistical analysis also reveals which reward components are redundant or counterproductive, enabling principled reward function simplification. This data-driven approach to reward engineering represents a middle ground between fully manual design and fully automated methods like Text2Reward.

## Core Contributions
- Statistical framework for reward function optimization using correlation analysis and regression
- Identification of non-linear interactions between reward components that explain failure of manual one-at-a-time tuning
- Near-doubling of locomotion efficiency and >60% reduction in fall rates compared to baseline rewards
- Principled methodology for detecting redundant or counterproductive reward terms
- Systematic ablation study isolating the contribution of correlation analysis, regression, and parameter optimization stages
- Reward function simplification through statistical significance testing of individual components
- Generalizable framework applicable beyond bipedal locomotion to any multi-objective RL reward

## Methodology Deep-Dive
The reward function optimization pipeline has three stages: correlation analysis, regression modeling, and iterative parameter optimization. Starting with a standard bipedal locomotion reward with N components (typically 6–10 terms including forward velocity, upright bonus, foot contact pattern, energy penalty, action smoothness, joint limit penalty, etc.), the authors first collect a dataset of training trajectories from policies trained with random reward weight vectors sampled from a predefined range.

In the correlation analysis stage, Pearson and Spearman correlation coefficients are computed between each reward component's cumulative value per episode and the target performance metrics (distance traveled, time alive, energy consumed). This reveals which reward terms are most predictive of desired outcomes. Surprisingly, some commonly used reward components (e.g., certain joint limit penalties) show near-zero or even negative correlation with performance, indicating they may hinder learning. Components with high cross-correlation are flagged as potentially redundant.

The regression stage fits multivariate models (linear regression, polynomial regression up to degree 3, and gradient-boosted regression trees) mapping reward weight vectors to performance metrics. The polynomial and tree models capture the non-linear interactions between reward components that the correlation analysis hints at. For example, the energy penalty weight and forward velocity reward weight interact strongly: high energy penalty with high velocity reward creates a difficult optimization landscape, while moderate values of both produce the best gaits.

The iterative parameter optimization stage uses the regression model as a surrogate objective and applies Bayesian optimization (specifically, Tree-structured Parzen Estimator / TPE) to find the weight vector maximizing predicted performance. The resulting weights are validated by training new policies and measuring actual performance. This surrogate-based optimization requires far fewer RL training runs than direct grid or random search over the weight space. The authors use 50–100 initial random training runs to fit the regression model, then 10–20 Bayesian optimization iterations to refine, compared to the thousands of runs that exhaustive search would require.

A critical implementation detail is the normalization strategy for reward components. Since different components operate on different scales (velocity in m/s vs. energy in Watts vs. angular deviation in radians), the authors normalize each component to zero mean and unit variance before correlation and regression analysis. This prevents scale effects from dominating the statistical analysis and ensures the weight optimization is operating in a balanced parameter space.

The bipedal walker model has 6 actuated joints per leg, a floating base, and makes contact through point feet. The observation space includes joint angles, joint velocities, body orientation (quaternion), body angular velocity, and foot contact booleans. The action space is continuous joint torques. PPO is used as the RL algorithm throughout, with fixed hyperparameters to isolate the effect of reward function changes.

## Key Results & Numbers
- Locomotion efficiency improved from 0.45 m/J to 0.82 m/J (82% improvement) with optimized rewards
- Fall rate reduced from 35% to 12% of episodes (66% reduction)
- Number of statistically significant reward components reduced from 8 to 5 after redundancy analysis
- Polynomial regression (degree 2) R² = 0.78 for predicting performance from reward weights
- Bayesian optimization converges in 15 iterations (vs. >500 for random search)
- Total optimization pipeline requires ~100 PPO training runs (feasible in 1–2 days on a single GPU)
- Forward velocity reward and upright bonus identified as the two most predictive components (r > 0.7)
- Energy penalty shows non-linear interaction with velocity reward (negative interaction at extreme values)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
While the statistical reward optimization methodology is general-purpose and could in principle be applied to quadruped locomotion, this paper focuses specifically on bipedal walking dynamics. The specific reward components analyzed (bipedal foot contact patterns, single-leg stance stability) are not directly transferable to Mini Cheetah's quadrupedal gaits. However, the general framework of using correlation analysis to identify important reward terms and Bayesian optimization to tune weights could be adapted for the Mini Cheetah reward function. The main barrier is the computational cost of the initial data collection phase (100 training runs), which may be prohibitive if Mini Cheetah training is already expensive.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper is directly applicable to Cassie's bipedal locomotion reward design, particularly at the Controller level of the 4-level hierarchy. Cassie's reward function likely includes many of the same components analyzed here (forward velocity, upright posture, foot contact patterns, energy efficiency), and the statistical optimization framework could be applied to systematically tune these weights. The near-doubling of locomotion efficiency is especially attractive for Cassie, where energy efficiency directly impacts battery life and operational range.

The methodology is also applicable to the Primitives level, where DIAYN/DADS diversity rewards interact with task rewards in complex ways. Statistical analysis could identify the optimal balance between diversity encouragement and task performance. For the Safety level (LCBF), the framework could optimize the weighting between control barrier function constraints and performance objectives. The data-driven approach reduces the manual tuning burden that scales multiplicatively across the four hierarchy levels.

## What to Borrow / Implement
- Apply the correlation analysis pipeline to existing Cassie reward components to identify which terms are actually driving learning vs. adding noise
- Use Bayesian optimization (TPE) to tune reward weights for the Controller level, starting from the current hand-tuned values
- Adopt the reward component normalization strategy to ensure balanced multi-objective optimization
- Run redundancy analysis to simplify reward functions at each hierarchy level, potentially removing unnecessary terms that slow training
- Extend the regression modeling approach to capture interactions between rewards at different hierarchy levels

## Limitations & Open Questions
- Computational cost: requires ~100 training runs for the initial data collection, which may be expensive for complex environments
- Assumes reward function structure (which terms to include) is fixed; only optimizes weights, not the functional form
- Statistical analysis is retrospective—it identifies good weights for the observed training distribution but may not generalize to different initial conditions or curriculum stages
- The bipedal walker model is simpler than Cassie (fewer actuated joints, point feet vs. Cassie's compliant ankle mechanism)
