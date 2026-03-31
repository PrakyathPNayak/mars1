# Informed Asymmetric Actor-Critic: Leveraging Privileged Signals Beyond Ground Truth

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv](https://arxiv.org/abs/2509.26000)

---

## Abstract Summary
This paper challenges the common assumption in asymmetric actor-critic methods that the critic should always receive the fullest possible privileged information (i.e., ground-truth state). The authors demonstrate theoretically and empirically that not all privileged information is equally useful, and that providing certain types of privileged signals can actually degrade learning by introducing bias in the policy gradient estimates. The key insight is that only privileged information satisfying a conditional independence property — specifically, information that helps predict future rewards conditioned on the actor's observation — yields unbiased advantage estimates. Information that merely helps predict value without improving reward-conditional predictions introduces systematic bias.

The paper introduces a principled framework, Informed Asymmetric Actor-Critic (IAAC), for selecting which privileged signals to provide to the critic. IAAC uses a mutual information criterion: privileged features are included only if they have high mutual information with future returns conditional on the actor's observation. Features with high unconditional mutual information but low conditional mutual information are explicitly excluded, as they provide the critic with "unreachable" information that biases the advantage function. The framework provides a practical algorithm based on variational mutual information estimation for automatic privileged feature selection.

Experiments across locomotion (quadruped, biped), manipulation, and navigation tasks demonstrate that IAAC consistently outperforms both standard (no privileged info) and full-privileged critics. The improvements are most pronounced in tasks with partial observability, where the gap between actor and critic observations is largest, and where naive privileged information actually hurts performance due to the bias mechanism identified in the theoretical analysis.

## Core Contributions
- Theoretical analysis showing that certain privileged signals introduce bias in policy gradient estimates under the asymmetric actor-critic framework
- Identification of the conditional independence criterion for beneficial privileged signals: I(s_priv; R_future | o_actor) > 0
- Development of the IAAC framework for automatic privileged feature selection using variational mutual information estimation
- Demonstration that partial privileged information outperforms full ground-truth state in the critic for multiple locomotion tasks
- Empirical evidence that terrain friction coefficients and contact normals are beneficial privileged signals, while terrain geometry far from the robot is detrimental
- Analysis of the bias-variance tradeoff in asymmetric actor-critic: more privileged info reduces variance but may increase bias
- Practical algorithm with negligible computational overhead over standard asymmetric actor-critic

## Methodology Deep-Dive
The asymmetric actor-critic framework trains a policy π(a|o) conditioned on partial observations and a value function V(s_priv) conditioned on privileged state. The policy gradient is: ∇J = E[∇log π(a|o) * A(s_priv)], where A(s_priv) = Q(s_priv, a) - V(s_priv). The authors show that if s_priv contains features that are statistically independent of the actor's return R = Σγ^t r_t given the actor's observation o, then V(s_priv) converges to a value function that factors in information the actor cannot act upon. This creates a systematic advantage estimation error: E[A(s_priv)] ≠ 0 even for optimal policies, introducing bias in the policy gradient.

Formally, let s_priv = [s_good, s_bad] where s_good satisfies I(s_good; R | o) > 0 and s_bad satisfies I(s_bad; R | o) = 0 (or ≈ 0). Including s_bad in the critic allows V to model value fluctuations driven by s_bad that the policy cannot influence, creating phantom advantages. For example, if the critic observes distant terrain that the robot will never reach given its current observation, the critic's value varies with that terrain, but the actor cannot adjust actions accordingly, leading to noisy gradients biased toward the terrain-conditional value.

IAAC estimates I(s_i; R | o) for each privileged feature s_i using a variational lower bound: I(s_i; R | o) ≥ E[log q(R | s_i, o)] - E[log q(R | o)], where q is a learned reward predictor. In practice, two small networks are trained: q_full(R | s_i, o) and q_base(R | o), and features where q_full significantly outperforms q_base (measured by log-likelihood ratio) are selected as beneficial privileged signals. This selection process runs periodically during training (every 100K steps) and adds <5% computational overhead.

The selected privileged features are concatenated with the actor's observation to form the critic input: V(o, s_good). The actor continues to receive only deployable observations o. The standard PPO training loop proceeds with this "informed" critic, producing advantage estimates that are both lower-variance (due to privileged information) and unbiased (due to feature selection). The theoretical guarantee is that A(o, s_good) is a valid advantage function under the actor's information constraint, meaning E_π[A(o, s_good)] → 0 as π → π*.

An important practical finding is the ordering of privileged features by usefulness for locomotion tasks: (1) contact forces and friction coefficients (highest conditional MI), (2) local terrain heightmap within 0.5m of the robot, (3) body dynamics parameters (mass, inertia), (4) distant terrain heightmap (>1m away, typically detrimental), (5) other agents' states (typically detrimental in single-robot tasks). This ordering provides a practical recipe for privileged signal selection in locomotion sim-to-real.

## Key Results & Numbers
- IAAC outperforms full-privileged critic by 12–25% in return on quadruped locomotion with partial terrain observability
- IAAC outperforms no-privileged (standard symmetric) critic by 30–45% across all tested tasks
- Feature selection identifies contact forces and local terrain as top features; distant terrain ranked as detrimental in 8/10 locomotion scenarios
- Bias reduction: IAAC advantage estimates show 60% lower bias compared to full-privileged critic (measured via ground-truth advantage comparison)
- Variance of IAAC advantage estimates is 40% lower than symmetric critic (benefit of retained privileged signals)
- Computational overhead of feature selection: <5% per training run (periodic MI estimation every 100K steps)
- Bipedal walking task: IAAC achieves 95% success rate vs. 82% full-privileged vs. 88% manual feature selection

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
IAAC provides a principled framework for deciding which privileged signals to feed the Mini Cheetah's critic during asymmetric PPO training. The current approach likely provides full ground-truth state (terrain, dynamics, contacts) to the critic, which this paper demonstrates can be suboptimal. The practical ranking (contact forces > local terrain > dynamics > distant terrain) directly informs Mini Cheetah training configuration. Implementing IAAC's automatic feature selection is straightforward: add two small reward predictor networks and run selection every 100K steps.

The bias-variance analysis is particularly relevant for Mini Cheetah's curriculum learning, where terrain complexity increases during training. As terrain becomes more complex, the gap between beneficial and detrimental privileged signals widens, making principled selection increasingly important.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
IAAC is critical for Cassie's asymmetric actor-critic training at multiple hierarchy levels. The Dual Asymmetric-Context Transformer uses privileged information in its privileged context stream; IAAC's feature selection framework determines exactly which privileged features should populate this stream. The theoretical guarantee of unbiased policy gradients under informed asymmetry is especially important for bipedal locomotion, where training instability from biased gradients can cause catastrophic policy degradation.

For Cassie's hierarchical architecture, IAAC should be applied at each level independently: the Planner critic may benefit from different privileged features than the Controller critic. The automatic selection adapts to each level's action space and observation structure, providing level-specific privileged signal recommendations.

## What to Borrow / Implement
- Implement IAAC's variational mutual information feature selection for both Mini Cheetah and Cassie's asymmetric critics
- Use the privileged feature ranking (contact forces > local terrain > dynamics > distant terrain) as initial feature set for locomotion training
- Apply feature selection independently at each hierarchy level in Cassie's architecture
- Add periodic MI estimation (every 100K steps) with negligible overhead to existing training pipelines
- Compare IAAC against current full-privileged critic baselines to quantify the bias-induced performance loss

## Limitations & Open Questions
- The conditional independence criterion assumes stationary dynamics; non-stationary environments (adaptive adversaries, changing terrain) may violate this assumption
- Variational MI estimation has known looseness; the feature selection may be conservative (excluding marginally useful features) or aggressive (including marginally detrimental ones)
- The hierarchical application (different features at different levels) is theoretically motivated but not empirically validated
- The periodic re-selection (every 100K steps) may miss transient feature relevance during curriculum learning transitions
