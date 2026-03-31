# Hierarchical Reinforcement Learning for Handling Sparse Rewards in Navigation and Locomotion

**Authors:** (Artificial Intelligence Review Contributors)
**Year:** 2024 | **Venue:** Artificial Intelligence Review (Springer)
**Links:** [DOI: 10.1007/s10462-024-10794-3](https://link.springer.com/article/10.1007/s10462-024-10794-3)

---

## Abstract Summary
This paper presents a hierarchical reinforcement learning (HRL) framework designed to address the fundamental challenge of sparse rewards in long-horizon navigation and locomotion tasks. The architecture consists of two modules: a high-level selector that decomposes long-horizon tasks into a sequence of subgoals, and a low-level actuator that learns primitive motor skills to achieve each subgoal. The key insight is that hierarchical decomposition transforms a single sparse reward problem into multiple shorter-horizon, denser reward subproblems—the selector receives sparse task completion signals but provides dense subgoal rewards to the actuator, and the actuator's success/failure at achieving subgoals provides feedback to the selector.

A critical innovation is the integration of Hindsight Experience Replay (HER) at both hierarchy levels. Standard HER replaces failed goals with achieved goals in replay buffers, but the authors extend this to hierarchical settings: when the selector's proposed subgoal is not reached by the actuator, the actually-reached state is substituted as a "hindsight subgoal," providing learning signal even from failed attempts. This dramatically accelerates learning in environments where goal achievement is initially rare. Dynamic goal detection further enhances the framework by automatically identifying reachable subgoal regions from the robot's experience, enabling the selector to propose increasingly sophisticated subgoal sequences as training progresses.

Experiments on navigation (maze environments), manipulation (block stacking), and locomotion (terrain traversal) demonstrate significant improvements over flat RL baselines and prior HRL methods, with particular gains in environments where the sparse reward is achieved in fewer than 1% of random episodes.

## Core Contributions
- Two-level HRL architecture (selector + actuator) that converts sparse task rewards into hierarchically dense subgoal rewards
- Extension of Hindsight Experience Replay (HER) to hierarchical settings with hindsight subgoal relabeling at both levels
- Dynamic goal detection mechanism that automatically identifies reachable subgoal regions from experience
- Theoretical analysis of reward densification: hierarchical decomposition reduces the effective exploration horizon from O(T) to O(T/K) where K is the number of subgoals
- Demonstration across navigation, manipulation, and locomotion domains with consistent improvements
- Subgoal space learning that adapts the abstraction level to the task complexity and agent's current capability
- Curriculum effect: easy subgoals are naturally learned first, bootstrapping the learning of harder subgoals

## Methodology Deep-Dive
The HRL architecture operates on two temporal scales. The high-level selector π_high(g|s) observes the environment state s and proposes a subgoal g every H time steps (where H is the subgoal horizon, typically 10–50 steps). The low-level actuator π_low(a|s, g) executes primitive actions to reach the proposed subgoal within H steps. After H steps (or upon subgoal achievement), the selector proposes the next subgoal. This creates a semi-Markov decision process (SMDP) at the high level, where each "action" (subgoal) has a variable duration.

The subgoal space is defined as a subset of the state space—specifically, the robot's position and orientation components. The selector proposes a target position/orientation that the actuator should reach. The actuator receives a dense reward based on the negative distance to the subgoal: r_low = -||s_current - g||, providing continuous gradient signal even when the task reward is completely sparse. The selector receives a reward that combines the sparse task reward (when the final goal is achieved) and an intrinsic curiosity bonus for proposing subgoals that lead to novel states.

The hierarchical HER mechanism operates as follows. Suppose the selector proposes subgoal g_1 and the actuator reaches state s' ≠ g_1 after H steps. In standard training, this is a failure for both levels. With hierarchical HER: (1) the actuator's replay buffer stores the transition (s, g_1, a, r_low, s') and additionally stores (s, g'=s', a, r'_low, s') where the subgoal is relabeled to the achieved state s'—this provides a positive example showing the actuator can reach s' from s; (2) the selector's replay buffer stores the transition with the hindsight subgoal, providing signal about which states are reachable from which starting points. This creates a rich learning signal from every trajectory, not just successful ones.

Dynamic goal detection uses a density model (kernel density estimation or a learned VAE) over visited states to identify "frontier" regions—states that the agent has visited but not yet thoroughly explored. The selector is biased toward proposing subgoals in these frontier regions, creating an automatic curriculum from explored to unexplored state space. As the actuator's capability grows, frontier regions shift further from the starting state, naturally increasing task complexity.

For locomotion tasks specifically, the actuator learns a library of primitive gaits (walking forward, turning, crouching, stepping over obstacles) through the subgoal pursuit. Different subgoal configurations activate different primitives: a distant forward subgoal elicits walking, a rotated subgoal elicits turning, and a low-height subgoal elicits crouching. This emergent skill decomposition, arising from the subgoal structure alone, produces a diverse primitive repertoire without explicit skill labeling.

The training uses PPO at both levels with separate replay buffers and learning rates. The selector updates less frequently (every K actuator updates) to allow the actuator to adapt to new subgoal distributions before the selector changes strategy. The total training uses 2048 parallel environments and runs for 200M time steps for locomotion tasks, completing in approximately 8 hours on a single GPU.

## Key Results & Numbers
- Sparse locomotion task (reach goal 20m away over rough terrain): 78% success rate vs. 12% for flat PPO and 45% for prior HRL baselines
- Navigation (complex maze): 85% success rate vs. 5% for flat RL
- Manipulation (block stacking): 70% success rate vs. 20% for flat RL with HER
- Hierarchical HER contributes +25% success rate over HRL without HER across all tasks
- Dynamic goal detection contributes +15% success rate over HRL with fixed subgoal spaces
- Effective exploration horizon reduction: from 1000 steps to ~100 steps per subgoal (10× reduction with H=10)
- Training time: ~8 hours on single GPU for locomotion (200M steps, 2048 environments)
- Subgoal horizon ablation: H=25 optimal for locomotion; H=10 for manipulation; H=50 too long (subgoal too hard to reach)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The hierarchical reward densification approach is relevant for Mini Cheetah tasks that extend beyond simple velocity tracking—specifically, goal-conditioned navigation across rough terrain. The selector-actuator decomposition could structure the Mini Cheetah pipeline with a high-level planner proposing waypoints and a low-level locomotion policy reaching each waypoint. The hierarchical HER mechanism is valuable for terrain traversal where the final goal (reaching a distant target) is rarely achieved during early training.

However, for the core locomotion training task (learning gaits on varied terrain), the standard PPO pipeline with dense rewards is already effective, making the HRL overhead potentially unnecessary. The relevance increases if Mini Cheetah's task scope expands to include navigation and long-horizon planning beyond pure locomotion control.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper's HRL framework maps directly onto Cassie's 4-level hierarchy. The selector-actuator decomposition corresponds to the Planner-Primitives relationship, where the Planner (RSSM-based) proposes subgoals and the Primitives (Option-Critic) select which skill to execute. The hierarchical HER mechanism is immediately applicable to the Planner level, where long-horizon goal reaching under sparse rewards is a primary challenge.

The dynamic goal detection is relevant to DIAYN/DADS at the Primitives level—it provides a complementary mechanism for identifying useful subgoal regions that the diversity-driven skill discovery might miss. The subgoal horizon parameter H corresponds to the option duration in the Option-Critic framework, and the ablation results (H=25 optimal for locomotion) provide practical guidance for configuring Cassie's option execution horizon.

The emergent skill decomposition from subgoal structure aligns with the goal-conditioned primitives concept in Project B. The insight that different subgoal configurations naturally activate different locomotion primitives suggests that explicit skill labeling (as in DIAYN) may be unnecessary if the subgoal space is well-designed. This could simplify the Primitives level architecture. The hierarchical HER relabeling at both levels provides a concrete mechanism for CPTE (Cross-Primitive Transfer Enhancement) to share experience across the hierarchy.

## What to Borrow / Implement
- Implement hierarchical HER at the Planner and Primitives levels of Cassie's hierarchy for sparse reward locomotion tasks
- Use the dynamic goal detection mechanism to automatically identify useful subgoal regions for the Planner's waypoint proposals
- Adopt the H=25 subgoal horizon as a starting point for Cassie's option execution duration in the Option-Critic framework
- Integrate the subgoal-conditioned actuator training as a pre-training stage for the Controller level, producing diverse locomotion primitives
- Apply the frontier-based exploration curriculum as a complement to the Adversarial Curriculum at the Primitives level

## Limitations & Open Questions
- The two-level hierarchy may not be expressive enough for Cassie's 4-level architecture; extending to deeper hierarchies introduces additional training stability challenges
- Subgoal space design (which state components to include) requires domain knowledge and may need adaptation for bipedal dynamics
- Dynamic goal detection adds computational overhead (~20% per iteration) and the density model quality degrades in high-dimensional state spaces
- The method is evaluated primarily on relatively simple locomotion tasks; scalability to the complexity of Cassie's full bipedal locomotion remains to be demonstrated
