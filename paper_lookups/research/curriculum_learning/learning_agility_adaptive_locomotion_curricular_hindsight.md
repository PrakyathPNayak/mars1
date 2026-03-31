# Learning Agility and Adaptive Legged Locomotion via Curricular Hindsight Reinforcement Learning

**Authors:** Various
**Year:** 2024 | **Venue:** Nature Scientific Reports
**Links:** https://www.nature.com/articles/s41598-024-79292-4

---

## Abstract Summary
This paper combines automatic curriculum learning with hindsight experience replay (HER) to train agile, adaptive legged locomotion. The curriculum dynamically adjusts task difficulty based on the agent's current capability, while HER allows learning from failed attempts by relabeling goals post-hoc. The approach demonstrates high-speed turning, fall recovery, and robust foot placement on unstructured terrain, enabling behaviors that are difficult to achieve with standard RL training.

## Core Contributions
- Integration of automatic curriculum learning with hindsight experience replay for locomotion
- Dynamic curriculum that adjusts terrain difficulty, speed requirements, and perturbation intensity
- HER enables learning from failures: failed traversals are relabeled as successful attempts at easier goals
- Agile locomotion behaviors including high-speed turning and fall recovery
- Robust foot placement on unstructured terrain without explicit foothold planning
- Significant improvement over standard PPO and curriculum-only baselines
- Validated on quadruped and bipedal platforms in simulation with complex terrain

## Methodology Deep-Dive
The core innovation is the synergy between curriculum learning and HER. Curriculum learning alone can get stuck: if the agent fails at the current difficulty level, it receives no useful learning signal. HER alone can be inefficient: relabeling goals works well for goal-conditioned tasks but doesn't address the difficulty of the physical challenge. Together, they create a powerful feedback loop: the curriculum presents progressively harder challenges, and HER extracts learning signal even from failures at those challenges.

The automatic curriculum maintains a distribution over task parameters: terrain roughness, gap width, step height, commanded velocity, and perturbation magnitude. At each training epoch, tasks are sampled from this distribution. The curriculum frontier is adjusted based on the agent's success rate—if the agent succeeds at the current difficulty >70% of the time, difficulty increases; if success drops below 30%, difficulty decreases. This creates a dynamic difficulty adjustment that keeps the agent in its zone of proximal development.

HER is adapted for locomotion by defining implicit goals. When the agent fails to traverse a difficult terrain patch, the trajectory is relabeled: "you failed to walk at 2 m/s over 15 cm gaps, but you successfully walked at 0.5 m/s over 15 cm gaps" or "you failed to cross the gap, but you successfully maintained balance for 3 seconds." These relabeled experiences are added to the replay buffer with modified rewards, providing dense learning signal from sparse successes.

The policy architecture includes a terrain encoder (CNN on heightmap or depth), proprioceptive encoder (MLP on joint states), and a command encoder (MLP on velocity targets). These features are concatenated and processed by an actor-critic MLP trained with PPO. The HER relabeling operates on the command encoder inputs—modifying the commanded velocity or task specification to match what was actually achieved.

Training proceeds in phases: Phase 1 focuses on basic locomotion with gentle curriculum; Phase 2 introduces difficult terrain with aggressive curriculum + HER; Phase 3 adds perturbations (pushes, sudden terrain changes) for robustness. The HER contribution is most impactful in Phase 2, where many attempts at difficult terrain would otherwise produce zero learning signal.

The resulting policies exhibit agile behaviors: quick recovery from stumbles, high-speed direction changes, adaptive foot placement that avoids gaps and edges, and robustness to unexpected obstacles. These emergent behaviors arise from the combination of the challenging curriculum pushing the agent's limits and HER extracting maximum value from each experience.

## Key Results & Numbers
- Agile locomotion with high-speed turning (up to 3 rad/s yaw rate)
- Fall recovery from lateral pushes up to 50 N
- Dynamic curriculum reduces training time by 40% compared to fixed-difficulty training
- HER contribution: 25% improvement in success rate on hard terrains
- Combined curriculum + HER: 50% improvement over standard PPO on challenging terrain
- Robust foot placement on terrain with gaps up to 20 cm and steps up to 15 cm
- Validated on both quadruped (A1-like) and bipedal (Cassie-like) robot models
- Training converges in 500M steps with curriculum vs. 1B+ steps without

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to Mini Cheetah's training pipeline. The curriculum learning component can be integrated with the existing MuJoCo simulation to progressively increase terrain difficulty during PPO training. HER provides a mechanism to learn from failed training episodes that would otherwise be wasted—particularly valuable during the early stages of training on challenging terrain. The fall recovery capability addresses a practical need for real-world Mini Cheetah deployment. The dynamic difficulty adjustment can be implemented as a wrapper around the MuJoCo terrain generator, modifying terrain parameters based on training performance. The combination with domain randomization is natural: the curriculum controls task difficulty while domain randomization controls physical variability.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The curricular approach is directly applicable to Project B's adversarial curriculum component. While Project B uses adversarial terrain generation, the automatic difficulty adjustment mechanism from this paper provides a complementary (or alternative) approach to curriculum management. The HER integration is particularly valuable for Cassie's bipedal locomotion, where failures on challenging terrain are common during training. HER could be applied at the Controller level to learn from failed balance recovery attempts, and at the Primitives level to learn from failed skill transitions. The fall recovery demonstrations are directly relevant to Cassie's safety requirements. The phased training approach (basic → terrain → perturbations) maps to a structured training curriculum across Project B's hierarchy levels.

## What to Borrow / Implement
- Implement automatic curriculum learning with success-rate-based difficulty adjustment
- Integrate HER with PPO for locomotion tasks where failures are informative
- Apply the phased training approach: basic locomotion → terrain adaptation → perturbation robustness
- Use curriculum + HER at the Controller level of Project B for balance recovery learning
- Combine with adversarial curriculum: use automatic curriculum for base difficulty, adversary for targeted challenges
- Implement fall recovery training using HER to extract value from fall episodes

## Limitations & Open Questions
- HER relabeling for locomotion requires defining meaningful "easier" versions of failed tasks
- The success-rate-based curriculum can oscillate at difficulty boundaries
- Curriculum scheduling hyperparameters (thresholds, step sizes) require some tuning
- HER replay buffer grows large with relabeled trajectories—memory management needed
- How to combine HER with hierarchical policies where goals exist at multiple levels?
- The relabeling strategy for bipedal locomotion is less straightforward than for reaching/navigation tasks
- Interaction between curriculum learning and domain randomization scheduling needs further study
