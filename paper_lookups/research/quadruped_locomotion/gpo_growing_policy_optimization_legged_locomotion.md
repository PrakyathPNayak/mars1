---
## 📂 FOLDER: research/quadruped_locomotion/

### 📄 FILE: research/quadruped_locomotion/gpo_growing_policy_optimization_legged_locomotion.md

**Title:** GPO: Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control
**Authors:** Xiaoyu Huang, Zhongyu Li, Koushil Sreenath
**Year:** 2025
**Venue:** arXiv preprint
**arXiv / DOI:** arXiv:2601.20668

**Abstract Summary (2–3 sentences):**
GPO proposes a time-varying action space approach for PPO where the policy starts with a restricted action space and progressively expands it as training stabilizes, encouraging efficient early exploration and more stable convergence. This growing action space curriculum achieves 15–25% improvement in final performance over standard PPO for high-dimensional legged robot control including whole-body tasks. The method is demonstrated on both quadruped and biped platforms, showing faster convergence and improved stability during training.

**Core Contributions (bullet list, 4–7 items):**
- Time-varying action space curriculum that progressively expands from restricted to full action dimensions during PPO training
- Theoretical motivation based on the exploration-exploitation tradeoff in high-dimensional action spaces
- 15–25% improvement in final task performance over standard PPO on legged locomotion benchmarks
- Faster convergence to high-quality policies with reduced training instability
- Demonstrated effectiveness on both quadruped (12 DoF) and biped (10+ DoF) control tasks
- Whole-body control results showing scalability to high-dimensional action spaces (20+ DoF)
- Simple implementation requiring only a scheduling mechanism on top of standard PPO

**Methodology Deep-Dive (3–5 paragraphs):**
The key observation motivating GPO is that high-dimensional action spaces create a challenging exploration problem for policy gradient methods like PPO. When all action dimensions are active from the start of training, the policy must simultaneously discover useful actions across many dimensions, leading to noisy gradients and slow initial progress. This is particularly problematic for legged locomotion, where the full action space includes 12 or more joint position targets that must be coordinated for stable movement. GPO addresses this by initially restricting the active action space to a subset of dimensions and gradually expanding it as the policy learns to use the currently available dimensions effectively.

The action space growing schedule is defined by a curriculum that specifies which action dimensions are active at each stage of training. For legged robots, the authors propose a biologically inspired ordering: proximal joints (hip joints) are activated first, followed by middle joints (upper leg), and finally distal joints (lower leg/ankle). This ordering reflects the biomechanical principle that proximal joints contribute most to gross body movement and stability, while distal joints provide fine-grained control. During the restricted phases, the inactive action dimensions are set to default values (typically standing joint positions), and only the active dimensions receive gradient updates from PPO. This focusing of the optimization on a lower-dimensional subspace produces cleaner gradient signals and faster learning of the fundamental locomotion patterns.

The transition between stages is governed by a stability criterion rather than a fixed schedule. The policy transitions to the next stage (activating additional action dimensions) when the current policy achieves a sustained performance threshold on a stability metric—typically a combination of average reward and reward variance over recent training iterations. Specifically, the transition occurs when the rolling average reward exceeds a threshold τ_r and the rolling reward variance drops below a threshold τ_v for a specified number of consecutive PPO iterations. This adaptive criterion ensures that the policy has genuinely stabilized with the current action dimensions before expanding, preventing premature expansion that could destabilize training. The thresholds are set relative to the theoretical maximum reward achievable with the restricted action space, so they adapt to different task difficulties.

When new action dimensions are activated, the corresponding policy output heads are initialized with small random weights (near zero), so the newly activated joints produce actions close to their default positions. This warm-start initialization ensures that activating new dimensions does not cause a sudden performance drop. The value function is not reset between stages, providing continuity in the baseline estimation. The authors also introduce a brief "integration period" after each expansion where the learning rate is temporarily reduced, allowing the policy to gradually integrate the new action dimensions without overwriting the previously learned behaviors. The combined effect of careful initialization and integration periods ensures monotonic performance improvement through the curriculum.

The experimental evaluation compares GPO against standard PPO, PPO with action space noise annealing (where noise is gradually reduced rather than dimensions being added), and curriculum learning on reward complexity (where reward terms are gradually introduced). On quadruped locomotion tasks in Isaac Gym, GPO achieves 15–25% higher final reward compared to standard PPO, with the improvement being larger for more complex tasks (rough terrain locomotion, velocity tracking with rapid direction changes). The convergence speed is also improved, with GPO reaching 90% of its final performance in approximately 60% of the training steps required by standard PPO. Biped experiments on a Cassie-like model show similar improvements, and whole-body control experiments (quadruped with a manipulator arm, totaling 18+ action dimensions) demonstrate that the benefits of GPO scale with action space dimensionality.

**Key Results & Numbers:**
- 15–25% improvement in final task performance over standard PPO across legged locomotion benchmarks
- Faster convergence: 90% of final performance reached in ~60% of standard PPO's training steps
- Benefits increase with action space dimensionality (larger gains for 18+ DoF whole-body control)
- Stable training with no catastrophic performance drops during action space expansion
- Demonstrated on quadruped (12 DoF), biped (10 DoF), and whole-body (18+ DoF) platforms
- Simple implementation: ~50 lines of code modification on top of standard PPO

**Relevance to Project A (Mini Cheetah):** HIGH — Directly applicable to improving PPO training for the 12 DoF Mini Cheetah. The growing action space curriculum could accelerate convergence and improve final locomotion performance. The biologically inspired proximal-to-distal activation ordering is natural for the Mini Cheetah's hip-thigh-calf joint structure. The simplicity of implementation (minimal code changes to standard PPO) makes adoption straightforward.

**Relevance to Project B (Cassie HRL):** HIGH — Applicable to PPO training at all levels of the Cassie HRL hierarchy. The low-level controller training on Cassie's 10 DoF action space would directly benefit from the growing action space approach. Additionally, the concept of progressive complexity could be applied to the hierarchical training pipeline—training lower levels first (analogous to proximal joints) before activating higher levels (analogous to distal joints).

**What to Borrow / Implement:**
- Proximal-to-distal action space activation ordering for legged robot PPO training
- Adaptive stability-based transition criterion for action space expansion
- Warm-start initialization for newly activated action dimensions
- Integration period with reduced learning rate after action space expansion
- Application of the growing curriculum concept to hierarchical training (activating hierarchy levels progressively)

**Limitations & Open Questions:**
- The proximal-to-distal ordering may not be optimal for all robots or tasks; data-driven ordering selection could improve generality
- The stability-based transition criterion requires tuning of threshold parameters τ_r and τ_v
- The approach assumes that restricted action spaces produce useful intermediate behaviors; this may not hold for all locomotion tasks
- Interaction between the growing curriculum and domain randomization schedules has not been studied
- The method has been validated primarily in simulation; real-robot training with growing action spaces may face additional challenges
- Comparison with other curriculum learning approaches (reward curriculum, terrain curriculum) as complementary or alternative strategies is limited
---
