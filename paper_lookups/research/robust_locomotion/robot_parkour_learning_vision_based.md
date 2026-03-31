# Robot Parkour Learning

**Authors:** Ziwen Zhuang, Zipeng Fu, Jianren Wang, Christopher Atkeson, Sören Schwertfeger, Chelsea Finn, Hang Zhao
**Year:** 2023 | **Venue:** CoRL 2023
**Links:** https://arxiv.org/abs/2309.05665

---

## Abstract Summary
End-to-end vision-based RL enabling low-cost quadrupeds (Unitree A1/Go1) to traverse challenging parkour environments: climbing high obstacles, leaping gaps, crawling under barriers. Uses staged training: first with relaxed physics for skill discovery, then full dynamics for transfer readiness. Skills are distilled into a single vision-based policy using egocentric depth sensing.

## Core Contributions
- Demonstrates end-to-end vision-based parkour on low-cost quadruped hardware without explicit motion planning
- Introduces a staged training pipeline: relaxed physics for exploration followed by full dynamics for sim-to-real fidelity
- Distills multiple specialized skill policies into a single unified vision-conditioned policy
- Achieves zero-shot sim-to-real transfer using only egocentric depth images
- Validates on diverse parkour courses including climbing, leaping, and crawling tasks
- Shows that low-cost robots can achieve agile locomotion previously limited to expensive platforms

## Methodology Deep-Dive
The approach begins with a skill discovery phase where physics constraints are intentionally relaxed. By reducing gravity, increasing friction, or softening contact models, the RL agent explores a wider range of dynamic behaviors that would be difficult to discover under realistic physics. This is critical because hard parkour maneuvers (e.g., climbing obstacles twice the robot's height) lie in narrow regions of the policy space that standard exploration struggles to reach.

Once a diverse set of skills is discovered, the training transitions to a second stage with full-fidelity physics simulation. Here, the previously learned behaviors are refined to be physically plausible and transferable to real hardware. This staged approach decouples the exploration problem from the sim-to-real problem, addressing each challenge in sequence rather than simultaneously.

The final step is policy distillation: multiple specialist policies (one per parkour skill) are compressed into a single generalist policy conditioned on egocentric depth observations. The student policy learns to select and execute the appropriate behavior based on the visual scene, eliminating the need for an explicit high-level skill selector. The depth sensor provides a compact yet informative representation of the upcoming terrain geometry.

The sim-to-real transfer leverages domain randomization over visual and dynamics parameters, combined with the staged training that ensures policies are already compatible with realistic physics. The result is a single deployable policy that handles diverse parkour scenarios without manual switching or terrain classification.

## Key Results & Numbers
- Successfully climbs obstacles up to 2× the robot's body height
- Traverses diverse parkour courses combining climbing, leaping, and crawling in sequence
- Single unified policy replaces multiple task-specific controllers
- Zero-shot sim-to-real transfer on Unitree A1 and Go1 platforms
- Egocentric depth sensing sufficient for terrain perception and skill selection

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to the Mini Cheetah RL pipeline. The staged training approach (relaxed physics → full dynamics) provides a principled curriculum design strategy that can be integrated with the existing PPO training and domain randomization framework. The egocentric depth-based policy could extend Mini Cheetah beyond blind locomotion into vision-guided agile behaviors. The distillation from multiple skills into a single policy is relevant to building a versatile locomotion controller that handles varied terrain in MuJoCo simulation before real deployment.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
The distillation of multiple skill policies into a single vision-conditioned policy is directly relevant to the Primitives level of Cassie's hierarchy. The approach of training specialists first, then distilling, mirrors the Option-Critic / DIAYN skill discovery followed by a unified controller. The staged training concept could inform how the Adversarial Curriculum is structured—starting with easier conditions and progressively tightening constraints.

## What to Borrow / Implement
- Implement staged training in Mini Cheetah pipeline: Phase 1 with relaxed physics for skill discovery, Phase 2 with full dynamics for transfer
- Adapt egocentric depth conditioning for terrain-aware locomotion on Mini Cheetah
- Use the distillation framework to combine multiple gait policies into a single versatile controller
- Apply the relaxed-physics exploration trick to discover agile behaviors for Cassie's primitives library

## Limitations & Open Questions
- Relaxed physics stage may discover behaviors that cannot be refined into physically realizable motions
- Depth-only sensing may fail in textureless or reflective environments
- Single-policy distillation may sacrifice peak performance on individual skills for generality
- Unclear how well the approach scales to more complex morphologies (e.g., bipeds)
- Real-world robustness to sensor noise and occlusion not extensively characterized
