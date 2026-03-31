# Option Discovery Using LLM-Guided Semantic Hierarchical Reinforcement Learning (LDSC)

**Authors:** (2025)
**Year:** 2025 | **Venue:** arXiv
**Links:** [arXiv:2503.19007](https://arxiv.org/abs/2503.19007)

---

## Abstract Summary
LDSC (LLM-Driven Subgoal and Capability discovery) presents a novel approach to hierarchical reinforcement learning that leverages Large Language Models for semantic understanding of tasks to guide option and subgoal discovery. Traditional HRL methods discover options purely from reward signals or state visitation patterns, often producing options that are difficult to interpret or that fail to align with meaningful task structure. LDSC uses an LLM to decompose tasks into semantically meaningful subgoals, which then serve as targets for option learning.

The method operates in two phases: first, the LLM analyzes the task description and environment specification to propose a set of subgoals and their logical ordering; second, an RL agent learns options (initiation sets, intra-option policies, and termination conditions) that achieve these subgoals. The LLM's semantic priors provide a warm-start for exploration, dramatically reducing the sample complexity of discovering useful temporal abstractions. The approach achieves significant gains in sample efficiency compared to pure RL option discovery methods.

For robotics applications, LDSC demonstrates that language-grounded subgoal decomposition produces options that are more transferable across task variations and more interpretable for human operators. The semantic alignment between LLM-proposed subgoals and actual task structure enables zero-shot adaptation to novel tasks through LLM re-decomposition without retraining the low-level option policies.

## Core Contributions
- Introduces LLM-guided subgoal discovery for HRL, using semantic understanding to propose meaningful temporal abstractions
- Achieves 3–5× sample efficiency improvements over pure RL-based option discovery methods (DIAYN, Option-Critic)
- Demonstrates zero-shot task adaptation by re-querying the LLM for new subgoal decompositions while reusing trained option policies
- Provides interpretable option structures aligned with human-understandable task semantics
- Shows that LLM priors significantly reduce the exploration burden in sparse-reward environments
- Proposes a principled interface between language models and reinforcement learning through the options framework
- Validates on both discrete (grid-world, Minigrid) and continuous (MuJoCo robotics) domains

## Methodology Deep-Dive
LDSC operates through a two-phase pipeline. In Phase 1 (LLM-guided Subgoal Discovery), the task description and environment specification (state space, action space, reward structure) are provided to a large language model (GPT-4 or similar) as a structured prompt. The LLM outputs a set of subgoals G = {g₁, g₂, ..., gₖ} along with a dependency graph specifying which subgoals must be achieved before others. Each subgoal includes a natural language description and a proposed state predicate (e.g., "robot is standing upright" → body_height > 0.8 AND body_tilt < 0.1). These predicates are translated into programmatic subgoal detectors using code generation capabilities of the LLM.

In Phase 2 (Option Learning), each subgoal gᵢ corresponds to an option oᵢ with: an initiation set Iᵢ (states where this option can be activated, derived from the dependency graph), an intra-option policy πᵢ(a|s) trained via PPO to reach the subgoal, and a termination condition βᵢ(s) that triggers when the subgoal predicate is satisfied or a timeout is reached. The policy-over-options πΩ(o|s) is trained to select among available options based on the current state and remaining subgoals.

A critical innovation is the **semantic reward shaping** mechanism. Rather than using only the sparse task reward, LDSC generates dense intermediate rewards for each option using the LLM. The LLM proposes reward components based on the subgoal semantics (e.g., for a "move forward" subgoal: reward = Δx_position + 0.1·upright_bonus - 0.01·energy_penalty). These shaped rewards dramatically accelerate intra-option policy training. The shaped rewards are validated against the actual task reward to prevent reward hacking.

For task transfer, when a new task is presented, only the LLM is re-queried to produce a new subgoal decomposition. If the new subgoals overlap with previously learned options (detected via semantic similarity of subgoal descriptions), the existing option policies are reused directly. New subgoals trigger training of new options, but the existing option library reduces the overall training burden.

The hierarchical execution follows the standard options framework: at each step, the policy-over-options selects an option (or continues the current one if it hasn't terminated), the intra-option policy generates actions, and the termination function determines when to return control to the policy-over-options. The LLM-derived dependency graph constrains which options can be initiated, reducing the effective option space and simplifying the policy-over-options learning.

## Key Results & Numbers
- 3–5× improvement in sample efficiency over DIAYN and vanilla Option-Critic on MuJoCo locomotion tasks
- 2–3× improvement over DIAYN in option quality (measured by subgoal achievement rate)
- Zero-shot adaptation to modified tasks achieves 60–80% of fine-tuned performance without additional training
- LLM subgoal proposals align with ground-truth task structure 75–85% of the time (measured by human evaluation)
- Semantic reward shaping reduces option training time by 50–70% compared to sparse subgoal rewards alone
- Scales to tasks requiring 8–12 subgoals without significant degradation in option quality
- GPT-4 produces more accurate subgoal decompositions than GPT-3.5 (85% vs. 65% alignment)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
For Mini Cheetah, LDSC's primary value is in task specification and curriculum design rather than direct control. An LLM could decompose complex locomotion tasks (e.g., "traverse rocky terrain to reach a goal") into semantically meaningful subgoals that inform the curriculum learning schedule. The semantic reward shaping could accelerate training of specific locomotion skills by providing dense intermediate rewards derived from language descriptions of desired behaviors.

However, the core locomotion control for Mini Cheetah (PPO with domain randomization) operates at a lower level than LDSC's subgoal abstraction. The LLM-guided approach is more beneficial for high-level task planning than for the joint-level control that is the primary focus of the Mini Cheetah project.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
LDSC is highly relevant to Cassie's Planner level, which must decompose high-level locomotion objectives into sequences of primitives. The LLM-guided subgoal discovery provides a principled method for generating the task decompositions that the GATv2 Planner must learn to produce. Instead of learning task decomposition purely from reward signals, the Planner could be initialized or guided by LLM-derived subgoal structures.

The semantic option discovery also complements the DIAYN/DADS skill discovery at the Primitives level. While DIAYN/DADS discover skills based on state-space coverage, LDSC discovers skills based on semantic task relevance—combining both approaches could produce a primitive library that is both diverse and task-aligned. The zero-shot transfer capability is valuable for adapting Cassie's behavior to new environments or tasks without full retraining.

## What to Borrow / Implement
- Use LLM-guided subgoal decomposition to initialize or warm-start Cassie's Planner level task decomposition
- Adopt semantic reward shaping to accelerate training of individual locomotion primitives with dense, meaningful rewards
- Implement the LLM-derived subgoal dependency graph to constrain and guide the policy-over-options at the Primitives level
- Combine LDSC's semantic option discovery with DIAYN/DADS's state-coverage options for a more comprehensive primitive library
- Leverage LLM task re-decomposition for zero-shot adaptation of Cassie's hierarchical controller to new locomotion tasks

## Limitations & Open Questions
- Dependency on LLM quality and prompt engineering—poor subgoal proposals can mislead the entire HRL pipeline
- Semantic reward shaping may introduce reward hacking if LLM-proposed rewards are misaligned with the actual task objective
- Scalability to very high-dimensional continuous control (e.g., full humanoid) with LLM-derived subgoals is not validated
- The approach assumes that the task can be meaningfully decomposed into discrete subgoals, which may not hold for all locomotion scenarios
