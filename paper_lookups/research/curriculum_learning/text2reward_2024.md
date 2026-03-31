# Text2Reward: Reward Shaping with Language Models for Reinforcement Learning

**Authors:** Xie et al.
**Year:** 2024 | **Venue:** ICLR
**Links:** https://arxiv.org/abs/2309.11489

---

## Abstract Summary
Text2Reward presents a framework for automatically generating dense reward functions from natural language task descriptions using large language models (LLMs). Instead of hand-engineering complex reward functions—a notoriously difficult and time-consuming process in RL—Text2Reward takes a text description of the desired behavior (e.g., "make the quadruped walk forward smoothly at 1 m/s") and generates executable reward code that can be directly used for policy training. The system supports iterative refinement through human feedback, allowing users to critique the generated reward and request modifications in natural language.

The framework operates in two stages: (1) an LLM generates an initial reward function as executable Python code given the task description and environment API documentation, and (2) a human-in-the-loop refinement process where the user observes the trained policy's behavior, provides natural language feedback (e.g., "the robot sways too much laterally"), and the LLM modifies the reward accordingly. This iterative process typically converges to a high-quality reward function in 2–4 refinement rounds.

Text2Reward is evaluated across diverse tasks including manipulation (ManiSkill2 benchmark), locomotion (quadruped and bipedal), and navigation. On locomotion tasks, the LLM-generated rewards achieve competitive or superior performance compared to expert-designed rewards, while requiring orders of magnitude less human engineering time. The framework demonstrates that LLMs can effectively serve as reward function designers when provided with environment specifications and iterative behavioral feedback.

## Core Contributions
- A general framework for LLM-based dense reward function generation from natural language task descriptions, applicable to both manipulation and locomotion domains
- Iterative human-in-the-loop refinement protocol where behavioral feedback is translated to reward function modifications via the LLM
- Environment API specification format that provides the LLM with sufficient context to generate executable reward code (observation space, action space, available state variables)
- Demonstration that LLM-generated rewards achieve 85–100% of expert-designed reward performance across 17 tasks in ManiSkill2 and locomotion environments
- Analysis of reward function quality showing LLM-generated rewards are more interpretable and modular than human-engineered alternatives
- Open-source implementation with templates for common RL environments (Isaac Gym, MuJoCo, ManiSkill2)
- Rapid convergence: LLM-generated rewards typically match expert rewards within 2–4 refinement iterations, reducing reward engineering time from days to hours

## Methodology Deep-Dive
Text2Reward's core pipeline begins with a structured prompt to the LLM (GPT-4) containing: (1) the environment API documentation (observation dictionary keys, action space dimensions, available sensor readings), (2) the task description in natural language, (3) a code template specifying the reward function signature and expected return format, and (4) few-shot examples of reward functions for related tasks. The LLM generates Python code implementing the reward function, including numerical constants and weighting terms.

The generated reward function is then used to train a policy via PPO in the target environment. After training (typically 1000–2000 iterations), the user evaluates the policy's behavior—either through visual inspection or quantitative metrics—and provides natural language feedback. Examples of feedback include: "the robot lifts its feet too high during walking," "add a penalty for lateral body sway," or "increase the importance of maintaining forward velocity." This feedback, along with the current reward code and training metrics, is provided to the LLM, which generates a modified reward function.

The reward function code is structured as a modular composition of reward components, each corresponding to a behavioral objective. For locomotion, typical components include: velocity tracking (exponential kernel), orientation maintenance (quaternion distance), foot contact pattern (phase-conditioned contact reward), energy penalty (joint power), smoothness (action rate penalty), and body height (desired height tracking). Each component has a weight parameter that the LLM adjusts based on feedback.

A critical design choice is providing the LLM with the environment's full state information (not just the policy's observation space). This allows the reward function to reference privileged information (ground truth velocity, contact forces, terrain height) that may not be available to the policy but is available in simulation for reward computation. This is standard practice in sim-to-real RL where the reward function has access to simulator state.

The iterative refinement process typically follows a pattern: Round 1 generates a functional but suboptimal reward; Round 2 addresses the most obvious behavioral issues; Round 3 fine-tunes weights and adds subtle terms; Round 4 (if needed) handles edge cases. The authors find diminishing returns beyond 4 rounds, suggesting the LLM's reward design capability saturates.

## Key Results & Numbers
- Locomotion tasks: LLM-generated rewards achieve 90–100% of expert reward performance on quadruped walking and trotting tasks
- ManiSkill2 benchmark: 85% average success rate across 17 manipulation tasks, compared to 92% with expert-designed rewards
- Reward engineering time: reduced from ~2–5 days (expert manual design) to 2–4 hours (Text2Reward iterative refinement)
- Convergence: 2–4 refinement rounds sufficient for competitive performance in 15/17 tasks
- Interpretability: users rated LLM-generated rewards as more interpretable than expert rewards in 70% of blind evaluations
- Quadruped velocity tracking: 0.04 m/s RMSE with LLM reward vs. 0.03 m/s with expert reward
- Cost: ~$2–5 in GPT-4 API costs per task for the full refinement process

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Text2Reward can significantly accelerate reward function design for Mini Cheetah's locomotion tasks. Rather than spending days manually crafting and tuning reward weights for velocity tracking, gait quality, energy efficiency, and robustness, Text2Reward enables rapid prototyping of reward functions from natural language specifications. The iterative refinement process aligns well with the typical sim-to-real workflow where reward functions are repeatedly adjusted based on simulated and real-world behavior.

For Mini Cheetah specifically, Text2Reward could generate initial reward functions for multiple locomotion objectives (walking, trotting, turning, obstacle traversal) in hours rather than days. The modular reward code structure also makes it easier to compose rewards from multiple objectives and adjust weights systematically. The framework's support for Isaac Gym and MuJoCo environments means direct integration with Mini Cheetah's training pipeline.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
Text2Reward is highly relevant to Cassie's hierarchical reward specification challenge. Each level of the 4-level hierarchy (Planner, Primitives, Controller, Safety) requires its own reward function with different objectives. Text2Reward could generate reward functions for each level from natural language descriptions: "the planner should select waypoints that avoid obstacles," "the controller should track joint targets while minimizing energy." The iterative refinement process is especially valuable for the hierarchical setting where reward interactions across levels are complex and difficult to manually design.

The LLM-driven approach could also help specify rewards for DIAYN/DADS skill discovery at the Primitives level, generating diversity-encouraging reward terms that are difficult to engineer manually. The interpretable, modular reward code structure aids debugging of reward interactions across hierarchy levels.

## What to Borrow / Implement
- Use Text2Reward for rapid prototyping of Mini Cheetah's reward functions, generating initial rewards from task descriptions and refining via behavioral feedback
- Apply the iterative refinement protocol to systematically improve reward functions for both Mini Cheetah and Cassie
- Adopt the modular reward code structure for organizing Mini Cheetah's multi-objective reward (velocity, energy, gait quality, stability)
- Use Text2Reward to generate level-specific reward functions for Cassie's 4-level hierarchy from natural language specifications
- Leverage Text2Reward as a starting point for reward engineering, then fine-tune critical terms manually for sim-to-real transfer

## Limitations & Open Questions
- LLM-generated rewards may miss subtle physical constraints (actuator limits, thermal considerations) that domain experts encode implicitly, potentially leading to sim-to-real transfer issues
- The quality of generated rewards depends heavily on the environment API documentation provided; incomplete or ambiguous API specs lead to poor rewards
- Iterative refinement requires the user to identify behavioral issues, which can be challenging for subtle locomotion quality differences (e.g., slight foot scuffing)
- LLM reward generation is non-deterministic; the same prompt can produce different reward functions, introducing variability in training outcomes
