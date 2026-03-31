---
## 📂 FOLDER: research/sim_to_real/

### 📄 FILE: research/sim_to_real/dreureka_language_model_guided_sim_to_real.md

**Title:** DrEureka: Language Model Guided Sim-to-Real Transfer
**Authors:** Yecheng Jason Ma, William Liang, Hung-Ju Wang, Sam Wang, Yuke Zhu, Linxi Fan, Osbert Bastani, Dinesh Jayaraman
**Year:** 2024
**Venue:** RSS 2024 (Robotics: Science and Systems)
**arXiv / DOI:** arXiv:2406.01967

**Abstract Summary (2–3 sentences):**
DrEureka leverages large language models (LLMs) to automatically design reward functions and domain randomization configurations for sim-to-real transfer in robot learning. Given a task description and robot specification, the LLM generates candidate reward functions and domain randomization parameter ranges, which are evaluated in simulation and iteratively refined. The approach is demonstrated on quadruped locomotion and dexterous manipulation tasks, achieving competitive or superior performance to hand-tuned baselines while reducing human engineering effort by approximately 80%.

**Core Contributions (bullet list, 4–7 items):**
- LLM-driven automated design of reward functions for RL-based robot control
- LLM-driven automated specification of domain randomization parameter ranges
- Iterative refinement pipeline where simulation evaluation feedback guides LLM-based redesign
- Demonstration on Unitree Go1 quadruped locomotion with successful real-world deployment
- 80% reduction in human engineering time compared to manual reward and DR tuning
- Comparative analysis showing competitive or superior transfer performance relative to expert-designed configurations
- Open-source framework enabling LLM-guided sim-to-real for new tasks and robots

**Methodology Deep-Dive (3–5 paragraphs):**
DrEureka's pipeline begins with a structured prompt to a large language model (GPT-4 or similar) that describes the robot morphology, available observations and actions, simulation environment details, and the desired task behavior. The LLM is asked to generate a reward function expressed as Python code that computes a scalar reward from the environment state. The prompt includes guidelines about reward design principles (shaping vs. sparse rewards, reward scaling, common pitfalls) and examples of reward functions for related tasks. The LLM generates multiple candidate reward functions, each representing a different hypothesis about what reward structure will produce the desired behavior. These candidates are then evaluated by training RL policies (using PPO) in simulation with each reward function and measuring task-relevant metrics (e.g., forward velocity, stability, energy efficiency for locomotion).

The domain randomization component follows a similar LLM-guided approach. The LLM receives information about the simulation parameters that can be randomized (masses, frictions, motor strengths, observation noise, action delays, terrain properties) along with their physically plausible ranges. The LLM is asked to suggest randomization distributions (mean and variance for each parameter) that would produce robust policies. The key insight is that the LLM can leverage its broad training knowledge about physics and engineering to make reasonable initial estimates—for example, knowing that ground friction coefficients typically range from 0.3 to 1.0 for rubber on concrete, or that communication delays in typical robot systems are 5–20 ms. These LLM-suggested ranges serve as starting points that are refined through simulation evaluation.

The iterative refinement loop is central to DrEureka's effectiveness. After the initial LLM-generated reward and DR configurations are evaluated in simulation, the results (training curves, final performance metrics, observed behaviors, failure modes) are fed back to the LLM as text descriptions. The LLM analyzes these results and proposes modifications—adjusting reward component weights, adding or removing reward terms, widening or narrowing DR ranges. This process repeats for several iterations (typically 3–5 rounds), with each iteration producing improved configurations. The refinement is guided by both quantitative metrics (velocity tracking error, survival rate, energy consumption) and qualitative descriptions of observed behaviors (e.g., "the robot drags its rear legs" or "the robot falls when turning sharply"), allowing the LLM to diagnose and address specific failure modes.

The final sim-to-real transfer uses the best reward function and DR configuration discovered through the LLM-guided search. The policy is trained with PPO in Isaac Gym using the selected reward and DR settings, then deployed directly on real hardware. For the quadruped locomotion experiments, the Unitree Go1 robot is used with onboard computation running the policy at 50 Hz. The paper compares against several baselines: (1) hand-tuned reward and DR by experienced roboticists, (2) random search over reward and DR hyperparameters, (3) Eureka (LLM reward design without DR optimization), and (4) standard DR with the LLM reward. DrEureka's combined optimization of both reward and DR consistently outperforms optimizing either component alone.

The paper also provides analysis of the LLM's reasoning process, showing that the model generates physically grounded justifications for its design choices. For example, when designing the locomotion reward, the LLM reasons about the importance of penalizing body roll and pitch for stability, rewarding foot clearance for obstacle avoidance, and penalizing joint acceleration for smooth motion. When refining DR ranges, the LLM identifies that widening friction randomization helps with surface transfer while narrowing mass randomization prevents unrealistic training scenarios. This interpretability is a significant advantage over black-box hyperparameter optimization methods, as engineers can inspect and validate the LLM's reasoning.

**Key Results & Numbers:**
- Automated reward and DR design achieving competitive performance with hand-tuned expert baselines on Go1 locomotion
- 80% reduction in human engineering time (from ~40 hours of manual tuning to ~8 hours of LLM-guided pipeline setup)
- Successful real-world deployment on Unitree Go1 quadruped with zero-shot sim-to-real transfer
- 3–5 iteration refinement loops sufficient for convergence in most tasks
- Combined reward + DR optimization outperforms reward-only or DR-only optimization by 15–25%
- LLM-generated reward functions achieve 90%+ of hand-tuned reward performance in simulation

**Relevance to Project A (Mini Cheetah):** HIGH — Could automate the reward function design and domain randomization tuning for Mini Cheetah PPO training, significantly reducing the engineering effort required for sim-to-real transfer. The quadruped locomotion demonstration directly validates the approach for the same class of robots and tasks.

**Relevance to Project B (Cassie HRL):** MEDIUM — Reward design automation is applicable to multiple levels of the Cassie HRL hierarchy, as each level requires its own reward function. However, the hierarchical structure adds complexity that may require specialized prompting strategies. The DR optimization component is directly useful for the Cassie simulation environment.

**What to Borrow / Implement:**
- LLM-guided reward function generation pipeline with iterative refinement
- Combined reward + domain randomization optimization for sim-to-real transfer
- Structured prompting template for robot locomotion reward design
- Simulation evaluation feedback loop for guiding LLM refinements
- Interpretable reward design process enabling human validation of LLM reasoning

**Limitations & Open Questions:**
- Heavily dependent on the quality and capabilities of the underlying LLM; performance may vary across LLM versions
- The approach assumes access to a well-calibrated simulation environment; LLM cannot fix fundamental simulation model errors
- Iterative refinement requires multiple full RL training runs in simulation, which can be computationally expensive
- The LLM's physical reasoning may contain errors or biases from training data; human validation is still recommended
- Scaling to complex multi-level hierarchical reward structures (as needed for Cassie HRL) has not been demonstrated
- The 80% engineering time reduction is relative to a specific expert baseline; actual savings may vary based on engineer experience
---
