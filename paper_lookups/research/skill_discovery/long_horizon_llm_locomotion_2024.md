# Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with Large Language Models

**Authors:** Yutao Ouyang, Mingyuan Zhong, Jianren Wang, et al.
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2404.05291

---

## Abstract Summary
This paper presents a framework that leverages Large Language Models (LLMs) as high-level reasoning agents for quadrupedal robots performing long-horizon tasks that combine both locomotion and manipulation. The system uses an LLM to decompose complex, multi-step task specifications (given in natural language) into sequences of hybrid discrete-continuous action plans. These plans consist of locomotion commands (walk to location, turn, climb) interleaved with manipulation primitives (push, grasp, place) that are grounded in the robot's physical capabilities.

The key architectural insight is the separation of semantic reasoning (handled by the LLM) from physical execution (handled by specialized low-level RL and model-based controllers). The LLM does not directly output motor commands; instead, it translates task objectives into a sequence of parameterized skill calls. A suite of specialized agents translates these skill calls into executable robot code, handling the details of motion planning, contact management, and dynamic balance that the LLM cannot reason about.

The framework is demonstrated on a quadrupedal robot equipped with a manipulator arm, performing tasks such as navigating through a cluttered environment, opening doors, retrieving objects, and placing them at specified locations. Real-world experiments demonstrate successful long-horizon task completion with multi-step reasoning, recovery from failures, and adaptation to environmental changes through LLM re-planning.

## Core Contributions
- **LLM-based task decomposition** for quadrupedal robots that translates natural language instructions into hybrid locomotion-manipulation plans
- **Specialized execution agents** that bridge the gap between LLM-generated plans and low-level motor control
- **Hybrid discrete-continuous planning** that handles the combinatorial complexity of task sequencing while maintaining continuous physical feasibility
- **Failure recovery through re-planning** where the LLM receives execution feedback and adjusts the plan accordingly
- **Real-world deployment** on a quadruped with manipulator, demonstrating multi-step tasks in unstructured environments
- **Modular skill library** that can be extended with new locomotion and manipulation primitives without retraining the LLM

## Methodology Deep-Dive
The system architecture consists of three layers: the LLM reasoning layer, the skill translation layer, and the execution layer. The LLM reasoning layer receives a natural language task description along with a structured prompt that includes the available skill library, the robot's current state estimate, and environmental context from perception. The LLM generates a plan as a sequence of parameterized skill calls, e.g., `navigate_to(door_handle) -> grasp(door_handle) -> pull(door_handle, angle=90) -> navigate_to(target_location)`.

The skill translation layer takes each parameterized skill call and converts it into executable trajectories or controller setpoints. For locomotion skills, this involves a velocity-command interface to a pre-trained RL walking policy. The locomotion policy was trained using PPO in simulation with standard domain randomization, producing robust trotting gaits that accept target linear and angular velocities. For manipulation skills, the translation layer uses inverse kinematics and motion planning to generate arm trajectories while the locomotion controller maintains balance.

The execution layer runs the low-level controllers on the robot hardware at high frequency (typically 200-500 Hz for the locomotion controller). It monitors execution progress through proprioceptive and exteroceptive feedback, detecting when a skill has completed or failed. Failure signals (e.g., manipulation contact not established, navigation timeout, instability detected) are propagated back to the LLM, which can re-plan. The feedback loop between execution and LLM reasoning enables reactive behavior over long horizons.

A critical component is the perception pipeline that provides environmental context to the LLM. This includes object detection and localization from onboard cameras, terrain classification from depth sensors, and the robot's pose estimate from state estimation. The perception outputs are converted into a structured text description that the LLM can reason about, e.g., "Door is 2.3m ahead, handle at height 0.9m, currently closed."

The LLM prompt engineering includes few-shot examples of successful task decompositions, constraints on physical feasibility (e.g., "the robot cannot carry objects heavier than 2 kg"), and safety rules (e.g., "always ensure three feet are on the ground during manipulation"). Chain-of-thought prompting is used to encourage the LLM to reason about physical constraints before committing to a plan.

## Key Results & Numbers
- **Task completion rate:** 75-85% on multi-step tasks (navigate + manipulate + navigate) in real-world experiments
- **Average task steps:** Successfully completed tasks with 5-12 sequential steps including locomotion and manipulation
- **Re-planning success:** 60% of initially failed executions were recovered through LLM re-planning
- **Planning latency:** LLM generates plans in 2-5 seconds per re-planning cycle
- **Locomotion policy:** Base walking policy achieves 0.5 m/s forward velocity with robust balance, trained with PPO in 2 hours
- **Manipulation accuracy:** Object grasping success rate of approximately 80% when within reach
- **Long-horizon tasks:** Demonstrated 4-minute task sequences involving 8+ skill transitions

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
The LLM-based high-level planning framework could augment Mini Cheetah's autonomous exploration mode. While Project A focuses on locomotion policy learning (PPO in MuJoCo), a long-term deployment scenario would benefit from semantic task planning. For example, an LLM planner could decompose "explore the building and report obstacles" into a sequence of navigation and terrain assessment subtasks, interfacing with Mini Cheetah's trained RL locomotion policy as a low-level skill.

However, the direct applicability is limited because Project A's core challenges are in low-level policy learning, sim-to-real transfer, and terrain adaptation, none of which are addressed by the LLM layer. The paper's RL locomotion policy is relatively simple (velocity-command trotting), which is the starting point rather than the goal for Project A's research. The modular skill library concept is useful for structuring Mini Cheetah's skill repertoire (trot, bound, crawl, climb) in a way that a future high-level planner could invoke.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The high-level planner concept is relevant to Cassie's Planner level in the 4-level hierarchy, though Project B uses a learned planner (Dual Asymmetric-Context Transformer) rather than an LLM. The paper's approach of decomposing long-horizon tasks into parameterized skill sequences mirrors how Project B's Planner should select among gait primitives and set velocity targets for the Primitives level. The re-planning mechanism provides a template for how the Planner should adapt when the Primitives level reports execution failures.

The key difference is that Project B's Planner must operate at much higher frequency (planning horizon of seconds, not minutes) and must reason about continuous physical state rather than discrete semantic concepts. The LLM approach is too slow (2-5 second latency) for Cassie's real-time gait planning needs. However, the structured interface between planning and execution layers (parameterized skill calls with feedback) is a good design pattern for the Planner-to-Primitives communication protocol in Project B.

## What to Borrow / Implement
- **Structured skill interface:** define a clean API between hierarchy levels with parameterized skill calls (e.g., `trot(velocity, direction)`, `stand()`, `transition_to(gait)`) and execution feedback
- **Failure detection and re-planning:** implement monitoring at the Planner level that detects when Primitives fail to achieve commanded behaviors and triggers re-selection
- **Few-shot task decomposition patterns:** even without LLMs, the structured decomposition of complex behaviors into skill sequences can inform the training curriculum
- **Perception-to-planner interface:** the structured text description approach could inform how terrain encoder outputs are formatted for the Planner module

## Limitations & Open Questions
- **LLM latency:** 2-5 second planning latency is prohibitive for real-time locomotion control; only suitable for high-level task planning at human timescales
- **No learning from experience:** the LLM does not improve from deployment experience; it relies entirely on pre-training and prompt engineering
- **Limited physical reasoning:** LLMs have poor understanding of dynamics, contact physics, and balance, requiring all physical reasoning to be offloaded to specialized controllers
- **Manipulation focus:** the paper emphasizes locomotion + manipulation tasks, with relatively simple locomotion (flat ground trotting); the contribution to pure locomotion research is limited
