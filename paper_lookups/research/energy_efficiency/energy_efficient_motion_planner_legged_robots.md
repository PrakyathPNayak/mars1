# Energy-Efficient Motion Planner for Legged Robots

**Authors:** Various
**Year:** 2025 | **Venue:** arXiv 2025
**Links:** https://arxiv.org/abs/2503.06050

---

## Abstract Summary
Proposes a model-based planner that achieves up to 50% lower cost of transport compared to pure RL policies while maintaining high robustness. Uses gait parameter optimization and online foot placement strategies on Unitree A1 quadruped, demonstrating that hybrid planning-RL approaches can significantly improve energy efficiency. The work bridges the gap between model-based control's efficiency and RL's robustness.

## Core Contributions
- Demonstrates that hybrid model-based planning + RL control achieves 50% lower CoT than pure RL approaches
- Develops a gait parameter optimizer that selects stride frequency, step height, and duty factor for energy efficiency
- Implements online foot placement optimization using simplified dynamics models for real-time execution
- Validates on Unitree A1 quadruped hardware with measured energy consumption improvements
- Shows that model-based planning and RL can be complementary rather than competing approaches
- Provides a modular architecture where the planner is swappable without retraining the RL controller
- Analyzes the energy breakdown across locomotion components (swing, stance, transition, stabilization)

## Methodology Deep-Dive
The architecture consists of two layers: a model-based planner that determines gait parameters and foot placements, and an RL-trained controller that tracks the planned trajectories. The planner operates at a slower frequency (~10 Hz) making high-level decisions, while the RL controller operates at the actuator frequency (500 Hz) for responsive tracking. This separation allows each component to focus on what it does best: the planner optimizes for long-horizon energy efficiency using dynamics models, while the RL controller ensures robust tracking under uncertainty and disturbances.

The gait parameter optimizer uses a simplified single rigid body dynamics (SRBD) model to evaluate the energy cost of different gait configurations. For a given velocity command and terrain estimate, the optimizer searches over stride frequency (1-4 Hz), step height (2-10 cm), and duty factor (0.4-0.8) to minimize predicted mechanical energy. The SRBD model captures the dominant energy costs: gravitational potential energy changes during CoM oscillation, kinetic energy changes during swing phases, and ground reaction force work during stance. The optimization is fast enough to run online (< 10ms per call) due to the simplified dynamics.

The foot placement strategy extends the gait parameter optimization to spatial planning. Given the selected gait parameters, the planner computes optimal foothold locations using the inverted pendulum model for balance and a terrain cost map for surface quality. Footholds are optimized to minimize the deviation from the nominal gait pattern while avoiding unfavorable terrain (slopes, edges, gaps). This optimization runs at each step cycle and produces a target foot trajectory for the RL controller to track.

The RL controller is trained in simulation using PPO with the standard locomotion reward (velocity tracking, stability, smoothness) but with an additional trajectory tracking reward that incentivizes following the planner's foot placement commands. The controller receives the planned foot trajectory as part of its observation and is trained with domain randomization to be robust to planner inaccuracies. Critically, the RL controller is trained with randomized planner inputs (varying gait parameters and foot placements) so it learns to be a general trajectory tracker rather than being coupled to a specific planning strategy. This enables the planner to be updated or replaced without retraining.

The energy analysis decomposes total locomotion energy into components: swing energy (moving legs through the air), stance energy (supporting body weight and generating propulsion), transition energy (impact losses at touchdown and liftoff), and stabilization energy (corrective actions for balance). The paper shows that pure RL policies waste significant energy on stabilization (due to conservative behavior) and transitions (due to suboptimal contact timing), while the model-based planner reduces these components by 40-60%.

## Key Results & Numbers
- 50% lower CoT than pure RL policies on Unitree A1 at nominal walking speed (0.5 m/s)
- 30% lower CoT than pure RL at trotting speed (1.0 m/s)
- Model-based planner runs at 10 Hz with < 10ms computation per planning cycle
- RL controller runs at 500 Hz for responsive trajectory tracking
- Energy breakdown: stabilization energy reduced by 60%, transition energy reduced by 40%
- Robust to ±10% mass perturbation and moderate terrain roughness
- Foot placement optimization improves stability on rough terrain by 25% vs. fixed-pattern gaits
- Planner-controller interface is modular; controller trained once works with different planners

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The hybrid planning-RL architecture could significantly improve Mini Cheetah's energy efficiency while maintaining the robustness of RL-trained controllers. The 50% CoT reduction is substantial for Mini Cheetah's battery-limited autonomous operation. The modular architecture (planner at 10 Hz, controller at 500 Hz) maps directly to Mini Cheetah's control hierarchy, where the PPO policy can serve as the RL controller while a model-based planner is added on top. The Unitree A1 validation is on a platform similar in size and weight to Mini Cheetah, making the results directly comparable. The gait parameter optimizer can be initialized with Mini Cheetah's known dynamics model, and the foot placement strategy provides a principled approach to terrain-aware locomotion.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The hybrid planner-controller architecture directly aligns with Project B's hierarchical design, specifically the Planner level. The model-based planner's role (gait parameters, foot placements) matches the Planner level's responsibility for high-level locomotion strategy. The energy decomposition analysis provides insight into where energy is wasted in bipedal locomotion, informing reward design at each hierarchy level. The foot placement optimization using the inverted pendulum model is particularly relevant for Cassie's bipedal balance, where foot placement is critical for stability (connecting to the Differentiable Capture Point module). The modular interface design supports Project B's goal of independently trainable hierarchy levels. The energy analysis methodology can be extended to bipedal locomotion to identify optimization opportunities.

## What to Borrow / Implement
- Add a model-based gait parameter optimizer on top of Mini Cheetah's PPO-trained controller
- Implement the SRBD energy model for real-time gait parameter selection on Mini Cheetah
- Use the modular planner-controller interface design for Project B's Planner ↔ Primitives interface
- Apply the energy decomposition analysis to identify energy waste in current policies
- Integrate foot placement optimization with the Differentiable Capture Point module for Cassie
- Train the RL controller with randomized planner inputs for robustness to planning errors

## Limitations & Open Questions
- SRBD model is a significant simplification; may miss energy costs from leg dynamics and motor losses
- 50% CoT reduction is measured at nominal speed; improvement may be less at higher speeds where RL excels
- The planner requires a dynamics model, which must be identified for each robot platform
- How to handle the planner's performance degradation on highly unstructured terrain where the model is inaccurate
- Integration with learning-based planners (replacing the model-based planner with a learned one) is not explored
- The modular architecture introduces communication latency between planner and controller layers
- Extension to dynamic maneuvers (jumping, bounding) where the SRBD assumption breaks down
