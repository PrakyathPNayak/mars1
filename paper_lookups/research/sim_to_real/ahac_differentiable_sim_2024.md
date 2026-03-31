# Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation

**Authors:** Georgiev, V., Schmid, L., Christensen, H. I., Allshire, A.
**Year:** 2024 | **Venue:** ICML 2024
**Links:** [Project Page](https://adaptive-horizon-actor-critic.github.io/)

---

## Abstract Summary
This paper introduces Adaptive Horizon Actor-Critic (AHAC), a first-order model-based reinforcement learning algorithm designed specifically for differentiable simulation environments with stiff contact dynamics. The central insight is that the optimal backpropagation horizon through a differentiable simulator is not fixed—it depends on the current contact regime, simulation stiffness, and training stage. AHAC dynamically adjusts this horizon during training to mitigate gradient bias introduced by contact-rich dynamics, achieving higher cumulative reward and superior sample efficiency compared to both model-free RL (PPO, SAC) and prior fixed-horizon differentiable simulation methods (SHAC).

The authors provide a theoretical analysis of how gradient bias accumulates over long horizons in stiff dynamical systems, showing that the bias–variance tradeoff in differentiable simulation is fundamentally different from model-free RL. Short horizons give low-bias but high-variance gradients; long horizons reduce variance but introduce systematic bias from compounding linearization errors through contacts. AHAC navigates this tradeoff by monitoring gradient statistics and adapting the horizon accordingly.

Experiments span locomotion (quadruped trotting, humanoid walking) and dexterous manipulation (in-hand rotation), demonstrating consistent improvements over baselines. The method is particularly effective in contact-rich scenarios where prior differentiable simulation approaches struggle or diverge.

## Core Contributions
- Adaptive horizon scheduling algorithm that dynamically adjusts backpropagation depth based on gradient quality metrics
- Theoretical analysis of gradient bias accumulation in differentiable simulation with stiff contacts
- Actor-critic architecture combining short-horizon analytic gradients (actor) with learned value function (critic) for long-term credit assignment
- Demonstrated superiority over SHAC (Short-Horizon Actor-Critic) and model-free baselines across locomotion and manipulation tasks
- Gradient quality monitoring using the ratio of gradient signal-to-noise as an adaptive horizon trigger
- Practical guidelines for hyperparameter selection in differentiable simulation-based RL
- Extensive ablation studies isolating the contribution of adaptive horizon vs. fixed horizon strategies

## Methodology Deep-Dive
AHAC builds on the Short-Horizon Actor-Critic (SHAC) framework, which uses differentiable simulation to compute analytic policy gradients over short rollout horizons and a learned value function to bootstrap beyond the horizon. The key issue with SHAC is that its fixed horizon is a sensitive hyperparameter: too short and the value function must carry most of the learning burden; too long and gradient bias from contact-induced stiffness corrupts the policy gradient.

AHAC addresses this with an adaptive mechanism. At each training iteration, the algorithm computes policy gradients at multiple candidate horizons (e.g., 4, 8, 16, 32 steps) and evaluates a gradient quality metric—specifically, the cosine similarity between the analytic gradient and a finite-difference estimate, or alternatively the gradient signal-to-noise ratio. When the quality metric drops below a threshold (indicating bias dominance), the horizon is shortened; when quality is high, the horizon is extended to reduce variance. This creates a dynamic schedule where the horizon typically starts short, extends during smooth locomotion phases, and contracts during contact-heavy phases.

The actor-critic architecture uses a standard MLP policy (actor) and a separate MLP value function (critic). The actor is updated via analytic gradients from differentiable rollouts plus the terminal value gradient from the critic. The critic is updated via temporal difference learning on the same rollouts. This hybrid leverages the strengths of both model-based (sample efficiency, low variance) and model-free (robustness, long-horizon credit assignment) approaches.

The differentiable simulator backbone uses GPU-accelerated parallel environments with automatic differentiation support. Contact forces are modeled using a smooth penalty-based approach (similar to Schwarke et al.) to ensure gradient flow. The authors test on NVIDIA Isaac Gym / Warp environments with up to 4096 parallel environments.

A practical algorithmic detail is the use of gradient checkpointing to manage memory consumption during long-horizon backpropagation. Without checkpointing, storing intermediate states for 64-step rollouts across 4096 environments would exceed GPU memory. The checkpointing scheme trades compute for memory by recomputing intermediate states during the backward pass.

## Key Results & Numbers
- 2–5× higher sample efficiency than PPO across locomotion tasks
- 15–40% higher final reward than fixed-horizon SHAC on contact-rich tasks
- Adaptive horizon typically oscillates between 8 and 32 steps during training, spending more time at shorter horizons during early training
- Training wall-clock time: 30–60 minutes on a single A100 GPU for locomotion tasks (vs. 4–8 hours for PPO)
- Humanoid walking: achieves stable gait in 2M environment steps (vs. 50M for PPO)
- Quadruped trotting: 0.05 m/s velocity tracking error (on par with PPO but with 10× fewer samples)
- Gradient signal-to-noise ratio drops by 3–10× when increasing horizon from 16 to 64 steps in contact-rich phases

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
AHAC directly addresses the contact-rich dynamics challenge in Mini Cheetah locomotion training. The quadruped trotting experiments in the paper are closely analogous to Mini Cheetah locomotion, making the results immediately transferable. The adaptive horizon mechanism is particularly valuable because Mini Cheetah gaits involve periodic contact-no contact transitions (swing/stance phases) that cause the optimal gradient horizon to vary within a single gait cycle.

For practical adoption, the Mini Cheetah MuJoCo environment would need to be ported to a differentiable simulator (Brax or Isaac Gym). The 10× sample efficiency improvement over PPO could significantly reduce the training iteration cycle, enabling more rapid reward function and curriculum design experiments. AHAC could serve as a drop-in replacement for PPO in the training pipeline, with the adaptive horizon handling the contact complexity automatically.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The contact-rich optimization capabilities of AHAC are relevant to Cassie's ground contact dynamics, particularly during the stance phases of bipedal walking where the compliant ankle and limited foot geometry create challenging contact conditions. At the Controller level of the hierarchy, AHAC could replace PPO for training low-level joint tracking policies with significantly better sample efficiency.

However, integrating AHAC into the full 4-level hierarchy presents challenges. The Planner level (RSSM-based) operates on abstract latent dynamics, not physical simulation, so AHAC's differentiable simulation assumption doesn't directly apply. The Primitives level (Option-Critic with DIAYN/DADS) relies on exploration mechanisms that may conflict with the exploitation-focused gradient-based optimization of AHAC. The most natural integration point is the Controller level, where AHAC's adaptive horizon could handle the bipedal contact dynamics more efficiently than PPO while the higher levels continue using model-free methods.

## What to Borrow / Implement
- Implement adaptive horizon scheduling for any differentiable simulation training pipeline—the gradient quality metric is broadly applicable
- Use the gradient signal-to-noise monitoring as a diagnostic tool even in PPO training to identify problematic reward or dynamics configurations
- Adopt the actor-critic hybrid architecture for Mini Cheetah: analytic gradients for short-term motor control, learned value function for long-term objectives
- Apply gradient checkpointing technique to manage memory when training with long rollout horizons in MuJoCo/Brax
- Benchmark AHAC against PPO on Mini Cheetah to quantify sample efficiency gains specific to the robot's morphology

## Limitations & Open Questions
- Requires a differentiable simulator, which limits applicability to environments with analytical gradient support
- The gradient quality metric adds computational overhead (~20% per iteration) for evaluating multiple candidate horizons
- Theoretical analysis assumes locally linear dynamics, which may not hold during highly dynamic maneuvers (jumping, recovery)
- Scalability to hierarchical architectures (as needed for Cassie's 4-level system) is not explored
