# Learning Deployable Locomotion Control via Differentiable Simulation

**Authors:** Schwarke, C., Klemm, V., van Duijkeren, N., Hutter, M. (ETH Zurich / NVIDIA)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv:2404.02887](https://arxiv.org/abs/2404.02887)

---

## Abstract Summary
This paper presents the first successful sim-to-real transfer of a locomotion policy trained entirely through differentiable simulation. The authors introduce a novel differentiable contact model that produces reliable, informative gradients even for the hard contact dynamics typical of legged locomotion. By leveraging these analytical gradients, they replace the zeroth-order stochastic optimization of standard reinforcement learning (e.g., PPO) with first-order gradient-based policy optimization, dramatically improving sample efficiency.

The approach is validated on a real quadruped robot, demonstrating that policies learned via short-horizon gradient-based rollouts in a differentiable simulator can transfer to hardware with minimal sim-to-real gap. The key enabler is the contact model design, which smooths the discontinuities inherent in rigid-body contacts to provide stable gradients without sacrificing physical fidelity. This work opens a new paradigm for locomotion policy training, offering an alternative to the computationally expensive, high-variance methods that dominate the field.

The authors also provide ablation studies demonstrating the importance of their contact smoothing strategy and horizon length selection. They show that naive differentiation through stiff contact dynamics leads to exploding or vanishing gradients, motivating their carefully designed relaxation scheme.

## Core Contributions
- First demonstrated sim-to-real transfer of a locomotion controller trained purely via differentiable simulation
- Novel differentiable contact model that yields reliable gradients for hard contact dynamics without sacrificing physical accuracy
- Gradient-based policy optimization framework replacing zeroth-order RL methods (PPO/SAC) with first-order analytic gradient descent
- Demonstrated significant improvement in training sample efficiency compared to model-free RL baselines
- Ablation analysis showing the critical role of contact smoothing and optimization horizon length
- Open discussion of failure modes when naive automatic differentiation is applied to stiff contact dynamics
- Practical deployment on real quadruped hardware with minimal additional domain adaptation

## Methodology Deep-Dive
The core technical innovation lies in the differentiable contact model. Standard rigid-body simulators like MuJoCo use complementarity-based contact solvers that introduce discontinuities in the dynamics—when a foot strikes the ground, forces change instantaneously, creating non-smooth gradients. The authors address this by designing a smoothed contact model where normal forces are computed via a soft penalty with carefully tuned compliance, and friction forces use a smooth Coulomb cone approximation. This ensures that the Jacobian of the simulation step with respect to policy parameters remains well-conditioned throughout contact events.

Policy optimization proceeds by unrolling the differentiable simulator for a short horizon (typically 16–64 steps), computing a scalar loss over the trajectory (matching desired velocities, penalizing energy, etc.), and backpropagating through the entire rollout to obtain gradients with respect to the neural network policy parameters. This is analogous to backpropagation through time (BPTT) in recurrent networks but applied to physics simulation. The gradient is then used in a standard optimizer (Adam) to update the policy.

A critical design choice is the optimization horizon length. Too short a horizon yields myopic policies; too long a horizon causes gradient explosion due to compounding Jacobians through stiff dynamics. The authors empirically find a sweet spot around 32–64 steps and employ gradient clipping as an additional safeguard. They also use a curriculum over terrain difficulty during training.

The simulator itself is implemented in a GPU-accelerated framework (building on NVIDIA Warp or similar), allowing thousands of parallel rollouts with full gradient tracking. The policy architecture is a standard MLP receiving proprioceptive observations (joint positions, velocities, body orientation) and outputting joint position targets fed to PD controllers—a common setup for sim-to-real locomotion.

For sim-to-real transfer, the authors apply light domain randomization over friction coefficients and mass properties, but notably require less randomization than PPO-trained policies due to the tighter optimization landscape provided by analytic gradients. The real-robot experiments use an ANYmal-class quadruped, demonstrating stable trotting gaits across flat and mildly uneven terrain.

## Key Results & Numbers
- Training converges in ~10× fewer environment steps compared to PPO baselines
- Successful zero-shot sim-to-real transfer on quadruped hardware
- Contact model gradient magnitude remains within 2 orders of magnitude across contact/no-contact transitions (vs. >6 orders for naive differentiation)
- Policy training completes in under 1 hour on a single GPU (vs. ~4–8 hours for PPO)
- Tracking error for commanded velocity: <0.1 m/s RMS on real hardware
- Energy efficiency comparable to PPO-trained policies with expert-tuned rewards
- Gradient norm stability maintained for horizons up to 64 steps

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper presents a direct alternative to the PPO-based training pipeline for Mini Cheetah locomotion. Instead of relying on stochastic policy gradient estimates that require millions of samples, differentiable simulation provides exact analytic gradients, potentially reducing training time by an order of magnitude. The contact model innovations are especially relevant since Mini Cheetah locomotion in MuJoCo involves frequent hard ground contacts during trotting and bounding gaits.

The practical implication is that the Mini Cheetah MuJoCo environment could be reimplemented in a differentiable framework (e.g., Brax, DiffTaichi, or NVIDIA Warp), enabling gradient-based training. The reduced need for domain randomization is attractive for sim-to-real transfer. However, the approach currently works best for simpler locomotion tasks (flat-ground trotting), and extending it to the diverse terrain scenarios in Project A would require careful horizon tuning and potentially hybrid approaches combining differentiable simulation with PPO for exploration.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The differentiable simulation paradigm could benefit Cassie's training pipeline, particularly at the Controller level of the 4-level hierarchy where low-level joint tracking policies are learned. Bipedal contact dynamics are even more challenging than quadruped due to the underactuated single-support phases, making the contact model contributions especially relevant. However, integrating differentiable simulation into the full hierarchical architecture (Planner→Primitives→Controller→Safety) adds significant complexity.

The gradient-based optimization could accelerate training of the Neural ODE Gait Phase module, where smooth differentiability through the ODE solver is natural. For the higher-level components (Option-Critic at Primitives, RSSM at Planner), the benefits are less direct since these operate on more abstract state spaces. The Dual Asymmetric-Context Transformer would need architectural modifications to support backpropagation through simulation dynamics.

## What to Borrow / Implement
- Implement the smoothed contact model in MuJoCo/Brax for Mini Cheetah to enable gradient-based fine-tuning of PPO-pretrained policies
- Use short-horizon differentiable rollouts as a local policy improvement step after PPO pretraining (hybrid approach)
- Adopt the gradient norm monitoring strategy to diagnose training instabilities in contact-rich scenarios
- Apply the reduced domain randomization insight: tighter optimization may require less DR for sim-to-real transfer
- Evaluate horizon length sensitivity for both quadruped and bipedal contact patterns

## Limitations & Open Questions
- Currently demonstrated only on relatively simple flat-ground locomotion; generalization to rough terrain and dynamic maneuvers is unproven
- Smoothed contact model introduces a fidelity–differentiability tradeoff: overly smooth contacts may not capture real impact dynamics
- Scalability to high-dimensional action spaces and complex policy architectures (e.g., transformers) is not explored
- The approach assumes access to a differentiable simulator, which is not yet standard for all robot platforms (e.g., Mini Cheetah's standard MuJoCo environment is not natively differentiable)
