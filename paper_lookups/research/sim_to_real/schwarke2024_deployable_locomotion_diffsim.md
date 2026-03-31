# Learning Deployable Locomotion Control via Differentiable Simulation

**Authors:** Schwarke et al. (ETH Zurich)
**Year:** 2024 | **Venue:** arXiv
**Links:** [arXiv:2404.02887](https://arxiv.org/abs/2404.02887)

---

## Abstract Summary
Schwarke et al. tackle the core challenge preventing widespread adoption of differentiable simulation for locomotion: computing useful gradients through contact-rich dynamics. While differentiable simulators promise orders-of-magnitude faster training than model-free RL, naive gradient computation through rigid contacts yields noisy or uninformative gradients that fail to drive meaningful policy improvement. This paper introduces a differentiable contact model specifically designed for physically accurate yet smooth optimization landscapes.

The authors achieve the first successful sim-to-real transfer of a quadruped locomotion policy trained entirely within a differentiable simulation framework. Unlike prior work that demonstrated differentiable simulation only in simplified settings (2D, low-contact), this paper addresses the full complexity of 3D quadruped locomotion with frequent multi-point contact, gait transitions, and dynamic balance. The resulting policies deploy directly to hardware without fine-tuning.

The technical contribution centers on a contact model that smoothly approximates rigid-body contact forces while preserving the physical fidelity needed for sim-to-real transfer. The model uses a differentiable penetration-based normal force with learnable stiffness and a velocity-dependent friction model with smooth transitions between stick and slip regimes. This enables gradient-based optimization (Adam/L-BFGS) to converge reliably where prior differentiable approaches produced chaotic gradients.

## Core Contributions
- First successful sim-to-real transfer of a quadruped policy trained entirely in differentiable simulation
- Novel differentiable contact model balancing gradient smoothness with physical accuracy
- Demonstrates that smooth contact approximations can produce transferable policies despite the inherent approximation error
- Systematic comparison of gradient quality metrics across different contact model parameterizations
- Analysis of the trade-off between contact model smoothness (gradient quality) and physical fidelity (sim-to-real gap)
- Ablation studies isolating the impact of contact stiffness, friction smoothing, and penetration regularization

## Methodology Deep-Dive
The contact model is the paper's central technical contribution. Normal forces are computed using a smooth penalty function: F_n(δ) = k · softplus(δ/ε) where δ is the signed penetration distance, k is the contact stiffness, and ε controls the smoothness of the transition from no-contact to contact. The softplus function (log(1 + exp(x))) provides a C∞-smooth approximation to the ReLU-like contact activation, ensuring well-defined gradients at the contact boundary. The authors systematically study the effect of ε: smaller values yield more physically accurate contact but noisier gradients; larger values smooth gradients but introduce "ghosting" (forces at non-zero distance).

Friction is modeled using a regularized Coulomb model with velocity-dependent blending: F_t = -μ · F_n · v_t / (||v_t|| + ε_f), where ε_f is a small regularization constant preventing division by zero and smoothing the static-to-kinetic friction transition. This avoids the Coulomb cone's non-differentiable vertex while maintaining directionally correct friction forces. The authors find that ε_f ∈ [0.01, 0.1] provides the best balance for quadruped locomotion.

Policy optimization uses short-horizon backpropagation through time (SHBPTT) with horizon lengths of 50–200 simulation steps (0.1–0.4 seconds at 500 Hz). This limits the gradient computation to short temporal windows, avoiding vanishing gradient issues while still capturing the essential dynamics of a single stride cycle. Multiple short-horizon rollouts are aggregated for policy updates, similar to mini-batch gradient descent.

The full pipeline operates as follows: (1) Initialize policy network (MLP, 3 layers, 256 units); (2) Roll out N parallel trajectories of length H in the differentiable simulator; (3) Compute the total reward/loss; (4) Backpropagate through the simulation to obtain ∂L/∂θ; (5) Update policy parameters with Adam optimizer; (6) Repeat. The entire training loop runs on GPU using JAX-based automatic differentiation.

For deployment, the trained policy outputs target joint angles at 50 Hz, which are tracked by a PD controller at the motor level. The sim-to-real gap is minimized by matching the simulation's contact parameters to the real robot's ground interaction through system identification. Minimal domain randomization (mass ±5%, friction ±15%) is applied.

## Key Results & Numbers
- Training convergence in 5–15 minutes on single GPU vs. 4–12 hours for PPO baselines
- Gradient signal-to-noise ratio: 10–50× better than finite-difference estimates across contact events
- Sim-to-real transfer without fine-tuning; velocity tracking error <8% on real quadruped
- Contact model with softplus ε=0.01 achieves best balance of gradient quality and physical accuracy
- Short-horizon BPTT (H=100 steps) performs comparably to full-episode BPTT with 5× lower memory
- Stable locomotion at speeds of 0.3–1.2 m/s on flat terrain and mild slopes (up to 10°)
- Energy consumption comparable to PPO-trained policies (within 10%)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper provides a practical blueprint for integrating differentiable simulation into the Mini Cheetah training pipeline. The contact model parameterization (softplus normal forces, regularized Coulomb friction) can be directly implemented in MuJoCo's MJX differentiable backend or a custom JAX-based simulator. The short-horizon BPTT approach is particularly attractive for Mini Cheetah, as it avoids the memory and gradient stability challenges of full-episode backpropagation while still capturing per-stride dynamics.

The minimal domain randomization required (mass ±5%, friction ±15%) is significantly less than the typical ±30–50% used with PPO, which could reduce the conservatism of Mini Cheetah's transferred policies. The system identification procedure for matching contact parameters is directly applicable to the Mini Cheetah's Teflon/rubber foot pads on various surfaces.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
The smooth contact gradient computation is directly applicable to Cassie's differentiable capture point (DCP) module, which requires backpropagation through foot-ground contact dynamics to compute balance-maintaining actions. The softplus contact model and regularized friction formulation provide the exact gradient properties needed for the DCP to produce reliable balance signals.

The short-horizon BPTT strategy aligns well with Cassie's hierarchical structure: the low-level Controller could be optimized using differentiable simulation over single-stride windows, while higher-level modules operate at coarser timescales. The contact stiffness parameter ε can be tuned to match Cassie's specific foot-ground interaction characteristics.

## What to Borrow / Implement
- Implement the softplus contact model (F_n = k · softplus(δ/ε)) in the Mini Cheetah simulator for smooth gradients
- Adopt short-horizon BPTT (50–200 steps) to balance gradient quality with memory efficiency
- Use the friction regularization parameter ε_f ∈ [0.01, 0.1] as starting range for both platforms
- Apply system identification procedure to match simulated contact parameters to real hardware
- Integrate the gradient SNR metric as a diagnostic tool for monitoring differentiable simulation quality

## Limitations & Open Questions
- Softplus contact model introduces non-zero forces before physical contact ("ghosting"), which may cause issues for precise foot placement
- Limited to flat or mildly sloped terrain; complex terrain with edges, gaps, and steps not evaluated
- Short-horizon BPTT may miss long-horizon dependencies important for gait transitions and turning
- Contact stiffness parameter requires careful tuning per robot-terrain pair, reducing generality
