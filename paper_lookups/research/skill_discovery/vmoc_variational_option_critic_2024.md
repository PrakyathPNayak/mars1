# Learning Variational Temporal Abstraction Embeddings in Option-Induced Markov Decision Processes

**Authors:** OpenReview Authors (2024)
**Year:** 2024 | **Venue:** OpenReview / ICLR Workshop
**Links:** [OpenReview](https://openreview.net/forum?id=eM3Wzs6Unt)

---

## Abstract Summary
This paper presents VMOC (Variational Mixture of Option-Critics), a framework that combines variational inference with the Option-Critic architecture to address two persistent challenges in hierarchical reinforcement learning: unstable option learning and lack of option diversity. Standard Option-Critic suffers from gradient instability during option learning and tends to produce degenerate options that all behave similarly. VMOC addresses both issues through a principled probabilistic formulation.

VMOC treats the option selection process as latent variable inference, modeling the joint distribution over options and trajectories using a variational autoencoder-style objective. The encoder maps states to option embeddings in a learned latent space, while the decoder generates actions conditioned on these embeddings. Maximum entropy intrinsic rewards encourage the agent to discover options that maximize behavioral diversity, measured as mutual information between options and resulting state transitions. This creates a natural pressure for each option to produce distinctive behaviors.

Evaluated on MuJoCo locomotion benchmarks (HalfCheetah, Ant, Humanoid) and navigation tasks, VMOC produces more diverse, stable, and performant option sets compared to standard Option-Critic, Variational Option-Critic (VOC), and flat RL baselines. The variational formulation provides a principled way to balance option utility (task performance) with option diversity (distinct behaviors), resulting in robust hierarchical policies.

## Core Contributions
- Variational inference formulation of option learning: treats option selection as latent variable inference with an evidence lower bound (ELBO) objective
- Maximum entropy intrinsic reward for option diversity: I(ω; s'|s) encourages options to produce distinguishable state transitions
- Stable option learning through variational regularization: KL divergence between option posterior and prior prevents abrupt option changes during training
- Option embedding space: continuous latent representation of options enabling smooth interpolation between behaviors
- Integration with multiple base RL algorithms (PPO, SAC) demonstrating framework generality
- Comprehensive evaluation on MuJoCo locomotion tasks showing superior diversity and performance over Option-Critic variants
- Theoretical analysis connecting VMOC to information-theoretic skill discovery methods (DIAYN, DADS)

## Methodology Deep-Dive
VMOC's core formulation models the option-trajectory interaction as a generative process. Given state s, the agent samples an option embedding z ~ q_φ(z|s) from an encoder network (the variational posterior), then executes actions according to the option policy π_θ(a|s,z) until termination β_ψ(s,z) triggers. The generative model assumes a prior p(z) (typically standard Normal) over option embeddings, and the training objective maximizes a variational lower bound: L = E_q[R(τ)] - β·D_KL(q_φ(z|s) || p(z)), where the first term is the expected task return under the variational policy and the second term regularizes the option posterior to stay close to the prior.

The maximum entropy intrinsic reward is formulated as r_intrinsic(s, z, s') = log q_φ(z|s') - log p(z), which is high when the next state s' is informative about which option z was active (i.e., options produce distinguishable outcomes). This is equivalent to maximizing I(z; s'|s), the mutual information between options and next states conditioned on current state. The total reward combines task reward and intrinsic reward: r_total = r_task + α·r_intrinsic, where α ∈ [0.1, 1.0] controls the diversity-performance trade-off.

The option embedding space z ∈ R^k (typically k=8-16) provides a continuous parameterization of options, unlike standard Option-Critic's discrete option set. This enables smooth interpolation between options and avoids the combinatorial challenges of discrete option selection. During training, the encoder learns to map states to regions of the latent space corresponding to different behavioral modes. At deployment, the agent samples from the posterior q_φ(z|s) to select a behavioral mode appropriate for the current state.

Termination in VMOC is also conditioned on the option embedding: β_ψ(s,z) = σ(f_ψ(s,z)), where f_ψ is a neural network. The termination gradient follows the Option-Critic derivation but includes the latent variable: ∂L/∂ψ ∝ -∂β_ψ(s,z)/∂ψ · A_Ω(s,z), where A_Ω is the option advantage. A deliberation cost ξ penalizes frequent termination, encouraging temporally extended options.

The full training loop alternates between: (1) Collecting trajectories using the current variational policy, (2) Updating the critic Q(s,z) via TD learning, (3) Updating the option policy π_θ via policy gradient with combined rewards, (4) Updating the encoder q_φ via the ELBO gradient, and (5) Updating the termination function β_ψ via the termination gradient. All updates use mini-batches from a replay buffer (for SAC-based variant) or on-policy rollouts (for PPO-based variant).

The connection to DIAYN/DADS is explicit: VMOC's intrinsic reward I(z; s'|s) is identical to DADS's skill-conditioned dynamics objective when the option embedding z is treated as a skill variable. However, VMOC additionally optimizes termination conditions and the policy-over-options, providing a complete hierarchical framework rather than just skill discovery.

## Key Results & Numbers
- MuJoCo HalfCheetah: VMOC achieves average return 8500 vs 7800 for Option-Critic and 8200 for flat SAC (3-9% improvement)
- MuJoCo Ant: VMOC 5200 vs Option-Critic 4600 and flat PPO 4800 (8-13% improvement)
- Option diversity (JS divergence between option policies): VMOC 0.52 vs Option-Critic 0.12 vs AOC 0.38 (4.3x and 1.4x improvement)
- Training stability (variance of returns across seeds): VMOC σ=180 vs Option-Critic σ=420 (57% lower variance)
- Option utilization entropy: VMOC H=1.8 (near-uniform over 8 options) vs Option-Critic H=0.9 (2 options dominate)
- Learned options in Ant correspond to identifiable locomotion modes: forward, backward, turning left, turning right, diagonal gaits
- Latent space interpolation produces smooth behavioral transitions between options
- Intrinsic reward weight α=0.3 provides best diversity-performance trade-off across tasks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
VMOC's diverse option learning could benefit Project A if the Mini Cheetah needs to learn multiple locomotion modes (walk, trot, gallop, canter) within a single policy. The variational framework would encourage each option to specialize in a different gait, with smooth transitions enabled by latent space interpolation. The MuJoCo evaluation makes results directly applicable to Project A's simulation environment.

However, if Project A focuses on a single locomotion mode (e.g., trotting at various speeds), the full VMOC framework may be overengineered. The intrinsic diversity reward could distract from optimizing the primary gait quality. A simpler approach might be to use VMOC for initial gait discovery, then fine-tune the best gait with flat PPO.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
VMOC is directly applicable to Project B's Primitives level, where Option-Critic is used for locomotion primitive selection. The variational formulation addresses the option degeneracy problem (identified as a risk for Cassie) through principled regularization rather than ad-hoc diversity penalties. The maximum entropy intrinsic reward I(z; s'|s) complements DIAYN/DADS at the skill discovery level, providing a unified framework for diverse primitive learning.

The continuous option embedding space is particularly valuable for Cassie, where locomotion behaviors form a continuum (slow walk → fast walk → jog → run) rather than discrete categories. The embedding space enables smooth gait transitions—critical for bipedal stability during speed changes. The theoretical connection to DIAYN/DADS means VMOC can serve as a drop-in replacement or complement to Project B's existing skill discovery pipeline. The termination learning is especially important for Cassie, where premature or delayed primitive switching directly impacts balance and fall risk.

## What to Borrow / Implement
- Implement VMOC as an enhanced Option-Critic at Project B's Primitives level, replacing discrete option selection with continuous embeddings
- Use the maximum entropy intrinsic reward (α=0.3) alongside DIAYN/DADS objectives for comprehensive skill diversity on Cassie
- Leverage the variational regularization (KL term) to stabilize option learning during the early phases of hierarchical training
- Apply latent space interpolation for smooth gait transitions between locomotion primitives on Cassie
- Adapt the option embedding dimensionality (k=8-16) to match the expected number of distinct locomotion modes for each platform

## Limitations & Open Questions
- Continuous option embeddings make the policy-over-options harder to interpret compared to discrete option indices
- The intrinsic reward weight α requires careful tuning; too high and the agent prioritizes diversity over task performance
- Scalability to very high-dimensional observation spaces (e.g., visual input) not demonstrated; variational encoder may struggle
- No evaluation on real robotic hardware; the diversity-performance trade-off may shift when facing real-world noise and constraints
