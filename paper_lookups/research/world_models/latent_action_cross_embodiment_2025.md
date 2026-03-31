# Latent Action Robot Foundation World Models for Cross-Embodiment Transfer

**Authors:** (2025)
**Year:** 2025 | **Venue:** ICLR Workshop
**Links:** [OpenReview](https://openreview.net/forum?id=vEZgPr1deb)

---

## Abstract Summary
This paper presents a foundation world model that learns unified latent action representations enabling cross-embodiment transfer of locomotion and manipulation skills. The core idea is that physically meaningful actions—such as "move forward," "turn left," or "lift"—exist in an abstract latent space that is shared across robot morphologies, even though the low-level joint commands differ dramatically between a quadruped, biped, and wheeled robot. By training a world model with a learned latent action space on data from multiple embodiments, the model discovers this shared structure and enables rapid adaptation to new robot platforms.

The architecture consists of a morphology-agnostic latent action encoder that maps robot-specific observations and actions into a unified latent action space, and a conditional world model that predicts future states given latent actions and embodiment-specific context. When deployed on a new robot, only a lightweight adapter module needs to be trained (mapping the new robot's action space to the latent space), while the world model backbone is frozen. This achieves data-efficient transfer, requiring only 10-50 real-world episodes on the target platform.

Experiments demonstrate successful cross-embodiment transfer between simulated quadrupeds (A1, Spot, ANYmal), bipeds (Cassie, humanoid), and manipulators (UR5, Franka). Locomotion skills transfer with 70-85% of single-embodiment performance using only 1% of the training data on the target platform. The latent action space exhibits interpretable structure, with principal components corresponding to physically meaningful motion primitives.

## Core Contributions
- Unified latent action space that generalizes across diverse robot morphologies (quadrupeds, bipeds, manipulators)
- Foundation world model architecture with morphology-agnostic dynamics backbone and embodiment-specific adapters
- Cross-embodiment transfer achieving 70-85% of single-embodiment performance with 1% of target data
- Interpretable latent action structure with principal components mapping to motion primitives
- Lightweight adapter training (10-50 episodes) for new robot deployment
- Data efficiency improvement: 10-50× less data required on target platform compared to training from scratch
- Demonstration across 6+ simulated robot platforms spanning locomotion and manipulation

## Methodology Deep-Dive
The foundation world model is built on top of the RSSM architecture but with a critical modification: the action input to the dynamics model is not the raw robot-specific action a_t ∈ R^{n_a} (where n_a varies per robot) but a latent action ã_t ∈ R^k drawn from a learned, fixed-dimensionality latent action space (k=16 in most experiments). This decouples the world model's dynamics core from any specific robot's action dimensionality.

The latent action encoder E_ψ maps (o_t, a_t, morphology_id) → ã_t for each embodiment. It is implemented as an embodiment-specific MLP that takes the robot's raw observation and action and produces the latent action. The encoder is trained jointly with the world model using three losses: (1) the standard RSSM reconstruction and prediction losses, (2) a cross-embodiment consistency loss that encourages similar physical behaviors (e.g., forward locomotion) to produce similar latent actions regardless of embodiment, and (3) a latent action regularization loss (KL divergence to a unit Gaussian) that prevents mode collapse and maintains a smooth latent space.

The cross-embodiment consistency loss is the key innovation. It operates on trajectory-level features: for each pair of embodiments, the loss computes the Dynamic Time Warping (DTW) distance between sequences of body-frame velocities and encourages embodiments producing similar body motions to have similar latent action sequences. Formally: L_cross = Σ_{i,j} DTW(v_body^i, v_body^j) · ||ã_seq^i - ã_seq^j||² + (1 - DTW(v_body^i, v_body^j)) · max(0, margin - ||ã_seq^i - ã_seq^j||²). This contrastive formulation clusters functionally equivalent actions while separating functionally distinct ones.

The world model backbone processes latent actions: h_t = GRU(h_{t-1}, z_{t-1}, ã_{t-1}), with posterior z_t ~ q(z_t | h_t, o_t) and prior z_t ~ p(z_t | h_t). The backbone is trained on pooled data from all embodiments, learning dynamics that are common across morphologies—such as the relationship between applied forces and resulting body velocities, or the effect of gravity and friction on contact dynamics. Embodiment-specific details (leg length, mass distribution, joint limits) are captured by the latent action encoder and a lightweight embodiment context vector c_emb that is concatenated to the GRU input.

For cross-embodiment transfer, the backbone is frozen and only a new latent action encoder E_ψ_new is trained for the target robot. This encoder is initialized from the nearest source embodiment (by morphological similarity) and fine-tuned using 10-50 episodes of data from the target platform. The fine-tuning loss is the standard RSSM loss with the backbone frozen, requiring only backpropagation through the encoder. This typically converges in 100-500 gradient steps—minutes of computation.

Analysis of the learned latent action space reveals interpretable structure: the first 3-4 principal components correspond to forward velocity, lateral velocity, yaw rate, and body height, regardless of embodiment. Higher-order components capture gait-specific details. t-SNE visualizations show clear clustering by motion type (walking, trotting, jumping) rather than by robot morphology, confirming that the latent space captures functional semantics.

## Key Results & Numbers
- Cross-embodiment transfer: 70-85% of single-embodiment performance with 1% of target data
- Latent action dimensionality: k=16 (shared across all embodiments)
- Adapter training: 10-50 episodes on target robot, 100-500 gradient steps
- Tested across 6+ platforms: A1, Spot, ANYmal (quadrupeds), Cassie, humanoid (bipeds), UR5 (manipulator)
- Data efficiency: 10-50× improvement compared to training from scratch on target platform
- First 4 latent action PCA components explain 75-85% of variance and correspond to interpretable motion primitives
- World model backbone: 200-dim GRU + 32×32 categorical latent (standard RSSM)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**

The cross-embodiment world model enables the Mini Cheetah to leverage locomotion knowledge from other quadrupeds (A1, Spot, ANYmal) through the shared latent action space. If the project has access to pre-trained world models from these platforms, the Mini Cheetah can achieve functional locomotion with minimal real-world data by training only a lightweight adapter. The 70-85% performance with 1% data makes this an attractive bootstrapping strategy.

The latent action space concept also informs the Mini Cheetah's skill representation. Instead of learning raw joint-space policies, the policy can operate in the latent action space, benefiting from the interpretable structure (velocity, yaw, height components) that naturally decomposes locomotion into meaningful sub-behaviors. This could simplify reward engineering and enable more intuitive high-level command interfaces.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**

For Cassie's hierarchical system, the latent action space concept directly maps to the interface between the Planner and Primitives levels. The Planner's world model can operate in latent action space rather than raw primitive indices, providing a smoother and more expressive planning space. The fact that latent action PCA components correspond to body velocity, yaw, and height aligns perfectly with Cassie's Planner output specification (subgoal velocities and orientations).

The cross-embodiment training approach could also benefit Cassie by incorporating locomotion knowledge from humanoid robots, which share Cassie's bipedal morphology but may have more available training data. The lightweight adapter mechanism (10-50 episodes) makes this practical even with limited Cassie hardware access.

## What to Borrow / Implement
- Implement a 16-dimensional latent action space as the interface between Cassie's Planner and Primitives
- Use the cross-embodiment consistency loss (DTW-based) to align latent actions across Mini Cheetah and similar quadrupeds
- Pre-train a world model backbone on pooled multi-robot data, then adapt to specific hardware with lightweight encoder
- Leverage the PCA-interpretable latent structure for debugging and high-level command mapping
- Apply the adapter fine-tuning protocol (10-50 episodes, frozen backbone) for rapid deployment on new hardware

## Limitations & Open Questions
- 70-85% transfer performance may be insufficient for safety-critical tasks; residual gap requires further fine-tuning
- Cross-embodiment consistency loss relies on body-frame velocity comparison, which may not capture manipulation-relevant similarities
- Latent action dimensionality k=16 may be insufficient for highly diverse behavior repertoires
- Real-robot validation not yet demonstrated; all results are in simulation
