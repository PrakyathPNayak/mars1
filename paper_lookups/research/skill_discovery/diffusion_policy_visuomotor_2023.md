# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**Authors:** Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song
**Year:** 2023 | **Venue:** RSS 2023 / International Journal of Robotics Research (IJRR)
**Links:** [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)

---

## Abstract Summary
Diffusion Policy is a foundational work that introduces the conditional denoising diffusion process as a policy representation for visuomotor robot control. The key insight is that robot action distributions are inherently multimodal—the same observation can warrant multiple valid actions—and diffusion models are uniquely suited to represent these complex distributions. The method conditions a DDPM on visual observations (from cameras) and proprioceptive state to generate action sequences through iterative denoising, starting from pure Gaussian noise and progressively refining toward valid action trajectories.

The paper achieves a remarkable 47% average improvement in success rate over prior state-of-the-art methods across 11 diverse manipulation tasks in both simulation and real-world settings. This significant gain stems from three key properties of the diffusion formulation: the ability to represent multimodal action distributions, temporal consistency through action sequence prediction, and training stability from the well-conditioned denoising objective. The approach handles high-dimensional visual inputs gracefully through learned visual encoders.

While originally demonstrated on manipulation tasks, the Diffusion Policy framework has become the foundational architecture for subsequent locomotion-focused diffusion policy works (including DiffuseLoco and others). Its contribution to the field extends beyond the specific tasks shown, establishing diffusion as a general-purpose policy representation paradigm for robotics.

## Core Contributions
- Introduces conditional denoising diffusion as a general-purpose policy representation for visuomotor robotic control
- Achieves 47% average success rate improvement over prior SOTA across 11 manipulation benchmarks
- Demonstrates two architecture variants: CNN-based (1D temporal convolution) and Transformer-based denoising networks
- Establishes action sequence prediction with receding-horizon execution for temporal consistency
- Shows that diffusion policies naturally handle multimodal action distributions without explicit mixture modeling
- Provides comprehensive analysis of key design decisions: noise schedules, prediction horizons, observation conditioning strategies
- Validates on real-world robotic manipulation with both single-arm and bimanual setups

## Methodology Deep-Dive
The Diffusion Policy framework consists of three main components: an observation encoder, a noise prediction network, and an action sequence execution strategy. The observation encoder processes visual inputs (RGB images from one or more cameras) through a pretrained or fine-tuned CNN (ResNet-18/34) to extract visual features, which are concatenated with proprioceptive state (joint positions, gripper state) to form the conditioning vector. This conditioning is provided to the denoising network at each denoising step.

Two denoising network architectures are explored. The **CNN-based** variant uses a 1D temporal U-Net that operates over the time dimension of the action sequence. The input is a noisy action sequence of shape (T_a × D_a) where T_a is the action horizon and D_a is the action dimension. The U-Net processes this through downsampling and upsampling blocks with skip connections, with the observation conditioning injected via FiLM layers. The **Transformer-based** variant treats each timestep's action as a token, adds sinusoidal positional encoding for both the time dimension and the diffusion step, and uses cross-attention to incorporate observation conditioning. The CNN variant is found to be more efficient for shorter horizons while the Transformer variant scales better to longer horizons.

The training objective is the standard DDPM loss. Given a ground-truth action sequence a₀ from the demonstration dataset, noise εₜ is added at a randomly sampled diffusion timestep t to produce aₜ = √(ᾱₜ)a₀ + √(1-ᾱₜ)εₜ. The network is trained to predict εₜ given aₜ, t, and the observation conditioning o. The loss is L = E[‖εθ(aₜ, t, o) - εₜ‖²]. The cosine noise schedule is used, which provides more uniform signal-to-noise ratios across diffusion steps compared to linear schedules.

The **receding-horizon action execution** strategy is critical for temporal consistency. The model predicts an action sequence of T_a steps but only executes the first T_e steps (T_e < T_a) before re-planning. This overlapping execution provides smooth action transitions and allows the policy to be reactive to new observations while maintaining temporal coherence. Typical values are T_a = 16 and T_e = 8 for manipulation tasks.

For efficient inference, DDIM sampling reduces the required denoising steps from 100 to 10–16 while maintaining action quality. The authors also explore exponential moving average (EMA) of model weights during training, which significantly stabilizes generation quality.

## Key Results & Numbers
- 47% average improvement in success rate over prior SOTA across 11 tasks (simulation + real)
- CNN variant achieves 10 Hz control with 100 DDPM steps or 50 Hz with 10 DDIM steps
- Transformer variant achieves comparable quality at ~5 Hz with 100 steps
- Real-world success rates: 90%+ on multiple manipulation tasks with 50 demonstrations
- Outperforms Implicit Behavior Cloning (IBC), BeT, and LSTM-GMM baselines consistently
- Multimodality handling: maintains separate action modes with <5% mode mixing in bimodal tasks
- Action horizon ablation: T_a=16, T_e=8 optimal across most tasks

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
As the foundational work for diffusion-based robot control, this paper provides the core architectural and training principles needed to implement diffusion policies for Mini Cheetah. While the original experiments focus on manipulation, the framework is directly applicable to locomotion by replacing visual encoders with proprioceptive encoders and adjusting action horizons for the higher control frequencies required (50–100 Hz for locomotion vs. 10 Hz for manipulation). The CNN-based temporal U-Net architecture is particularly suitable for the Mini Cheetah's 12 DoF action space.

The visuomotor capability is also relevant if the Mini Cheetah project incorporates camera-based terrain perception—the same diffusion framework can condition on visual terrain features to generate terrain-adaptive gaits. The demonstrated training stability and multimodal action handling address key challenges in locomotion policy learning.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
For Cassie's hierarchical architecture, Diffusion Policy provides the theoretical and practical foundation for implementing diffusion-based components at multiple levels. At the **Primitives level**, the multimodal action distribution capability directly supports generating diverse locomotion skills from a single model. At the **Controller level**, the receding-horizon execution strategy provides temporal consistency crucial for bipedal balance.

The visuomotor integration is relevant if Cassie uses terrain-aware planning—visual observations of upcoming terrain could condition the Planner level's diffusion model to generate appropriate locomotion strategies. The cross-attention conditioning mechanism provides a natural interface between hierarchical levels, where higher-level commands serve as conditioning for lower-level diffusion policies.

## What to Borrow / Implement
- Implement the 1D temporal U-Net with FiLM conditioning as the base diffusion architecture for locomotion policies
- Adopt receding-horizon action execution (predict T_a=16, execute T_e=4–8) adapted for locomotion control frequencies
- Use DDIM sampling with 8–16 steps for real-time inference on robot hardware
- Apply the cosine noise schedule for stable training on locomotion action sequences
- Leverage EMA weight averaging during training for improved generation quality and stability

## Limitations & Open Questions
- Original experiments are on manipulation, not locomotion—adaptation to high-frequency locomotion control requires validation
- Relies on demonstration data (behavior cloning paradigm)—integration with online RL for locomotion improvement is not addressed
- Computational cost of diffusion inference, even with DDIM, may be prohibitive for the highest control frequencies (>100 Hz)
- No explicit mechanism for safety constraints or joint limit enforcement during the denoising process
