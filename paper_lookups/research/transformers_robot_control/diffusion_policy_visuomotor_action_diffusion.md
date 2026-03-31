# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**Authors:** Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song
**Year:** 2024 | **Venue:** IJRR 2024
**Links:** https://arxiv.org/abs/2303.04137

---

## Abstract Summary
Diffusion Policy represents robot control as a conditional denoising diffusion process that progressively refines random noise into action sequences conditioned on visual observations. This approach naturally handles multimodal action distributions and high-dimensional action spaces with remarkable training stability. The method outperforms prior imitation learning approaches by 46.9% on average across 12+ robotic manipulation tasks.

## Core Contributions
- Formulates robot policy as a conditional denoising diffusion process over action sequences
- Handles multimodal action distributions naturally through the diffusion process
- Introduces action sequence prediction with receding horizon control for temporal consistency
- Demonstrates 46.9% average improvement over state-of-the-art imitation learning baselines
- Compares U-Net and Transformer-based denoising architectures for policy learning
- Achieves stable training without mode collapse, unlike GAN-based alternatives
- Validates across 12+ diverse manipulation tasks in simulation and real hardware

## Methodology Deep-Dive
The key formulation treats the policy π(a|o) as a conditional denoising diffusion model. Given observation o (visual + proprioceptive), the policy generates a sequence of future actions [a_t, a_{t+1}, ..., a_{t+H}] by iteratively denoising from Gaussian noise. The forward diffusion process adds noise to expert action sequences, and the reverse process (the policy) learns to remove noise conditioned on observations. This is trained with a simple MSE denoising objective, avoiding the adversarial training instabilities of GAN-based methods.

Action sequence prediction is critical. Rather than predicting a single action per timestep, the policy predicts a chunk of H future actions simultaneously. This provides temporal consistency—successive actions are correlated through the joint prediction—and enables look-ahead planning. At execution time, a receding horizon strategy executes the first K actions from the predicted sequence, then re-plans. The ratio H/K controls the trade-off between temporal consistency and responsiveness to new observations.

Two denoising architectures are explored: U-Net and Transformer. The U-Net architecture uses a 1D temporal convolution network with skip connections, processing the action sequence as a 1D signal. FiLM (Feature-wise Linear Modulation) conditioning injects observation information at each layer. The Transformer architecture uses cross-attention between noisy action tokens and observation tokens. Both achieve strong performance, with U-Net being slightly more compute-efficient and Transformer scaling better to longer horizons.

The observation encoder processes visual inputs through a CNN backbone (ResNet-18 or ViT) pretrained on ImageNet, with spatial softmax for compact feature extraction. Proprioceptive inputs (joint angles, velocities) are concatenated with visual features. The observation history uses a short sliding window (2-3 timesteps) to capture dynamics without overwhelming the diffusion model with temporal information.

Training uses DDPM (Denoising Diffusion Probabilistic Model) with 100 diffusion steps during training. At inference, DDIM (Denoising Diffusion Implicit Model) scheduling reduces this to 10-20 steps for faster action generation. The training objective is simply the MSE between predicted and actual noise at each diffusion step, making training extremely stable compared to adversarial or contrastive alternatives.

## Key Results & Numbers
- 46.9% average improvement over best prior imitation learning methods across 12+ tasks
- Handles multimodal action distributions without mode collapse
- U-Net architecture: inference in ~50ms for 10 DDIM steps (20 Hz control feasible)
- Transformer architecture: inference in ~80ms for 10 DDIM steps
- Action chunk size H=16 with execution horizon K=8 provides best temporal consistency
- Training converges in ~200 epochs on most tasks (stable, no mode collapse)
- Real-robot success rates: 80-95% on pick-and-place, pushing, and assembly tasks
- FiLM conditioning improves performance by 12% over concatenation-based conditioning

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Diffusion Policy's multimodal action generation could represent multiple valid gaits (trot, gallop, bound) for a given observation, with the diffusion process selecting contextually appropriate actions. The action sequence prediction provides temporal consistency important for smooth gait generation. However, the 50-80ms inference time limits control frequency to ~15-20 Hz, far below the 500 Hz PD control requirement. This could be addressed by using Diffusion Policy at a higher level (trajectory generation at 20-50 Hz) with a low-level PD controller tracking generated trajectories. The stable training is attractive compared to the reward engineering challenges in PPO.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Diffusion Policy is highly relevant at the Primitives level of Cassie's 4-level hierarchy. The multimodal action distribution naturally handles the selection between locomotion skills (walking, turning, stair climbing) without explicit mode switching. At the Planner level, diffusion could generate diverse trajectory proposals conditioned on terrain observations from the CPTE. The action sequence prediction with receding horizon control maps to trajectory chunk generation that lower levels track. The Transformer denoising architecture could integrate with the Dual Asymmetric-Context Transformer already in the hierarchy. The stable training complements PPO by providing a demonstration-based component without adversarial instabilities.

## What to Borrow / Implement
- Use Diffusion Policy at the Primitives level for multi-modal locomotion skill generation
- Implement action chunking for temporally consistent gait generation in both projects
- Adopt FiLM conditioning for observation-conditioned action generation
- Use DDIM inference scheduling to reduce latency for real-time control
- Integrate Transformer denoising with the Dual Asymmetric-Context Transformer in Project B
- Consider hierarchical diffusion: coarse trajectory from Planner → fine actions from Primitives

## Limitations & Open Questions
- Inference latency (50-80ms) is too slow for 500 Hz control; must be used at higher abstraction
- Diffusion models require large demonstration datasets; expensive for locomotion
- Action sequence prediction assumes smooth dynamics—may struggle with contact-rich locomotion
- DDIM acceleration trades off action quality for speed; optimal schedule unknown for locomotion
- Open question: How does diffusion policy interact with domain randomization for sim-to-real?
- Mode coverage vs. mode quality trade-off not well understood for locomotion primitives
- Energy consumption during inference on embedded hardware (robot onboard compute) not evaluated
