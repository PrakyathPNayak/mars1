# DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets

**Authors:** Huang et al. (HybridRobotics Lab, UC Berkeley)
**Year:** 2024 | **Venue:** arXiv
**Links:** [GitHub – HybridRobotics/DiffuseLoco](https://github.com/HybridRobotics/DiffuseLoco)

---

## Abstract Summary
DiffuseLoco introduces a diffusion-based policy framework for real-time legged locomotion control that learns directly from offline datasets. Rather than relying on online reinforcement learning with reward engineering, the method trains a conditional denoising diffusion model to generate multi-step action sequences conditioned on proprioceptive observations and velocity commands. The key innovation is achieving real-time inference speeds sufficient for deployment on physical robots while retaining the expressiveness of diffusion models.

The framework demonstrates multimodal gait generation—producing distinct locomotion patterns such as trot, pace, and bound—from a single unified controller without explicit gait scheduling. This is accomplished by learning from diverse offline demonstration data that captures multiple gait modalities. The approach was validated on the Unitree CyberDog quadruped robot, showing smooth transitions between gaits and robust locomotion across varied terrains. The real-time capability is achieved through an efficient denoising schedule that reduces the number of diffusion steps required at inference time.

The work positions diffusion policies as a viable alternative to PPO-based methods for locomotion, offering advantages in handling multimodal action distributions and leveraging offline datasets without the instabilities of online training.

## Core Contributions
- Demonstrates real-time diffusion policy inference for legged locomotion control on physical hardware (CyberDog quadruped)
- Achieves multimodal gait generation (trot, pace, bound) from a single diffusion-based controller without explicit mode switching
- Introduces an efficient conditional denoising pipeline that generates multi-step action sequences within the control loop timing constraints (~50 Hz)
- Shows that offline dataset training with diffusion models can match or exceed online RL methods for locomotion quality
- Provides a framework for leveraging diverse demonstration data to capture the full spectrum of locomotion behaviors
- Validates sim-to-real transfer of diffusion policies on real quadruped hardware with terrain robustness

## Methodology Deep-Dive
The core architecture employs a Denoising Diffusion Probabilistic Model (DDPM) adapted for continuous control. The model takes as input the current proprioceptive state (joint positions, velocities, body orientation from IMU, and velocity commands) and generates a sequence of future actions (target joint positions) over a prediction horizon of H steps. The conditioning mechanism uses a feature encoder that maps observations into a latent representation, which is then injected into the denoising U-Net or transformer backbone via cross-attention or FiLM conditioning layers.

Training follows the standard diffusion objective: Gaussian noise is added to ground-truth action sequences from the offline dataset at varying noise levels, and the model learns to predict the noise (or directly the denoised actions via v-prediction). The offline dataset is collected from multiple specialist policies (each trained for a specific gait) or from motion capture data, ensuring coverage of diverse locomotion modes. The loss is the standard MSE between predicted and actual noise across all diffusion timesteps.

For real-time inference, DiffuseLoco employs DDIM (Denoising Diffusion Implicit Models) sampling with a drastically reduced number of denoising steps—typically 4–8 steps instead of the standard 50–1000. This is combined with action chunking, where the model predicts a window of future actions and only the first few are executed before re-planning, providing temporal consistency while allowing reactivity to state changes. The chunked execution also amortizes the inference cost across multiple control steps.

The multimodal gait emergence is a natural property of the diffusion framework: unlike unimodal policies (e.g., Gaussian policies in PPO), the diffusion model can represent arbitrary action distributions, allowing it to capture distinct gait modes present in the training data. At inference time, the stochastic sampling process can produce different gaits depending on the initial noise seed and conditioning, with velocity commands biasing the distribution toward appropriate gaits.

Domain randomization is applied during data collection (varying friction, mass, terrain) to ensure the learned diffusion policy generalizes across sim-to-real gaps. The real-time deployment uses ONNX-optimized inference on the robot's onboard compute.

## Key Results & Numbers
- Real-time inference at ~50 Hz control frequency on onboard compute with 4–8 DDIM denoising steps
- Successfully generates trot, pace, and bound gaits from a single controller
- Demonstrated on Unitree CyberDog quadruped in real-world experiments
- Smooth gait transitions triggered by velocity command changes without explicit scheduling
- Comparable or improved locomotion quality versus PPO-trained specialist policies
- Robust to moderate terrain variations (slopes, soft ground) in real-world deployment

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
DiffuseLoco is directly applicable to the Mini Cheetah project as it offers a fundamentally different approach to quadruped locomotion control. Instead of PPO with reward engineering, diffusion policies can leverage offline datasets (potentially from existing Mini Cheetah motion data or MuJoCo-trained specialists) to produce multimodal gaits. The real-time inference capability at 50 Hz is compatible with Mini Cheetah's 12 DoF control requirements. The multimodal gait generation aligns with the project's goal of achieving diverse locomotion behaviors (walk, trot, bound) without training separate policies.

The approach could complement the existing domain randomization and curriculum learning pipeline by providing a more expressive policy class that naturally handles the multimodality of locomotion. The sim-to-real transfer methodology demonstrated on CyberDog provides a roadmap for deploying diffusion policies on Mini Cheetah hardware.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
For Cassie's hierarchical architecture, DiffuseLoco's multimodal action generation is highly relevant to the Primitives level. The diffusion model's ability to generate diverse action sequences from a single model maps directly onto the need for multiple locomotion primitives (walking, turning, standing, recovery). Rather than using DIAYN/DADS for skill discovery followed by Option-Critic for selection, a diffusion-based primitive generator could provide richer, more continuous skill representations.

The conditional generation framework also aligns with the Planner→Primitives interface: high-level commands from the GATv2 planner could serve as conditioning signals for the diffusion model, which then generates appropriate primitive action sequences. The temporal consistency of action chunking is beneficial for Cassie's balance-critical control.

## What to Borrow / Implement
- Implement DDIM-accelerated diffusion inference pipeline for real-time locomotion control at 50 Hz
- Adopt action chunking strategy (predict H-step windows, execute first K steps) for temporal consistency in Mini Cheetah
- Use multimodal offline dataset collection from specialist policies as training data for diffusion model
- Apply the conditioning architecture (FiLM/cross-attention) for velocity-command-conditioned gait generation
- Explore diffusion-based primitive generation as alternative to DIAYN/DADS at Cassie's Primitives level

## Limitations & Open Questions
- Offline data quality and coverage directly limit the diversity and quality of generated behaviors—poor demonstrations lead to poor policies
- Real-time inference with reduced DDIM steps may sacrifice action quality compared to full diffusion sampling
- Scalability to more complex behaviors beyond basic locomotion gaits is unclear
- No direct comparison with state-of-the-art online RL methods (PPO with curriculum) on standardized benchmarks
