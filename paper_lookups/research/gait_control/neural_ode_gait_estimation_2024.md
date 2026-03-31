# IMU-Based Real-Time Estimation of Gait Phase Using Multi-Resolution Convolutional Neural Networks

**Authors:** (MDPI Sensors 2024)
**Year:** 2024 | **Venue:** MDPI Sensors
**Links:** https://www.mdpi.com/1424-8220/24/8/2390

---

## Abstract Summary
This paper presents a multi-resolution convolutional neural network (CNN) architecture for real-time gait phase estimation using inertial measurement unit (IMU) data. Gait phase — the continuous scalar representing progress through the gait cycle from 0 (heel strike) to 1 (next heel strike) — is critical for wearable robots, exoskeletons, and legged robot controllers. The proposed system achieves sub-2% mean phase error while running in real-time on embedded hardware, enabling deployment on resource-constrained robotic platforms.

The multi-resolution design processes IMU signals (accelerometer + gyroscope, 6-axis) at multiple temporal scales simultaneously. Parallel convolutional branches with different kernel sizes capture both fine-grained gait events (heel strike, toe-off transitions that occur over ~20ms) and coarse temporal patterns (overall gait cycle periodicity at ~500ms-1200ms). The outputs of these branches are fused through learned attention weights, allowing the network to dynamically weight fine vs. coarse temporal features depending on the current gait phase and walking conditions.

The system is validated across multiple walking speeds (0.5-2.0 m/s), terrain types (flat, incline, decline, stairs), and subject demographics. It demonstrates robust generalization with minimal fine-tuning, achieving state-of-the-art continuous gait phase tracking that outperforms both traditional threshold-based methods and prior deep learning approaches. The architecture is sufficiently lightweight for deployment on ARM Cortex-M microcontrollers, making it suitable for embedded robotic applications.

## Core Contributions
- **Multi-resolution CNN architecture:** Designed parallel convolutional branches with kernel sizes spanning 5ms to 200ms temporal windows, capturing both transient gait events and periodic patterns simultaneously
- **Attention-based multi-scale fusion:** Learned to dynamically weight different temporal resolutions depending on gait phase and walking conditions, improving accuracy at critical gait transitions
- **Sub-2% mean phase error:** Achieved <2% mean absolute phase error across diverse conditions, outperforming prior methods by 30-40%
- **Real-time embedded deployment:** Demonstrated inference at 200Hz on ARM Cortex-M7, with total latency <5ms including preprocessing
- **Cross-speed generalization:** Showed robust performance across walking speeds from 0.5 m/s to 2.0 m/s without speed-specific models
- **Minimal sensor requirements:** Used a single shank-mounted 6-axis IMU, making the system practical for deployment

## Methodology Deep-Dive
The input to the network is a sliding window of 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope) sampled at 200Hz. The window length is 100 samples (500ms), chosen to capture slightly more than one full gait cycle at fast walking speeds. Preprocessing includes: mean subtraction per axis, Butterworth low-pass filtering at 25Hz to remove high-frequency noise, and normalization to unit variance per channel.

The multi-resolution encoder consists of 4 parallel convolutional branches, each with different kernel sizes: Branch 1 (kernel=3, capturing 15ms features — individual footfall impacts), Branch 2 (kernel=11, capturing 55ms features — heel strike to foot flat transition), Branch 3 (kernel=25, capturing 125ms features — single stance phase), Branch 4 (kernel=41, capturing 205ms features — approximate half gait cycle). Each branch applies 2 convolutional layers with batch normalization and ReLU activation, producing feature maps at different temporal resolutions. Zero-padding ensures all branches produce outputs of the same temporal dimension.

The multi-scale fusion module uses channel-wise attention to combine the 4 branch outputs. Each branch's feature map is globally average-pooled to produce a scalar importance weight, passed through a shared FC layer with softmax to produce normalized attention weights. The weighted sum of branch outputs produces a fused feature representation. Importantly, these attention weights are input-dependent — during fast walking, the network upweights fine-resolution branches; during slow walking, coarse-resolution branches receive more attention.

The gait phase prediction head is a 2-layer MLP that outputs two values: `sin(2πφ)` and `cos(2πφ)`, where `φ ∈ [0, 1)` is the gait phase. Using sinusoidal encoding avoids the phase wrapping discontinuity at φ=0/1, which would cause issues with standard regression losses. The phase is recovered as `φ = atan2(sin, cos) / (2π)`. The loss function is the angular mean squared error in the sinusoidal representation, which naturally handles the circular topology of gait phase.

Training uses data from 15 subjects walking on a treadmill with synchronized force plates (providing ground-truth gait events) and IMU recordings. Data augmentation includes: temporal jittering (±5% speed variation), magnitude scaling (±10% simulating different IMU mounting orientations), and Gaussian noise injection (σ=0.05g for accelerometer, σ=2°/s for gyroscope). Leave-one-subject-out cross-validation is used for evaluation.

## Key Results & Numbers
- Mean absolute phase error: 1.7% ± 0.4% across all subjects and conditions (1 full gait cycle = 100%)
- Heel strike detection accuracy: 98.3% with mean timing error of 12ms
- Toe-off detection accuracy: 96.8% with mean timing error of 18ms
- Cross-speed performance: 1.5% error at 1.0 m/s, 2.1% error at 0.5 m/s, 1.9% error at 2.0 m/s
- Stair climbing: 2.8% error (degraded but still functional)
- Inference time: 4.2ms on ARM Cortex-M7, 0.8ms on NVIDIA Jetson Nano, 0.1ms on desktop GPU
- Model size: 45K parameters, 180KB model file — suitable for microcontroller deployment
- Comparison: 30% lower phase error than LSTM baseline (2.4%), 40% lower than adaptive oscillator method (2.8%)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Low**
Gait phase estimation is less critical for the Mini Cheetah quadruped, which typically uses RL-learned gaits without explicit phase tracking. Quadruped locomotion can emerge from reward shaping without decomposing the gait cycle into phases. However, if explicit gait phase is needed for curriculum learning or reward shaping (e.g., rewarding foot contact at specific phases), the multi-resolution approach could be adapted for the Mini Cheetah's IMU data. This remains a secondary consideration.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
This paper provides critical architectural insights for the Neural ODE Gait Phase component in Project B. While the paper uses CNNs rather than Neural ODEs, the multi-resolution approach to capturing both fine-grained gait events and coarse periodic patterns is directly transferable to the ODE dynamics function design. Specific connections include: (1) the sinusoidal phase representation `(sin(2πφ), cos(2πφ))` should be adopted in Project B's gait phase module to avoid phase wrapping issues — the Neural ODE should output in this sinusoidal space; (2) the multi-resolution concept can inform the ODE dynamics function architecture, where different components of the ODE handle fast vs. slow dynamics; (3) the 200Hz estimation rate provides a target for Project B's gait phase inference frequency; (4) the cross-speed generalization validates that a single model can handle variable walking speeds, supporting the design choice of a unified gait phase module across Cassie's speed range; (5) the attention-based fusion between temporal scales could be incorporated into the ODE function's internal architecture.

## What to Borrow / Implement
- **Sinusoidal phase representation:** Use `(sin(2πφ), cos(2πφ))` output encoding in the Neural ODE Gait Phase module to avoid phase wrapping discontinuities; train with angular MSE loss
- **Multi-resolution concept for ODE dynamics:** Design the ODE function `f_θ` with internal parallel pathways processing state at different time-constant scales — analogous to the multi-resolution CNN branches
- **Target inference latency:** Use the paper's 0.8ms (Jetson Nano) as a latency budget target for the Neural ODE gait phase inference on Cassie's onboard compute
- **Data augmentation strategy:** Apply temporal jittering and sensor noise augmentation during training of the gait phase module to improve robustness to Cassie's noisy proprioceptive sensors
- **Gait event detection as auxiliary task:** Train the gait phase module with auxiliary heel-strike and toe-off detection losses to improve phase accuracy at critical gait transitions

## Limitations & Open Questions
- The paper uses CNNs on fixed-window inputs, while the Neural ODE approach in Project B operates in continuous time — the multi-resolution insights must be translated from convolutional to ODE architectural choices
- Ground-truth gait phase from force plates is available for human subjects but not for simulated Cassie; generating accurate gait phase labels for RL training requires careful design of the reward function or supervised pretraining
- The paper evaluates on steady-state walking; transient behaviors (gait initiation, stopping, turning) are not extensively covered, yet these are critical for Cassie's diverse locomotion repertoire
- Single shank-mounted IMU may not capture all relevant dynamics for Cassie; the full proprioceptive state (10 joint encoders + IMU) provides richer information that the architecture should exploit
