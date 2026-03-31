# Continuous Learning and Adaptation of Neural Control for Proprioceptive Quadruped Locomotion

**Authors:** (Journal of Bionic Engineering 2025)
**Year:** 2025 | **Venue:** Journal of Bionic Engineering
**Links:** https://link.springer.com/article/10.1007/s42235-025-00742-4

---

## Abstract Summary
This paper presents a modular closed-loop neural controller for quadruped locomotion that relies exclusively on minimal proprioceptive signals per leg — joint angles, joint velocities, and contact binary per foot. The controller uses a biologically-inspired central pattern generator (CPG) coupled with a learned neural adaptation module that enables continual online learning of robust motor patterns. The system demonstrates that sharing proprioceptive information across legs enables proactive terrain adaptation, where the front legs' contact experiences inform the rear legs' behavior before they encounter the same terrain features.

The key innovation is the continual learning mechanism that allows the neural controller to adapt its motor patterns during deployment without catastrophic forgetting of previously learned behaviors. Using an elastic weight consolidation (EWC) approach, the controller maintains a stable base locomotion policy while incrementally learning terrain-specific adaptations. This enables the robot to improve its performance over time on repeatedly-encountered terrains without losing the ability to walk on previously mastered surfaces.

The system demonstrates remarkable robustness, including graceful degradation under partial sensor failure — when one leg's proprioceptive sensors are disabled, the remaining three legs compensate through the inter-leg communication mechanism. This fault tolerance is critical for real-world deployment where sensor reliability cannot be guaranteed.

## Core Contributions
- **Minimal proprioceptive control:** Demonstrated robust quadruped locomotion using only 3 signals per leg (joint angle, velocity, binary contact), totaling 12 inputs for the entire robot — far fewer than typical RL approaches
- **Inter-leg proprioceptive sharing:** Showed that front-to-rear proprioceptive communication enables proactive terrain adaptation, where rear legs adjust their trajectory before encountering terrain changes based on front leg experience
- **Continual online learning:** Implemented EWC-based adaptation that allows the controller to incrementally learn new terrain behaviors without forgetting previously learned patterns
- **Graceful sensor degradation:** Demonstrated that the controller maintains functionality when individual leg sensors fail, with the remaining legs compensating through the shared proprioceptive network
- **CPG-neural hybrid architecture:** Combined a central pattern generator for base rhythm with a learned neural module for terrain-dependent modulation, separating rhythmic generation from adaptive control

## Methodology Deep-Dive
The controller architecture has three layers: a central pattern generator (CPG) that produces base rhythmic signals, a proprioceptive feature extraction module, and an adaptive neural modulator. The CPG is a network of coupled oscillators (one per leg) with learnable coupling weights and phase offsets. The oscillator dynamics follow the Matsuoka model: mutually inhibitory neuron pairs that naturally produce anti-phase oscillations suitable for walking gaits. The CPG parameters (frequency, amplitude, phase offsets) are initialized for a trot gait and allowed to adapt during learning.

The proprioceptive feature extraction module processes each leg's minimal sensor inputs (joint angle `q`, joint velocity `q̇`, binary foot contact `c`) through a per-leg LSTM with 64 hidden units. The LSTM captures temporal patterns in proprioceptive data, including contact timing, swing duration, and terrain-induced deviations from nominal trajectories. Crucially, the LSTM outputs from all four legs are concatenated and passed through a cross-leg attention layer, where each leg's representation attends to all other legs' representations. This cross-leg attention enables the inter-leg communication: front legs' terrain information flows to rear legs through the attention mechanism.

The adaptive neural modulator takes the combined CPG output and proprioceptive features as input and produces joint angle corrections `Δq_i` that modulate the CPG's nominal trajectory. The modulator is a 3-layer MLP (128-128-12 units, producing 3 DOF corrections per leg). During deployment, the modulator is trained online using a simplified reward signal: forward velocity tracking + stability penalty (based on IMU angular velocity) + energy penalty. The key is that this online learning uses the EWC regularizer: `L_EWC = L_task + λ Σ_i F_i (θ_i - θ*_i)²`, where `F_i` are Fisher information matrix diagonal entries computed on previous terrain data, `θ*_i` are the parameters after learning the previous terrain. This prevents new terrain adaptations from overwriting previously learned motor patterns.

The continual learning procedure operates in episodes. When the robot encounters a new terrain type (detected by a terrain classifier using foot contact patterns), it activates online adaptation: the neural modulator's parameters are updated via gradient descent on the deployment reward, with EWC regularization preventing forgetting. After ~100 steps of online adaptation (~5 seconds of walking), the robot's gait noticeably improves on the new terrain. Importantly, switching back to a previously-learned terrain does not require re-adaptation — the EWC-protected parameters retain the previous solution.

Sensor fault tolerance is achieved through the cross-leg attention mechanism. When a leg's sensors fail (outputting zeros), the attention mechanism naturally upweights the remaining legs' information. The LSTM hidden state for the faulty leg is driven entirely by cross-leg attention messages from functional legs. The authors demonstrate that the robot can maintain walking (at reduced speed) with up to 2 legs' sensors disabled, as long as at least one front and one rear leg have functional sensors.

## Key Results & Numbers
- Flat terrain walking: Stable at 0.3-1.2 m/s with minimal proprioceptive input (12 signals total)
- Rough terrain (5cm height variation): Maintains 0.8 m/s forward velocity after ~100 steps of online adaptation, vs. 0.5 m/s without adaptation
- Continual learning: After sequentially learning 5 terrain types, performance on the first terrain degrades by only 4% (vs. 35% without EWC)
- Sensor failure: With 1 leg's sensors disabled, speed drops by 15%; with 2 legs (diagonal) disabled, speed drops by 40% but walking remains stable
- Inter-leg communication benefit: Rear leg stumble rate decreases by 60% with cross-leg attention vs. without (front legs warn of upcoming terrain changes)
- Online adaptation convergence: ~100 steps (5 seconds) to reach 90% of adapted performance on new terrain
- Model size: 250K total parameters (CPG: 500 parameters, LSTM+attention: 180K, modulator MLP: 70K)

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to the Mini Cheetah RL project in multiple ways. First, the minimal proprioceptive control approach validates that robust quadruped locomotion is achievable with very limited sensory input — relevant for simplifying the Mini Cheetah's observation space. Second, the inter-leg proprioceptive sharing through attention provides a compelling architecture for the Mini Cheetah's policy network, enabling proactive terrain handling. Third, the continual learning with EWC is directly applicable to the Mini Cheetah's deployment scenario: instead of extensive domain randomization to cover all terrains in simulation, the policy can adapt online to encountered terrains. Fourth, the CPG-neural hybrid approach provides a baseline architecture that combines the stability of rhythmic pattern generators with the adaptability of neural networks, which is a proven approach for quadruped locomotion. Fifth, the sensor fault tolerance is critical for real-world Mini Cheetah deployment where joint encoders can fail.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The proprioceptive-only control paradigm is relevant to Project B, though Cassie's richer sensing (10 joint encoders, IMU, contact sensors) means the minimal-sensor focus is less critical. However, several concepts transfer well: (1) the inter-leg proprioceptive sharing through attention maps to cross-leg information flow in MC-GAT, where left and right leg subgraphs in the kinematic tree exchange information via graph attention; (2) the continual learning with EWC could be applied at the Primitives level, allowing Cassie to learn new gait primitives online without forgetting existing ones; (3) the CPG-neural hybrid provides an alternative to the Neural ODE Gait Phase component — instead of a learned ODE, a CPG with neural modulation could generate base gait rhythms; (4) the graceful sensor degradation validates the importance of the Safety level's ability to maintain locomotion under partial system failure.

## What to Borrow / Implement
- **Cross-leg attention for inter-limb communication:** Implement attention-based proprioceptive sharing across legs in the Mini Cheetah policy, enabling front-to-rear terrain communication
- **EWC-based continual adaptation:** Add elastic weight consolidation to the Mini Cheetah policy for online terrain adaptation during deployment, complementing domain randomization for sim-to-real transfer
- **Minimal observation space validation:** Test the Mini Cheetah policy with reduced observation spaces (joint angle + velocity + contact only) to determine minimum viable sensory input
- **CPG-neural hybrid as baseline:** Implement a CPG-modulated architecture as a baseline comparison against the pure RL policy for the Mini Cheetah, measuring the stability benefits of rhythmic pattern generators
- **Sensor fault tolerance testing:** Systematically evaluate both projects' policies under simulated sensor failure (zeroing individual sensor channels) to quantify robustness

## Limitations & Open Questions
- The CPG-neural hybrid inherently constrains the policy to rhythmic behaviors; agile non-periodic maneuvers (jumping, recovery from falls) may be difficult to express, which is a significant limitation for the Mini Cheetah's agile locomotion goals
- The continual learning with EWC has limited capacity — the number of terrains that can be sequentially learned without significant degradation is bounded by the Fisher information matrix's ability to protect important weights
- The minimal proprioceptive input (no joint torque, no IMU angular velocity, no body velocity estimate) may limit performance ceiling compared to richer observation spaces used in state-of-the-art RL locomotion
- The paper evaluates in simulation only; sim-to-real transfer of the continual learning mechanism (which requires online gradient computation on real hardware) is computationally challenging
