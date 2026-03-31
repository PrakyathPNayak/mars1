# Terrain-Adaptive Central Pattern Generators with Reinforcement Learning for Hexapod Locomotion

**Authors:** Various
**Year:** 2023 | **Venue:** IEEE / Robotics Conference 2023
**Links:** https://ieeexplore.ieee.org/document/10342300

---

## Abstract Summary
This paper combines Central Pattern Generators (CPGs) with RL for terrain-adaptive gait generation on hexapod robots. The RL agent tunes CPG parameters (frequency, amplitude, phase offsets) in real-time based on terrain feedback. This hybrid approach reduces action space complexity compared to pure RL while maintaining adaptability across terrain types.

## Core Contributions
- Proposes a CPG-RL hybrid architecture where RL operates in a reduced parameter space (CPG parameters) rather than raw joint commands
- Demonstrates that RL-tuned CPG parameters enable smooth, terrain-adaptive gaits without manual gait design
- Reduces the effective action space from joint-level commands to 4-6 CPG parameters, significantly improving sample efficiency
- Achieves real-time CPG parameter adaptation based on proprioceptive terrain feedback
- Shows that CPG-generated rhythmic patterns provide a strong inductive bias for locomotion, reducing the learning burden on RL
- Validates across flat, inclined, rough, and mixed terrain with smooth gait transitions

## Methodology Deep-Dive
Central Pattern Generators (CPGs) are neural oscillator networks that produce rhythmic output patterns, inspired by biological neural circuits that generate locomotion patterns in animals. In this work, each leg is driven by a CPG oscillator characterized by parameters: frequency (ω), amplitude (A), phase offset (φ), and duty cycle (d). The CPG produces smooth, periodic joint trajectories that define the basic gait pattern (trot, walk, gallop, etc.).

The key innovation is using RL to modulate CPG parameters rather than directly outputting joint positions or torques. The RL agent receives proprioceptive observations (body orientation, angular velocity, joint states, foot contacts) and outputs CPG parameter modifications: Δω, ΔA, Δφ, and Δd for each leg group. These modifications are added to nominal CPG parameters, allowing the RL agent to adapt the gait in real-time while maintaining the smooth, rhythmic structure guaranteed by the CPG.

The CPG layer acts as a structured action space prior. Pure RL must discover rhythmic locomotion patterns from scratch, which requires extensive exploration. By constraining the policy to output CPG modulations, the search space is dramatically reduced — the RL agent only needs to learn when and how much to modify the gait, not how to generate locomotion from raw joint commands. This results in 3-5x faster training convergence compared to pure RL baselines.

Training uses PPO in simulation with terrain randomization. The reward function encourages forward velocity, energy efficiency, and stability. Terrain-specific adaptations emerge naturally: on rough terrain, the RL agent increases step height (amplitude) and decreases frequency; on slopes, it adjusts phase offsets to maintain body orientation; on flat terrain, it optimizes for energy-efficient trotting.

The CPG architecture uses coupled oscillators with inter-leg coupling to maintain gait coordination. The coupling terms ensure that modifying one leg's parameters doesn't disrupt the overall gait pattern. This built-in coordination reduces the number of independent parameters the RL agent must control and prevents uncoordinated leg movements that could cause falls.

## Key Results & Numbers
- 3-5x faster training convergence compared to pure RL (joint-level control) baselines
- Action space reduced from 18 dimensions (hexapod joint commands) to 6 dimensions (CPG parameters)
- Smooth gait transitions between terrain types with no abrupt motion changes
- 95%+ stability rate across flat, inclined (up to 20°), and rough terrain
- Energy efficiency improved by 15-20% compared to pure RL due to CPG's inherent smoothness
- Real-time CPG parameter adaptation at 50 Hz (RL policy) with CPG execution at 500 Hz

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The CPG-RL hybrid could provide structured gait generation for Mini Cheetah, reducing the action space from 12 joint positions to 4-6 CPG parameters. This would improve sample efficiency during PPO training and produce smoother gaits that are easier to transfer sim-to-real. However, pure RL with 12-DoF control at 500 Hz has been shown to work well for Mini Cheetah in prior work (e.g., MIT's own results), so the CPG layer may be unnecessary complexity. The approach is most beneficial if Mini Cheetah struggles with gait smoothness or sim-to-real transfer of gait patterns.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The CPG concept directly relates to Project B's Neural ODE Gait Phase module, which generates continuous phase variables for Cassie's periodic locomotion. The CPG's oscillator dynamics are mathematically similar to Neural ODEs with periodic attractors — both produce smooth, rhythmic signals that modulate gait timing. The RL-modulated CPG architecture maps onto Project B's hierarchy: the Planner/Primitives levels determine high-level locomotion strategy, while the Neural ODE Gait Phase (analogous to the CPG) provides rhythmic timing signals to the Controller level. The inter-leg coupling in CPGs is relevant to coordinating Cassie's left and right legs during walking and running. The reduced action space benefit also applies — the Controller outputs gait phase modulations rather than raw joint commands.

## What to Borrow / Implement
- Use the CPG-RL architecture as inspiration for the Neural ODE Gait Phase module in Project B
- Implement inter-leg coupling in the gait phase generator to ensure coordinated bipedal locomotion
- Test CPG parameter modulation as an alternative action space for Mini Cheetah if gait smoothness is an issue
- Adopt the hierarchical control frequency design: RL at 50 Hz modulating CPG/gait phase at 500 Hz
- Use the CPG's amplitude and frequency parameters as interpretable gait descriptors for curriculum learning
- Implement the coupled oscillator dynamics as a differentiable module compatible with PPO training

## Limitations & Open Questions
- CPG structure imposes a periodic locomotion prior — non-periodic behaviors (jumping, recovering from falls) are harder to represent
- Hexapod-to-biped transfer is non-trivial — bipedal locomotion has fundamentally different stability requirements
- The reduced action space may limit the policy's expressiveness for highly dynamic maneuvers
- CPG parameter ranges must be manually specified, which requires domain knowledge
- Inter-leg coupling parameters add another layer of tuning complexity
- The 50 Hz RL / 500 Hz CPG frequency split may need adjustment for different platforms
