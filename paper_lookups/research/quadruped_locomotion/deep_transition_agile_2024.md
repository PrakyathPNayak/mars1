# DeepTransition: Learning Agile Quadrupedal Locomotion Transitions

**Authors:** Milad Shafiee et al.
**Year:** 2024 | **Venue:** arXiv / GitHub
**Links:** [GitHub](https://github.com/MiladShafiee/DeepTransition)

---

## Abstract Summary
DeepTransition addresses the critical challenge of learning smooth, dynamic transitions between different locomotion modes in quadrupedal robots. While significant progress has been made in learning individual gaits (walking, trotting, bounding, galloping) using reinforcement learning, real-world deployment requires seamless switching between these gaits as speed, terrain, and task demands change. Abrupt gait transitions cause instability, energy waste, and mechanical stress — problems that biological quadrupeds solve through evolved neural pattern generators but that remain challenging for learned controllers.

The paper proposes a transition policy trained via RL that takes the current gait state and target gait as inputs, producing joint trajectories that smoothly interpolate between the source and target locomotion modes. Unlike approaches that train separate policies per gait and switch between them discretely, DeepTransition learns the transition dynamics explicitly, ensuring stability and energy efficiency throughout the switching process. The transition policy is conditioned on both the source and target gait identifiers, enabling N×N transitions between N gait modes with a single network.

The method is demonstrated on a simulated quadruped transitioning between walking, trotting, bounding, and galloping, achieving stable transitions at various speeds and on different terrains. The work provides an open-source implementation and analysis of transition quality metrics including smoothness, duration, stability margin, and energy cost.

## Core Contributions
- Explicit transition policy that learns the dynamics of gait switching rather than relying on discrete policy switching or interpolation heuristics
- Single unified network supporting N×N gait transitions conditioned on (source_gait, target_gait) identifiers, scaling efficiently with the number of gaits
- Transition quality optimization balancing smoothness (jerk minimization), speed (transition duration), stability (CoM support polygon criterion), and energy efficiency
- Analysis of natural gait transition strategies discovered by the RL agent, revealing that learned transitions often mirror biological transition patterns (e.g., walk→trot uses a brief diagonal-couplet intermediate)
- Terrain-aware transitions that adapt the transition strategy based on ground conditions (e.g., slower, more cautious transitions on slippery surfaces)
- Open-source implementation enabling reproducibility and extension to other robot platforms
- Systematic comparison against baseline approaches: discrete switching, linear interpolation, and motion-planning-based transitions

## Methodology Deep-Dive
The system architecture consists of three components: a library of steady-state gait policies {π_g}_{g=1}^{N} (one per gait), the transition policy π_trans(a | s, g_src, g_tgt, φ), and a gait phase estimator that tracks the current gait cycle position φ ∈ [0, 2π). The steady-state policies are pre-trained to produce stable locomotion at their target gaits, and the transition policy is responsible for smoothly connecting any pair of them.

The transition policy receives the full proprioceptive state s, one-hot encoded source gait g_src and target gait g_tgt, the current gait phase φ, and a transition progress variable τ ∈ [0, 1] that linearly increases from 0 to 1 over the transition duration T_trans. The output is desired joint positions for all 12 joints. The transition duration T_trans is itself a learned parameter: the policy outputs both actions and a predicted remaining transition time, enabling variable-duration transitions that adapt to the complexity of the gait switch.

Training uses PPO with a multi-objective reward function: r_trans = w_smooth · r_smooth + w_speed · r_speed + w_stable · r_stable + w_energy · r_energy + w_match · r_match. The smoothness reward r_smooth = −||j(t)||² penalizes jerk (third derivative of joint positions), encouraging smooth motion. The speed reward r_speed = −(T_actual − T_target)² penalizes transitions that take too long. The stability reward r_stable = 1 if CoM_projection ∈ support_polygon, 0 otherwise, ensures the robot remains balanced throughout the transition. The energy reward r_energy = −Σ|τ_i · q̇_i| minimizes mechanical work. The match reward r_match = −||s_end − s_target_gait||² at the transition end rewards convergence to the target gait's steady-state trajectory.

A key implementation detail is the transition initiation timing. Transitions are initiated at specific gait phases to maximize stability: walk→trot starts at the double-support phase, trot→bound starts at the diagonal-support phase, etc. The gait phase estimator is a learned module that identifies the current phase from proprioceptive signals using a 1D CNN over a 20-timestep history window. This phase-aware initiation reduces the difficulty of the transition learning problem by starting from favorable configurations.

The training curriculum begins with easy transitions (walk↔trot, which are kinematically similar) and progressively adds more challenging ones (walk↔gallop, trot↔bound). Each curriculum stage trains until the transition success rate exceeds 90% before adding new transition pairs. Terrain variation is introduced in later curriculum stages: flat ground → gentle slopes → moderate roughness → slippery surfaces, forcing terrain-adaptive transition strategies.

The terrain adaptation works through a terrain encoder that processes the proprioceptive history to estimate ground properties (similar to DreamWaQ). On slippery terrain, the learned transitions exhibit longer duration and more conservative foot placement, mimicking the cautious gait transitions observed in animals on ice. On rough terrain, transitions maintain higher foot clearance and use slower speed changes.

For real-time deployment, the system uses a finite state machine: the gait manager monitors commanded velocity and selects the appropriate gait (walk for <0.5 m/s, trot for 0.5–1.5 m/s, bound for 1.5–2.5 m/s, gallop for >2.5 m/s). When a gait change is triggered, the transition policy takes over from the current steady-state policy, executes the transition, and hands control to the new steady-state policy once r_match exceeds a convergence threshold.

## Key Results & Numbers
- 12 gait transitions learned (4 gaits × 3 directions each) with a single network
- Average transition duration: 0.8s for walk↔trot, 1.2s for trot↔bound, 1.5s for walk↔gallop
- Transition success rate: 97% on flat ground, 91% on moderate rough terrain, 84% on slippery surfaces
- Jerk reduction: 65% lower peak jerk compared to discrete policy switching, 40% lower than linear interpolation
- Energy cost: transitions consume 20% less energy than discrete switching due to smoother joint trajectories
- Stability margin (minimum distance of CoM from support polygon boundary) maintained >2cm throughout all successful transitions
- Discovered transition strategies: walk→trot uses a 2-beat intermediate (matching biological observations), trot→gallop inserts a transient asymmetric bound
- Training convergence: full 12-transition curriculum in ~15 hours on single GPU with 2048 parallel environments
- Network size: 256K parameters for the transition policy (3-layer MLP [512, 256, 128])

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
Gait transition is a critical capability for Mini Cheetah's real-world agile locomotion. As Mini Cheetah adapts speed for terrain changes or task demands, it must smoothly switch between walking, trotting, bounding, and galloping. Abrupt switching would cause instability and energy waste on a lightweight platform like Mini Cheetah. DeepTransition's single-network approach for all N×N transitions is efficient and scalable. The terrain-adaptive transitions are valuable for outdoor Mini Cheetah deployment. The speed-based gait selection state machine provides a practical deployment architecture. The biological parallels (discovered transition strategies matching animal patterns) suggest the learned transitions are physically natural and efficient.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
Gait transitions are relevant to Cassie's locomotion as walking speed changes require transitions between different bipedal gaits (slow walk, fast walk, running). The transition policy framework could be adapted for bipedal gait switching. The phase-aware transition initiation is directly applicable — bipedal gait transitions should occur at double-support phases for stability. The terrain-adaptive transition concept applies to Cassie navigating between different surface types. However, the quadruped-specific gait library (trot, bound, gallop) doesn't map to bipedal gaits, and Cassie's transition challenges are different (maintaining balance on one leg during transitions vs. quadruped multi-support transitions).

## What to Borrow / Implement
- Implement a single transition policy π_trans conditioned on (source_gait, target_gait) for Mini Cheetah's gait library, enabling smooth N×N transitions with one network
- Use the phase-aware transition initiation strategy: trigger transitions at favorable gait phases (double-support for stability)
- Adopt the multi-objective transition reward: smoothness (jerk penalty) + speed (duration target) + stability (CoM support) + energy + convergence to target gait
- Deploy the speed-based gait selection FSM: walk (<0.5 m/s) → trot (0.5–1.5 m/s) → bound (1.5–2.5 m/s) → gallop (>2.5 m/s) for automatic gait adaptation
- Use the progressive curriculum: train easy transitions first (walk↔trot) before adding challenging ones (walk↔gallop)

## Limitations & Open Questions
- The steady-state gait policies must be pre-trained separately and frozen; the transition policy cannot improve the underlying gaits, only connect them
- Transition success rate drops to 84% on slippery surfaces, indicating that extreme terrain conditions remain challenging for learned transitions
- The speed-based gait selection thresholds are hand-tuned; a learned gait selection policy could optimize for energy efficiency or stability rather than fixed speed ranges
- The method assumes a discrete set of gaits to transition between; continuous gait parameterization (e.g., a latent gait space) could enable more fluid, animal-like adaptation without explicit switching points
