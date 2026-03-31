# One Policy to Run Them All: An End-to-end Learning Approach to Multi-Morphology Locomotion

**Authors:** Nico Bohlinger et al.
**Year:** 2024 | **Venue:** CoRL
**Links:** https://nico-bohlinger.github.io/one_policy_to_run_them_all_website/

---

## Abstract Summary
This paper introduces URMA (Unified Robot Morphology Architecture), a single learned policy that controls locomotion across fundamentally different robot morphologies — quadrupeds, bipeds, and hexapods — without any morphology-specific modules or fine-tuning. URMA achieves this through a carefully designed observation encoding scheme that splits the input into morphology-general features (velocity commands, body orientation, gravity vector) and morphology-specific features (per-joint angles, velocities, and morphological descriptors), processed through an attention-based observation encoder that learns cross-morphology representations.

The key architectural innovation is an attention-based mechanism that processes variable-length per-joint observation tokens alongside fixed global observation tokens. The self-attention layer learns to weight joint-level information differently depending on the morphology and task context, enabling a single set of policy weights to produce appropriate behaviors for quadrupedal trotting, bipedal walking, and hexapodal gaits. The action space is similarly unified through a per-joint output scheme where each joint token produces its own target position.

URMA demonstrates strong sim-to-real transfer, deploying the same trained policy on a Unitree Go1 quadruped and a Cassie-like biped in the real world. The zero-shot cross-morphology generalization capability is validated by training on a subset of morphologies and deploying on held-out designs. Results show that the unified policy achieves performance competitive with or exceeding morphology-specific baselines, while requiring only a single training run.

## Core Contributions
- **URMA architecture:** Designed a unified observation encoder that handles variable morphology structures through attention-based joint-token processing with global context tokens
- **Multi-morphology single policy:** Demonstrated that one policy can walk quadrupeds, bipeds, and hexapods with competitive performance, trained end-to-end with PPO
- **Observation splitting scheme:** Separated morphology-general (commands, orientation) from morphology-specific (per-joint states) observations, with learned attention between them
- **Sim-to-real on multiple platforms:** Deployed the same policy weights on Go1 and Cassie-like hardware, validating cross-morphology sim-to-real transfer
- **Zero-shot morphology generalization:** Trained on a subset of morphologies and successfully transferred to held-out robots without fine-tuning
- **Scalable training pipeline:** Developed efficient multi-environment training setup where different morphologies are simulated simultaneously in parallel Isaac Gym environments

## Methodology Deep-Dive
URMA's observation encoder processes two streams. The global stream encodes morphology-independent information: commanded forward/lateral velocities, commanded yaw rate, body orientation quaternion, gravity vector in body frame, and a one-hot morphology identifier. These are projected to a fixed-dimensional embedding `g ∈ R^d`. The per-joint stream tokenizes each actuated joint with its local state (joint angle, joint velocity, previous action) concatenated with morphological features (joint axis, parent link mass, position relative to body center). Each joint produces a token `j_i ∈ R^d`.

The attention-based encoder processes the combined sequence `[g; j_1; j_2; ...; j_N]` through 2 layers of multi-head self-attention (4 heads, hidden dim 128). This allows every joint token to attend to the global context and to all other joints, learning cross-morphology coordination patterns. The output is a fixed-size policy representation obtained by: (1) taking the global token output as morphology context, (2) applying mean pooling over joint token outputs, and (3) concatenating both into the policy input.

The action space uses per-joint target position outputs. Each joint token's output representation is passed through a shared per-joint MLP that outputs a target angle offset relative to a morphology-specific default stance. This ensures the action dimensionality automatically scales with the number of joints. For policy optimization, PPO is used with a shared value function that uses the same encoder but a separate value head (mean over all token outputs → scalar).

Training is conducted in Isaac Gym with 4096 parallel environments, randomly assigned to different morphologies in each episode. The morphology set includes: Unitree Go1 (quadruped, 12 joints), a Cassie-like biped (10 joints), an ANYmal-like quadruped (12 joints), and a custom hexapod (18 joints). The reward function combines: forward velocity tracking (primary), lateral velocity tracking, yaw rate tracking, energy penalty, smoothness penalty (action rate), and alive bonus. Domain randomization applies perturbations to link masses (±15%), friction coefficients (±30%), motor strength (±10%), and observation noise.

A curriculum learning strategy gradually increases command difficulty: training begins with low-speed forward-only commands and progressively introduces higher speeds, lateral movement, and yaw commands. The morphology sampling is also curricularized — initially training on the "easiest" morphology (quadruped) before introducing bipeds and hexapods. The authors find this curriculum critical for bipedal walking, which is inherently harder to learn from scratch.

## Key Results & Numbers
- Single policy controls Go1 (12-DOF quadruped), Cassie-like (10-DOF biped), and hexapod (18-DOF) with a single set of ~2M parameters
- Compared to morphology-specific PPO policies: URMA achieves 95% (Go1), 89% (biped), 93% (hexapod) of individual performance on velocity tracking
- Zero-shot transfer to held-out ANYmal-C morphology: 82% of individually-trained performance without fine-tuning
- Sim-to-real on Go1: Successfully tracks velocity commands up to 1.5 m/s forward, 0.5 m/s lateral
- Sim-to-real on Cassie-like biped: Stable walking at 0.3-0.8 m/s with robust disturbance recovery
- Training: 2 billion total environment steps, ~48 hours on 1 GPU (A100), converges faster than training 3 separate policies
- Encoder: 2-layer Transformer, 128 hidden dim, 4 heads, ~800K encoder parameters

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
URMA is directly applicable to the Mini Cheetah project in multiple ways. First, the Mini Cheetah is morphologically similar to the Go1 quadruped on which URMA was successfully deployed, suggesting the architecture transfers well. Second, the observation splitting scheme (global commands vs. per-joint states) provides a clean template for the Mini Cheetah's observation space design. Third, the domain randomization strategy (mass, friction, motor strength perturbations) maps directly to the Mini Cheetah pipeline's sim-to-real transfer. Fourth, the curriculum learning strategy for velocity tracking is a validated approach for training locomotion on similar hardware. If the project expands to multi-robot scenarios, URMA enables training one policy that transfers across Mini Cheetah variants.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
URMA's explicit inclusion of a Cassie-like biped makes it directly relevant to Project B. The attention-based observation encoder provides an alternative or complement to the MC-GAT and Dual Asymmetric-Context Transformer components. Specific connections include: (1) the per-joint tokenization scheme can inform how proprioceptive observations are structured for input to the Dual Asymmetric-Context Transformer; (2) the global context embedding (commands, orientation) maps to the Planner level's output in the 4-level hierarchy; (3) the curriculum learning strategy (easy morphologies first, bipedal later) validates the Adversarial Curriculum approach used in Project B; (4) the sim-to-real transfer results on bipedal hardware provide a performance baseline for Project B's deployment. The key difference is that URMA operates as a flat policy while Project B uses a 4-level hierarchy — integrating URMA's observation encoding into the Controller level is a natural design choice.

## What to Borrow / Implement
- **Observation splitting into global/per-joint streams:** Adopt this design for the Controller level input in Project B's hierarchy, separating Planner outputs (global commands) from per-joint proprioception
- **Attention-based observation encoder:** Consider replacing or augmenting MC-GAT with URMA's simpler attention encoder as an ablation baseline; if performance is comparable, the simpler architecture may train faster
- **Curriculum learning strategy:** Implement URMA's morphology-aware curriculum (simple before complex) and speed curriculum (low speed before high speed) for Project B's Adversarial Curriculum
- **Domain randomization ranges:** Use URMA's validated randomization ranges (mass ±15%, friction ±30%, motor ±10%) as starting points for both Project A and Project B
- **Per-joint action heads with default stance offsets:** Output actions as offsets from a Cassie-specific default stance, reducing the action space complexity for the Controller level

## Limitations & Open Questions
- URMA's flat policy architecture does not support the hierarchical control structure (Planner→Primitives→Controller→Safety) used in Project B; significant architectural redesign is needed to integrate its principles into the hierarchy
- The Cassie-like biped in URMA may differ from the actual Cassie model used in Project B (different URDF, actuator models, gear ratios); direct performance comparisons require careful setup
- URMA's attention encoder uses 2 layers with 128 hidden dim — it may lack the representational capacity needed for Project B's more complex objectives (multi-terrain, safety constraints, diverse gait primitives)
- Sim-to-real results on the Cassie-like biped show significantly lower performance (89%) than the quadruped (95%), suggesting bipedal locomotion with a universal policy remains challenging
