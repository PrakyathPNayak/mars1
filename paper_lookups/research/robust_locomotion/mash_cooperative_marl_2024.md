# MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Humanoid Locomotion

**Authors:** (arXiv 2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2508.10423

---

## Abstract Summary
MASH introduces a cooperative-heterogeneous multi-agent reinforcement learning (MARL) paradigm for humanoid locomotion where individual limbs act as independent agents that share a global critic. Instead of treating the entire humanoid as a single RL agent with a high-dimensional action space, MASH decomposes control authority among limb-level agents — each responsible for the actuated joints within its kinematic subtree. These limb agents receive specialized local observations (joint angles, velocities, contact forces within their subtree) while sharing a centralized value function that evaluates the global locomotion performance.

The cooperative-heterogeneous design distinguishes MASH from prior MARL locomotion work. "Cooperative" means all agents share the same team reward (locomotion performance), and "heterogeneous" means each limb agent has a different observation space, action space, and potentially different policy architecture — reflecting the physical asymmetry between, for example, a left arm and a right leg. This heterogeneity allows each limb to specialize while the shared critic coordinates their collective behavior toward coherent whole-body locomotion.

MASH demonstrates improved learning efficiency and policy generalization compared to single-agent PPO on humanoid locomotion tasks. The emergent inter-limb coordination is qualitatively similar to biological motor control, where limbs operate semi-independently under the coordination of higher-level neural structures. The approach achieves robust locomotion with better disturbance recovery and more natural gait patterns than monolithic single-agent baselines.

## Core Contributions
- **Limb-as-agent decomposition:** Formulated humanoid locomotion as a cooperative MARL problem where each limb is an independent agent controlling its own joints
- **Heterogeneous agent design:** Allowed each limb agent to have specialized observation spaces, action spaces, and policy architectures reflecting physical asymmetries
- **Centralized training with decentralized execution (CTDE):** Used a shared global critic during training while enabling per-limb inference at deployment, reducing the per-agent computational burden
- **Emergent cooperative coordination:** Showed that inter-limb coordination (gait phase locking, anti-phase arm swing) emerges naturally from the shared reward signal without explicit coordination mechanisms
- **Improved generalization:** Demonstrated that the multi-agent decomposition acts as an implicit inductive bias that improves policy generalization to unseen perturbations and terrains

## Methodology Deep-Dive
MASH decomposes a humanoid with `N` limbs into `N` cooperative agents `{A_1, ..., A_N}`. Each agent `A_i` controls the joints within its kinematic subtree. For a biped, a natural decomposition is: left leg agent (hip, knee, ankle), right leg agent (hip, knee, ankle), and optionally torso/arm agents. Each agent's observation includes: local joint states (angles, velocities, torques), local contact forces (foot ground reaction forces for leg agents), local body-frame IMU (accelerometer, gyroscope from the nearest link), and a shared global observation component (body velocity command, gravity vector, body orientation).

The policy for each agent is a separate neural network (MLP with 2 hidden layers, 256 units each), though parameter sharing between symmetric agents (left/right legs) is optionally enabled. During training, all agents share a single centralized critic `V(s)` that takes the full global state as input (all joint states, all contacts, body state). This CTDE paradigm provides a stable training signal — each agent's policy gradient uses the global value baseline, reducing the credit assignment problem inherent in multi-agent settings.

Communication between agents is implemented through a lightweight message-passing mechanism. Each agent broadcasts a fixed-size message embedding (32-dimensional) derived from its local observation and current action. Other agents receive these messages as additional input, enabling implicit coordination. The message content is learned end-to-end without explicit structuring, and ablation studies show that message passing improves gait symmetry and coordination but is not strictly necessary for stable locomotion.

Training uses Multi-Agent PPO (MAPPO) with a shared replay buffer. In each environment step, all limb agents simultaneously produce actions, the environment steps forward, and all agents receive the same team reward (velocity tracking + alive bonus - energy penalty - smoothness penalty). The advantage function is computed using the shared critic with GAE (λ=0.95). Each agent's policy is updated independently using its own advantage estimates but with the shared baseline, which significantly reduces variance compared to independent advantage estimation.

The curriculum starts with flat terrain and low-speed commands, gradually introducing rough terrain, slopes, and higher speeds. The authors find that the multi-agent decomposition learns faster in the early stages (low-speed flat walking) but requires more samples for complex behaviors (running, rough terrain), likely because coordination takes longer to emerge. However, the final performance exceeds single-agent baselines, particularly in generalization to unseen terrain types and external perturbations.

## Key Results & Numbers
- Humanoid flat-terrain walking: MASH achieves 12% higher reward than single-agent PPO and 8% higher than IPPO (independent PPO without shared critic)
- Disturbance recovery: MASH recovers from 40N lateral pushes that topple the single-agent baseline 60% of the time vs. 15% for MASH
- Gait symmetry: Left-right phase difference within 2° for MASH vs. 8° for single-agent PPO, measured over steady-state walking
- Sample efficiency: MASH requires 1.5x more environment steps to converge on flat terrain, but 0.8x steps on rough terrain (the inductive bias helps with complex terrains)
- Emergent coordination: Without explicit phase constraints, limb agents converge to anti-phase leg coordination and compensatory arm swing within 200M environment steps
- Per-agent inference cost: Each limb agent's forward pass is ~0.1ms (256-unit MLP), enabling parallel execution on separate CPU threads for real-time control

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: Medium**
MASH's per-limb agent decomposition is naturally applicable to the Mini Cheetah's 4 legs. Each leg (hip, knee) could be an independent agent with local proprioceptive observations, sharing a global critic that evaluates overall locomotion performance. Potential benefits include: (1) reduced per-agent action space (3 DOF per leg vs. 12 DOF total), which may simplify learning; (2) implicit gait phase coordination emerging from the shared reward, potentially replacing explicit gait generators; (3) better disturbance recovery through limb-level specialization. However, the Mini Cheetah's relatively simple morphology may not benefit as much from multi-agent decomposition compared to the more complex humanoid in the original paper.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: High**
MASH is highly relevant to Project B's hierarchical control architecture, particularly at the Primitives and Controller levels. The per-limb agent decomposition directly maps to Cassie's bipedal structure: left leg agent and right leg agent, each controlling their respective joints (hip roll, hip yaw, hip pitch, knee, toe). This maps naturally to the Primitives level, where different gait primitives (walking, turning, standing) could be implemented as coordination patterns between the two leg agents.

The centralized-critic-decentralized-execution paradigm aligns with Project B's Dual Asymmetric-Context Transformer architecture: during training, the Transformer has access to privileged information (terrain maps, exact contact states) in the centralized critic, while during deployment each limb agent operates with only local proprioceptive observations. The message-passing mechanism between limb agents could inform how the MC-GAT module routes information between left and right leg subgraphs. The emergent coordination results validate that explicit gait phase constraints may be unnecessary — the Neural ODE Gait Phase component might serve as a learned alternative to the hard-coded coordination that MASH discovers naturally.

## What to Borrow / Implement
- **Per-limb agent architecture for Cassie:** Decompose Cassie's controller into left-leg and right-leg agents with specialized observations, sharing a centralized critic — this can serve as the Primitives level architecture
- **CTDE training paradigm:** Use privileged critic (full state + terrain) during training with decentralized per-limb policies at deployment, directly compatible with the Dual Asymmetric-Context Transformer's asymmetric design
- **Lightweight message passing between limb agents:** Implement 32-dim message passing between left/right leg policies to enable implicit coordination, processed through MC-GAT's graph attention
- **Emergent coordination analysis:** Monitor gait phase relationships during training to verify that anti-phase coordination emerges, as a validation metric for the Neural ODE Gait Phase component
- **Disturbance recovery benchmarking:** Use MASH's lateral push test (40N) as a standardized disturbance recovery benchmark for Project B

## Limitations & Open Questions
- MASH does not incorporate hierarchical control — it operates at a single level, and integrating per-limb agents into the 4-level hierarchy (Planner→Primitives→Controller→Safety) requires additional architectural design
- The shared reward signal may be insufficient for Cassie's multi-objective tasks (velocity tracking + safety + energy + terrain adaptation); per-agent shaped rewards may be needed
- Message passing adds communication overhead that could be problematic for real-time deployment on Cassie's onboard compute if the number of agents or message dimensions grows
- Credit assignment remains challenging — when the robot falls, it is unclear which limb agent's action was at fault, potentially slowing learning of recovery behaviors
