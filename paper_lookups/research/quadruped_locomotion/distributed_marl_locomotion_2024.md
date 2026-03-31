# Learning Advanced Locomotion for Quadrupedal Robots: A Distributed Multi-Agent Reinforcement Learning Approach

**Authors:** (MDPI Robotics 2024)
**Year:** 2024 | **Venue:** MDPI Robotics
**Links:** https://www.mdpi.com/2218-6581/13/6/86

---

## Abstract Summary
This paper introduces a distributed Multi-Agent Reinforcement Learning (MARL) framework for quadruped locomotion where each limb of the robot is treated as an independent intelligent agent. Rather than training a single centralized policy that controls all 12 joints simultaneously, the MARL approach decomposes the control problem into four cooperating agents—one per leg—each responsible for its own 3-DoF joint control. The agents share a common reward signal but maintain independent observations and policies, learning to coordinate through implicit communication via the shared body dynamics.

The distributed approach offers several compelling advantages: simplified reward design (per-limb rewards are easier to specify than whole-body rewards), modular learning (individual limbs can be trained or fine-tuned independently), high adaptability (the framework naturally handles asymmetric terrains and limb damage), and the emergence of complex locomotion behaviors including bipedal walking skills on a quadruped platform. The paper demonstrates that MARL-trained policies achieve comparable or superior performance to centralized single-agent policies on standard locomotion benchmarks while exhibiting greater adaptability to novel situations.

Particularly noteworthy is the emergence of bipedal locomotion skills on the quadruped—the robot learns to walk on just two legs when challenged with obstacles or when front legs are disabled—demonstrating that the per-limb agent decomposition naturally supports diverse locomotion strategies that would require explicit reward engineering in centralized approaches.

## Core Contributions
- Per-limb MARL framework where each of a quadruped's four legs operates as an independent RL agent with its own policy network, observation space, and action output
- Centralized Training Decentralized Execution (CTDE) paradigm with a shared value function that has access to all agents' observations during training, while individual policies receive only local observations at execution time
- Demonstration that MARL naturally produces coordinated gaits (walk, trot, pace, gallop) without explicit gait rewards or contact pattern specifications
- Emergence of advanced locomotion skills including bipedal walking, tripod gait, and adaptive limb coordination under asymmetric perturbations
- Simplified reward design: per-limb rewards focus on local objectives (foot clearance, ground reaction force, swing timing) while whole-body objectives (velocity, orientation) are handled by the shared reward
- Robustness to limb damage: when one leg is disabled, the remaining three agents naturally adapt their coordination without retraining
- Scalability analysis showing MARL training is 2.5× more sample efficient than centralized approaches for 12-DoF quadrupeds

## Methodology Deep-Dive
The MARL framework uses a Centralized Training with Decentralized Execution (CTDE) architecture based on Multi-Agent PPO (MAPPO). Each leg agent i ∈ {FL, FR, HL, HR} has: (1) a local observation o_i consisting of its own joint positions (3), joint velocities (3), foot contact state (1), and the shared body state (base velocity 3, angular velocity 3, projected gravity 3, commanded velocity 3), totaling 19 dimensions per agent; (2) a policy π_i(a_i | o_i) outputting 3 target joint positions (hip, thigh, calf); and (3) a local value function V_i(o_i) for advantage estimation.

During training, a centralized value function V_cent(o_1, o_2, o_3, o_4) has access to all four agents' observations (76 dimensions total) and is used for more accurate advantage estimation. At execution time, only the decentralized policies are used, each receiving its local 19-dimensional observation. This CTDE approach provides the benefits of global information during training without requiring global communication at deployment.

The reward function has two components: a shared team reward r_team and per-agent local rewards r_local_i. The team reward covers whole-body objectives: r_team = w_vel · exp(-||v - v_cmd||²) + w_orient · exp(-||ω||²) + w_height · exp(-(h - h_target)²) - w_energy · Σ|τ·ω|. The local reward for each agent covers limb-specific objectives: r_local_i = w_clearance · max(0, h_foot_i - h_threshold) · I(swing) + w_contact · I(stance ∧ contact) - w_slip · ||v_foot_i|| · I(contact), where I(·) are indicator functions for swing and stance phases.

The key insight is that inter-limb coordination emerges from the shared body dynamics and team reward without explicit coordination mechanisms. When one leg pushes forward, it creates body motion that all agents observe, implicitly communicating the locomotion phase. The centralized critic during training further facilitates coordination by capturing inter-agent dependencies in the value estimate.

Training uses MAPPO with independent policy networks per agent (each a 2-layer MLP with 128 units and Tanh activations). The centralized critic uses a larger network (256×256). Training hyperparameters: clip ε=0.2, γ=0.99, λ_GAE=0.95, learning rate 5×10⁻⁴, batch size 16384 (4096 per agent). Domain randomization includes friction (0.3–1.5), payload (0–3 kg), motor strength (±10%), terrain roughness (0–5 cm), and communication delay (0–10 ms). Training runs for 8000 iterations with 2048 parallel environments, approximately 12 hours on a single A100 GPU.

## Key Results & Numbers
- Velocity tracking RMSE: 0.05 m/s (MARL) vs. 0.04 m/s (centralized), showing comparable performance
- Sample efficiency: MARL converges in 3200 iterations vs. 8000 for centralized single-agent PPO (2.5× improvement)
- Emergent gaits: walk (0–0.6 m/s), trot (0.6–1.5 m/s), pace (lateral movement), gallop (>1.5 m/s) without gait-specific rewards
- Bipedal walking: quadruped successfully walks on two legs for up to 15 seconds when front legs are disabled
- Limb damage adaptation: 75% of pre-damage velocity maintained with one leg disabled (no retraining)
- Robustness: survives 50 N lateral pushes with <0.3 s recovery time
- Per-agent policy size: 33K parameters vs. 131K for centralized policy (4× smaller per agent, similar total)
- Training time: ~12 hours on A100 with 2048 environments

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
The MARL framework offers a fundamentally different and potentially superior approach to Mini Cheetah locomotion control. By decomposing Mini Cheetah's 12-DoF control into four 3-DoF per-limb agents, the approach simplifies reward design and naturally handles asymmetric scenarios (e.g., one leg on a different surface). The 2.5× sample efficiency improvement over centralized PPO would significantly reduce Mini Cheetah's training time.

The per-limb agent architecture is particularly attractive for Mini Cheetah's sim-to-real transfer: if per-limb policies are more modular, individual limb controllers could potentially be fine-tuned on real hardware without retraining the entire system. The emergent coordination behavior—gaits arising from shared dynamics rather than explicit rewards—aligns with Mini Cheetah's goal of learning natural, adaptive locomotion.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The per-limb agent concept is adaptable to Cassie's bipedal control, though with only two legs the decomposition is less dramatic. The CTDE architecture concept, however, is highly relevant to Cassie's hierarchical framework—each level of the hierarchy could be treated as an agent in a MARL formulation with centralized training and decentralized execution. The emergence of bipedal locomotion skills on the quadruped platform also provides interesting insights for Cassie's bipedal control.

The implicit coordination mechanism (agents coordinating through shared body dynamics) is relevant to understanding how Cassie's left and right leg controllers should interact—rather than explicit phase coordination, the shared dynamics may be sufficient for natural bipedal gait emergence.

## What to Borrow / Implement
- Implement the per-limb MARL decomposition for Mini Cheetah with CTDE-MAPPO, treating each leg as an independent agent
- Adopt the two-tier reward structure: team reward for whole-body objectives + local rewards for per-limb quality
- Use the centralized critic architecture during training for better inter-limb coordination learning
- Test the limb damage adaptation capability on Mini Cheetah for robust locomotion under actuator failures
- Explore the CTDE concept for Cassie's hierarchical architecture, treating hierarchy levels as cooperating agents

## Limitations & Open Questions
- The 4-agent decomposition assumes symmetric limb structure; extension to robots with asymmetric limb configurations (e.g., Cassie with different leg kinematics) requires careful observation space design
- Communication between agents is purely implicit (through body dynamics); explicit communication channels might improve coordination for complex tasks like parkour or precise footstep planning
- The centralized critic requires all agents' observations, which may become impractical for higher-DoF robots or larger agent counts
- Sim-to-real transfer of the MARL framework has not been demonstrated; coordinated multi-agent policies may be more sensitive to sim-to-real gaps than centralized policies
