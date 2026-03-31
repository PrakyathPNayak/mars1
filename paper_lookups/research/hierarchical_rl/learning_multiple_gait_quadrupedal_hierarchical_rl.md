---
## 📂 FOLDER: research/hierarchical_rl/

### 📄 FILE: research/hierarchical_rl/learning_multiple_gait_quadrupedal_hierarchical_rl.md

**Title:** Learning Multiple-Gait Quadrupedal Locomotion via Hierarchical Reinforcement Learning
**Authors:** Kyu-Byoung Kim, Seung-Yun Song, Jae-Han Park
**Year:** 2023
**Venue:** International Journal of Precision Engineering and Manufacturing (Springer)
**arXiv / DOI:** 10.1007/s12541-023-00885-6

**Abstract Summary (2–3 sentences):**
This paper presents a hierarchical reinforcement learning framework with a high-level gait selection module and a low-level reactive controller for quadrupedal locomotion. The high-level module selects among walk, trot, and gallop gaits based on speed commands and terrain conditions, while the low-level controller generates joint-level commands for the selected gait. Both modules are trained in simulation and successfully transferred to real hardware, demonstrating smooth gait transitions and energy-efficient multi-gait locomotion.

**Core Contributions (bullet list, 4–7 items):**
- Two-level hierarchical framework separating gait selection (high-level) from gait execution (low-level)
- Automatic gait selection policy that learns speed-dependent and terrain-dependent gait transitions
- Low-level reactive controllers specialized for walk, trot, and gallop gaits
- Smooth gait transition mechanism with interpolation during switching periods
- Energy efficiency optimization showing 20% reduction compared to single-gait baselines
- Successful sim-to-real transfer of the complete hierarchical system
- Analysis of emergent gait transition boundaries matching biological quadruped locomotion patterns

**Methodology Deep-Dive (3–5 paragraphs):**
The hierarchical architecture is organized into two distinct levels with different temporal resolutions. The high-level gait selector operates at a lower frequency (approximately 2–5 Hz) and makes discrete decisions about which gait mode to employ: walk, trot, or gallop. Its observation space includes the commanded forward velocity, current body velocity, terrain inclination estimates, and energy consumption metrics. The high-level policy is parameterized as a discrete action policy with three outputs corresponding to the three gait modes. It is trained using PPO with a reward function that balances velocity tracking accuracy with energy efficiency—the agent learns to select gaits that achieve the desired speed while minimizing actuator power consumption. The high-level policy naturally discovers speed-dependent gait transition boundaries: walking at low speeds, trotting at moderate speeds, and galloping at high speeds, mirroring the Froude number-based gait transitions observed in biological quadrupeds.

The low-level controller consists of three separate policy networks, one for each gait mode. Each low-level policy receives proprioceptive observations (joint positions, joint velocities, body orientation via IMU, contact force estimates) and the desired velocity command, outputting target joint positions for all 12 joints (3 per leg on a standard quadruped). The low-level policies are trained independently using PPO, each with a gait-specific reward function that includes: (1) velocity tracking reward, (2) gait-specific contact pattern reward that encourages the appropriate foot contact sequence (e.g., diagonal pairs for trot, sequential for walk, synchronized front-back for gallop), (3) smoothness penalties on joint accelerations and torques, and (4) stability rewards based on body orientation and height maintenance. By training each gait policy independently, the authors ensure that each controller is an expert at its specific gait pattern before being composed into the hierarchical system.

Gait transitions are handled through a temporal interpolation mechanism that blends the outputs of the outgoing and incoming gait controllers over a transition window. When the high-level selector switches from gait A to gait B, the system computes the final joint targets as q_target = (1 - α(t)) · q_A + α(t) · q_B, where α(t) smoothly transitions from 0 to 1 over a fixed window (typically 0.2–0.5 seconds). This interpolation prevents abrupt changes in joint targets that could cause instability or mechanical stress. The authors experiment with different interpolation profiles (linear, cosine, sigmoid) and find that cosine interpolation provides the smoothest transitions. The transition window duration is a tunable parameter—shorter windows enable faster gait changes but risk instability, while longer windows are smoother but slower to respond.

The training pipeline uses a staged approach. First, the three low-level gait controllers are trained independently in simulation with gait-specific reward functions and terrain randomization. Each controller is trained for approximately 100M environment steps until convergence. Then, the high-level gait selector is trained with the low-level controllers frozen, learning to select the optimal gait for each speed and terrain condition. The high-level training uses the same simulation environment but with varying speed commands and terrain profiles. Domain randomization is applied throughout, including randomization of robot mass (±10%), joint friction (±20%), ground friction (±30%), and observation noise. The simulation uses a realistic quadruped model with accurate actuator dynamics including torque limits and communication delays.

Sim-to-real transfer is achieved by deploying the complete hierarchical system on real quadruped hardware. The authors find that the gait-specific low-level controllers transfer well due to the domain randomization applied during training. The high-level gait selector also transfers effectively because it operates on relatively low-dimensional and slowly-varying features (speed, terrain slope) that are well-estimated on real hardware. The real-robot experiments validate smooth gait transitions on flat ground, mild slopes, and varying surfaces. The energy consumption analysis confirms that the multi-gait system uses approximately 20% less energy than a single-gait (trot-only) baseline when traversing speed-varying trajectories, as the system appropriately selects walking at low speeds and galloping at high speeds rather than forcing a trot across all conditions.

**Key Results & Numbers:**
- Smooth transitions between 3 gaits (walk, trot, gallop) with transition times of 0.2–0.5 seconds
- 20% energy reduction compared to single-gait (trot-only) baseline across varied speed profiles
- Successful sim-to-real deployment on quadruped hardware
- Emergent gait transition boundaries at ~0.8 m/s (walk→trot) and ~1.5 m/s (trot→gallop), consistent with biological Froude number predictions
- Each low-level gait controller achieves >90% velocity tracking accuracy within its operating range
- High-level gait selector converges in approximately 10M environment steps (after low-level pretraining)

**Relevance to Project A (Mini Cheetah):** HIGH — Multi-gait framework is directly applicable to Mini Cheetah's diverse locomotion modes. The Mini Cheetah is capable of walking, trotting, bounding, and galloping, and this hierarchical gait selection approach provides a structured method for learning when to use each gait and how to transition between them. The energy efficiency benefits are also relevant for extending operation time.

**Relevance to Project B (Cassie HRL):** MEDIUM — The gait selection hierarchy is relevant to the primitives level of the Cassie HRL system, where different locomotion modes (walking, running, turning) need to be selected and composed. However, Cassie as a biped has fewer distinct gaits than a quadruped, and the four-level hierarchy in the Cassie project goes significantly beyond the two-level structure presented here.

**What to Borrow / Implement:**
- Speed-dependent gait selection policy design for automatic gait mode management
- Cosine interpolation mechanism for smooth transitions between locomotion primitives
- Staged training pipeline: independent primitive training followed by hierarchical composition
- Gait-specific contact pattern rewards for training distinct locomotion modes
- Energy efficiency reward component to encourage appropriate gait selection

**Limitations & Open Questions:**
- Limited to three pre-defined gaits; the system cannot discover novel gaits or intermediate locomotion modes
- Two-level hierarchy may be insufficient for complex tasks requiring additional planning or safety layers
- Gait transition interpolation is purely kinematic; dynamic-aware transition planning could improve stability
- The high-level selector is trained with frozen low-level controllers, preventing co-adaptation that might improve performance
- Terrain perception is limited to inclination estimates; more complex terrains (stairs, gaps) may require richer perception
- The energy efficiency comparison uses a single-gait baseline; comparison with other multi-gait methods would strengthen the analysis
---
