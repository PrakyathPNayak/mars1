---
## 📂 FOLDER: research/hierarchical_rl/

### 📄 FILE: research/hierarchical_rl/hilma_res_hierarchical_residual_rl_loco_manipulation.md

**Title:** HiLMa-Res: A General Hierarchical Framework via Residual RL for Combining Quadrupedal Locomotion and Manipulation
**Authors:** Yiming Ni, Zhongyu Li, Xue Bin Peng, Sergey Levine, Koushil Sreenath
**Year:** 2024
**Venue:** IROS 2024
**arXiv / DOI:** arXiv:2407.06584

**Abstract Summary (2–3 sentences):**
HiLMa-Res introduces a two-level hierarchical framework that uses residual reinforcement learning to combine stable quadrupedal locomotion with manipulation capabilities. A pre-trained low-level locomotion controller provides a reliable walking foundation, while a high-level residual policy learns to modify joint trajectories for task-specific manipulation goals. The framework is demonstrated on Unitree hardware across tasks including ball dribbling, obstacle stepping, and object pushing with successful sim-to-real transfer.

**Core Contributions (bullet list, 4–7 items):**
- Two-level hierarchical framework combining pre-trained locomotion with residual manipulation policies
- Residual RL formulation that adds corrective actions on top of a frozen base locomotion controller
- General framework applicable to diverse loco-manipulation tasks without task-specific architectural changes
- Successful real-world deployment on Unitree quadruped for dribbling, stepping, and pushing tasks
- Demonstrated superiority over single-level end-to-end baselines and non-residual hierarchical approaches
- Efficient training pipeline leveraging existing locomotion controllers as stable foundations
- Analysis of residual action magnitudes showing the policy learns minimal but effective corrections

**Methodology Deep-Dive (3–5 paragraphs):**
The foundational component of HiLMa-Res is the pre-trained low-level locomotion controller. This controller is trained independently using PPO in a standard quadruped locomotion setup, learning to track velocity commands (forward, lateral, yaw rate) while maintaining balance and producing stable gaits. The locomotion policy maps proprioceptive observations—joint positions, joint velocities, body orientation, angular velocity, and previous actions—to desired joint position targets that are tracked by PD controllers at each joint. Once trained, this locomotion controller is frozen and serves as the base behavior upon which manipulation capabilities are layered. The key insight is that stable locomotion represents a difficult-to-learn but reusable skill that should not be disrupted by task-specific training.

The high-level residual policy operates by observing the current task state (e.g., ball position for dribbling, obstacle location for stepping, object pose for pushing) along with the robot's proprioceptive state, and outputs residual actions that are added to the base locomotion controller's outputs. Formally, the final joint position targets are computed as q_target = q_base + α · q_residual, where q_base comes from the frozen locomotion policy, q_residual comes from the high-level policy, and α is a scaling factor that constrains the magnitude of residual corrections. This additive formulation ensures that the residual policy cannot deviate too far from stable locomotion, providing a natural safety constraint. The residual policy is trained with PPO using task-specific reward functions that encode manipulation objectives while including regularization terms to minimize residual action magnitudes.

Training of the residual policy is conducted in simulation using Isaac Gym or similar GPU-accelerated simulators. The observation space for the high-level policy includes both the proprioceptive state and task-relevant exteroceptive information. For dribbling, this includes the ball's relative position and velocity; for stepping, the obstacle height and distance; for pushing, the object pose relative to the robot. The reward function is task-specific but follows a common template: a primary task reward (e.g., ball reaching target, robot clearing obstacle) combined with regularization penalties on residual action magnitude, energy consumption, and body orientation deviation. Domain randomization is applied to terrain friction, object properties, sensor noise, and robot dynamics to facilitate sim-to-real transfer.

The sim-to-real transfer pipeline builds on the robust base locomotion controller and adds task-specific considerations. Since the base controller has already been validated for real-world deployment, the primary challenge is ensuring that the residual corrections transfer well. The authors find that constraining residual magnitudes (via the α scaling factor and regularization penalties) is crucial for transfer success—large residual actions tend to exploit simulator inaccuracies and fail on hardware. The final deployed system runs at 50 Hz on the Unitree platform, with both the base controller and residual policy executing on the onboard computer. Task perception (ball tracking, obstacle detection) uses onboard cameras processed through lightweight perception modules.

Experimental evaluation compares HiLMa-Res against three baselines: (1) end-to-end single-level policies trained from scratch for each task, (2) hierarchical policies without residual connections where the high-level directly commands velocity targets to the low-level, and (3) fine-tuning the base locomotion policy on each task. The residual approach consistently outperforms all baselines, achieving higher task success rates while maintaining locomotion stability. The analysis of learned residual actions reveals that the policy discovers interpretable modifications—for example, during dribbling, the residual policy primarily adjusts the front leg trajectories to contact the ball while leaving the rear legs largely unchanged from the base gait.

**Key Results & Numbers:**
- Successful generalization across three distinct loco-manipulation tasks (dribbling, stepping, pushing) using the same framework
- Real-world deployment on Unitree quadruped with robust task completion in unstructured environments
- 30-50% improvement in task success rate over end-to-end single-level baselines
- Residual actions remain within 15% of base controller magnitudes, ensuring locomotion stability
- Training convergence achieved in approximately 2000 PPO iterations (~50M environment steps) for each task
- Base locomotion controller maintained >95% stability during manipulation tasks

**Relevance to Project A (Mini Cheetah):** MEDIUM — The locomotion component and residual RL training methodology are relevant to Mini Cheetah policy training. The manipulation aspects are less directly applicable, but the concept of building upon a frozen locomotion base could be useful for extending Mini Cheetah behaviors beyond basic locomotion.

**Relevance to Project B (Cassie HRL):** HIGH — The residual RL concept is directly applicable for combining locomotion primitives with higher-level skills in the Cassie HRL system. The idea of a frozen base controller with learned residual corrections maps naturally to the Controller level modulating Primitives outputs. The hierarchical decomposition and the constraint on residual magnitudes provide a practical template for maintaining stability while adding task-specific capabilities.

**What to Borrow / Implement:**
- Residual RL formulation for layering new capabilities onto pre-trained locomotion controllers
- α-scaling mechanism for constraining residual action magnitudes during training and deployment
- Training pipeline that freezes the base controller and only trains the residual policy
- Regularization strategy balancing task performance with residual magnitude minimization
- Transfer strategy leveraging pre-validated base controllers to reduce sim-to-real risk

**Limitations & Open Questions:**
- The frozen base controller limits the system's ability to learn fundamentally new locomotion patterns needed for some tasks
- Scaling factor α requires manual tuning per task—too small limits manipulation capability, too large risks instability
- The framework has been demonstrated on relatively simple manipulation tasks; complex multi-step manipulation may require additional hierarchy
- Real-world perception pipeline (ball/object tracking) is assumed reliable, but perception failures are not extensively analyzed
- The approach assumes the base locomotion controller is near-optimal; suboptimal base controllers may limit residual policy effectiveness
---
