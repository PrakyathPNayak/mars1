# Guided Reinforcement Learning for Robust Multi-Contact Loco-Manipulation

**Authors:** ETH Zurich (2024)
**Year:** 2024 | **Venue:** arXiv
**Links:** https://arxiv.org/abs/2410.13817

---

## Abstract Summary
This paper presents a guided reinforcement learning framework for robust multi-contact loco-manipulation on quadruped robots. The approach uses a model-based trajectory optimizer (specifically a contact-implicit trajectory optimizer) to generate reference demonstrations for complex tasks that involve simultaneous locomotion and manipulation, such as opening heavy doors, pushing objects, and navigating through constrained spaces while maintaining balance. These demonstrations are then used to guide RL policy training, combining the contact-rich planning capability of optimization-based methods with the robustness and generalization of learned policies.

The trained RL policies significantly outperform both the original trajectory optimizer (which is brittle to perturbations) and standard RL trained from scratch (which struggles with the exploration challenge of multi-contact tasks). A key finding is that the learned policies exhibit emergent recovery behaviors—the robot can recover from pushes, slips, and unexpected contacts that were never present in the demonstration data. This emergent robustness arises from the RL fine-tuning process, which exposes the policy to perturbations during training.

The framework is validated in both simulation and on real quadruped hardware (ANYmal), demonstrating successful door opening, box pushing, and multi-contact locomotion through narrow passages. The sim-to-real transfer is achieved through domain randomization and the inherent robustness of the learned policies.

## Core Contributions
- Proposes a two-stage pipeline: contact-implicit trajectory optimization generates demonstrations, then RL fine-tunes robust policies from these demonstrations
- Demonstrates that RL policies guided by optimization-based demonstrations exhibit emergent recovery behaviors not present in the original demonstrations
- Achieves robust multi-contact loco-manipulation including door opening, object pushing, and constrained passage navigation on real quadruped hardware
- Shows that the guided RL approach significantly outperforms both pure optimization (brittle) and pure RL from scratch (poor exploration)
- Validates real-world transfer on ANYmal quadruped through domain randomization and robust policy learning
- Introduces a multi-contact reward shaping strategy that balances locomotion stability with manipulation task progress
- Provides ablation studies showing the contribution of each component (demonstrations, domain randomization, reward shaping)

## Methodology Deep-Dive
The pipeline begins with a contact-implicit trajectory optimizer that plans through multi-contact sequences. Unlike standard trajectory optimization that pre-specifies contact schedules, the contact-implicit formulation discovers contact sequences as part of the optimization. This is crucial for loco-manipulation, where the robot must simultaneously plan footstep contacts for locomotion and end-effector contacts for manipulation. The optimizer uses a complementarity-based contact model with friction cone constraints, solved via sequential quadratic programming (SQP) with warm-starting across time horizons.

The generated trajectories serve as demonstrations for the RL phase. The RL agent (PPO with GAE) is initialized with behavior cloning pre-training on the demonstrations, then fine-tuned with a composite reward function. The reward combines: (1) a tracking reward that penalizes deviation from the reference trajectory (joint positions, base pose, contact timing), (2) a task-progress reward (e.g., door angle for door opening, object displacement for pushing), (3) stability rewards (base orientation, foot clearance, contact force regularity), and (4) energy/smoothness penalties (joint velocity, joint acceleration, torque). The tracking reward weight is annealed during training, starting high (to stay close to demonstrations) and decreasing over time (to allow the policy to discover more robust strategies).

Domain randomization is applied extensively during RL training. Randomized parameters include: ground friction (0.3-1.2), payload mass (±30%), motor strength (±20%), sensor noise (Gaussian on joint positions, velocities, IMU), communication latency (0-20ms), terrain height perturbations, and initial state perturbations. This randomization is critical for both robustness and sim-to-real transfer, as it exposes the policy to conditions far beyond the original demonstrations.

The policy architecture is a multi-layer perceptron (MLP) with 3 hidden layers of 512 units each, using ELU activations. The observation space includes proprioceptive state (joint positions, velocities, base orientation, angular velocity), exteroceptive information (door handle position, object position), and the current phase of the reference trajectory. The action space is desired joint positions, commanded to PD controllers at 50 Hz. A privileged teacher-student training approach is used: the teacher has access to ground-truth object states, while the student relies only on onboard sensing.

The asymmetric actor-critic framework uses a privileged critic with access to the full simulation state (including contact forces, terrain information, and object dynamics), while the actor only receives the observation space available on the real robot. This asymmetry accelerates training while ensuring deployability.

## Key Results & Numbers
- Door opening success rate: 94% (guided RL) vs. 73% (trajectory optimization alone) vs. 12% (RL from scratch)
- Box pushing: 89% success with 15cm target accuracy vs. 61% (optimization) vs. 8% (RL from scratch)
- Recovery from 50N lateral pushes during door opening: 87% success for guided RL vs. 23% for optimization
- Sim-to-real transfer: 82% success on real ANYmal for door opening (vs. 94% in simulation)
- Training time: 4 hours on single GPU (NVIDIA A100) for 50M environment steps
- Domain randomization improves real-world success by 34% compared to no randomization
- Tracking reward annealing improves final performance by 18% compared to fixed tracking weight
- Policy inference time: <1ms per step, enabling 50 Hz real-time control

## Relevance to Project A (Mini Cheetah Quadruped RL)
**Rating: High**
This paper is directly applicable to Mini Cheetah loco-manipulation tasks. The guided RL framework addresses the core challenge of learning complex multi-contact behaviors (door opening, object interaction) on quadruped hardware. The two-stage pipeline (trajectory optimization → RL fine-tuning) provides a practical approach for Mini Cheetah, where pure RL exploration for manipulation tasks is prohibitively sample-inefficient.

The emergent recovery behaviors are particularly valuable for Mini Cheetah, which will encounter unexpected perturbations during real-world deployment. The domain randomization protocol and sim-to-real transfer methodology provide a tested recipe. The asymmetric actor-critic with privileged training is directly implementable on Mini Cheetah's sensor suite. The ANYmal hardware platform is comparable in scale and capability to Mini Cheetah, making the results directly transferable.

## Relevance to Project B (Cassie Bipedal Hierarchical RL)
**Rating: Medium**
The multi-contact planning methodology is relevant to Cassie's whole-body control challenges, particularly for tasks requiring hand contacts (if manipulators are added) or complex terrain negotiation where multiple body parts contact the environment. The guided RL framework could enhance the Primitives level by providing trajectory optimization references for complex locomotion maneuvers (e.g., stepping over obstacles, climbing stairs).

However, the paper focuses on quadruped loco-manipulation, and the contact-implicit optimizer would need significant adaptation for bipedal dynamics. Cassie's primary challenge is balance-critical locomotion rather than manipulation, which reduces the direct applicability. The asymmetric actor-critic training and domain randomization strategies are applicable regardless of morphology and should be adopted for Cassie's training pipeline.

## What to Borrow / Implement
- Implement the two-stage guided RL pipeline for Mini Cheetah: use trajectory optimization to generate demonstrations, then fine-tune robust policies via PPO
- Adopt the tracking reward annealing strategy—start with tight tracking, relax over time to discover more robust strategies
- Apply the domain randomization protocol (friction, mass, motor strength, latency, terrain) for both Mini Cheetah and Cassie sim-to-real transfer
- Use the asymmetric actor-critic framework with privileged critic for both projects' training pipelines
- Leverage the contact-implicit trajectory optimizer for generating reference trajectories for complex multi-contact scenarios

## Limitations & Open Questions
- Contact-implicit trajectory optimization is computationally expensive and requires accurate dynamic models, which may not be available for all tasks
- The method assumes access to a good trajectory optimizer; tasks where optimization also fails (highly dynamic maneuvers, extreme terrain) remain unsolved
- Real-world success rates (82%) lag behind simulation (94%), indicating remaining sim-to-real gaps in contact-rich scenarios
- The paper focuses on quasi-static manipulation; dynamic manipulation (throwing, catching) and high-speed locomotion during manipulation are not addressed
