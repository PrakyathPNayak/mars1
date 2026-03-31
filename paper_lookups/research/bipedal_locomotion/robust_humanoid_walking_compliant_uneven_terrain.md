# Robust Humanoid Walking on Compliant and Uneven Terrain with Deep Reinforcement Learning

**Authors:** Various
**Year:** 2024 | **Venue:** IEEE (2024)
**Links:** https://ieeexplore.ieee.org/document/10769793

---

## Abstract Summary
This paper develops deep RL controllers for humanoid walking on compliant (soft/deformable) and uneven terrain, addressing the challenge of ground compliance where traditional rigid-body assumptions fail. The approach uses domain randomization of terrain stiffness and damping alongside proprioceptive feedback for blind robust walking. The resulting controller achieves stable locomotion on surfaces with widely varying compliance properties without requiring terrain sensing.

## Core Contributions
- RL-based locomotion controller robust to compliant (soft, deformable) terrain — a challenge largely ignored by prior work
- Domain randomization over terrain stiffness and damping parameters, extending the standard suite of randomized physical quantities
- Proprioceptive-only (blind) control that handles terrain compliance without explicit terrain sensing or estimation
- Demonstration that rigid-body simulation can train policies robust to compliant terrain through sufficient randomization
- Successful hardware transfer showing real-world robustness on foam, mattresses, and uneven outdoor surfaces
- Analysis of how terrain compliance affects gait characteristics and policy behavior
- Comparison against classical controllers and RL baselines showing significant improvement on compliant surfaces

## Methodology Deep-Dive
Traditional legged locomotion assumes rigid ground contact — the foot hits a solid surface and forces are transmitted instantaneously. Compliant terrain (foam, sand, mud, grass) violates this assumption: the ground deforms under the robot's weight, absorbing energy and creating unpredictable contact dynamics. Classical controllers designed for rigid contact often fail on compliant terrain because their contact models are fundamentally wrong.

The paper addresses this by training RL policies with extensive domain randomization over terrain compliance parameters. During training, the ground's stiffness (spring constant) and damping (viscous coefficient) are randomized across a wide range, from very stiff (near-rigid) to very soft (foam-like). The policy must learn locomotion strategies that work across this entire spectrum. This forces the policy to avoid strategies that exploit rigid contact assumptions (e.g., precise foot placement timing, high-frequency corrections that depend on instantaneous force feedback) and instead adopt inherently robust behaviors.

The observation space includes standard proprioceptive information: joint positions, joint velocities, body orientation (estimated via IMU), body angular velocity, and gravity direction. Critically, no terrain information is provided — the policy must infer terrain properties from the dynamic response of its own body. The authors find that the policy implicitly estimates terrain compliance through observed discrepancies between commanded and actual body motion, effectively using the robot's body as a sensor.

The policy outputs target joint positions tracked by PD controllers. The authors note that the PD controller itself provides a degree of compliance matching — when the foot contacts soft terrain, the PD controller's impedance interacts with the terrain's impedance, creating a coupled system. Training with varied terrain compliance effectively teaches the policy to modulate its implicit impedance (through position targets and timing) to achieve stable contact across surfaces.

The reward function balances velocity tracking, energy minimization, body stability, and ground contact regularity. Specific attention is given to rewarding consistent foot-ground contact patterns, as compliant terrain tends to create irregular contact timing. A curriculum progressively introduces more challenging compliance values, starting with near-rigid terrain and gradually including softer surfaces as the policy develops basic walking skills.

## Key Results & Numbers
- Stable walking on terrain with stiffness varying from 500 N/m (very soft foam) to 50,000 N/m (near-rigid)
- Proprioceptive-only control achieves 85-95% success rate on compliant terrain vs. 30-50% for rigid-terrain-trained baselines
- Hardware transfer successful on foam mats, mattresses, grass, and gravel without policy modification
- Classical model-based controller achieves <20% success rate on the same compliant terrain suite
- Energy efficiency decreases 15-30% on compliant terrain (expected due to energy absorption by terrain)
- Gait automatically adapts: shorter, faster steps on soft terrain; longer, slower steps on rigid terrain
- Training with compliance randomization does not degrade performance on rigid terrain (<3% performance loss)

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
Compliant terrain handling is applicable to outdoor Mini Cheetah deployment scenarios where the robot encounters grass, mud, sand, or other deformable surfaces. The domain randomization approach over terrain stiffness and damping can be directly integrated into Project A's MuJoCo training pipeline. The finding that compliance randomization doesn't degrade rigid-terrain performance suggests it's a "free" robustness improvement. The proprioceptive-only approach aligns with Mini Cheetah's sensor suite. However, quadrupeds are inherently more stable on compliant terrain than bipeds due to their wider support polygon, so the gains may be less dramatic for Project A.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Directly relevant to Cassie walking on deformable or uneven surfaces, a critical real-world deployment scenario. Cassie's bipedal morphology makes it particularly vulnerable to compliant terrain effects, as the smaller support polygon provides less margin for the timing and force disruptions caused by ground deformation. The terrain compliance randomization should be incorporated into Project B's adversarial curriculum. The proprioceptive terrain inference mechanism aligns with Project B's contrastive terrain encoder (CPTE), though CPTE could be extended to explicitly estimate compliance parameters. The interaction between PD controller impedance and terrain compliance is relevant to Cassie's series elastic actuators (SEAs), which already introduce a compliance layer. The gait adaptation findings (shorter steps on soft terrain) inform the Neural ODE Gait Phase module's expected behavior.

## What to Borrow / Implement
- Add terrain stiffness and damping to the domain randomization parameter set for both projects
- Implement the compliance curriculum: start rigid, gradually introduce soft terrain during training
- Use the proprioceptive terrain inference concept to inform Project B's CPTE design — add compliance estimation
- Analyze how Cassie's SEA compliance interacts with terrain compliance to optimize PD controller gains
- Apply the gait adaptation findings to inform expected behavior of Project B's Neural ODE Gait Phase on varied terrain
- Use the paper's evaluation methodology (stiffness spectrum testing) to benchmark both projects' terrain robustness

## Limitations & Open Questions
- Simulation of compliant terrain is approximate — real deformable materials have complex nonlinear, hysteretic behavior
- Proprioceptive-only approach may fail on terrain where compliance changes mid-step (e.g., thin crust over mud)
- Limited to walking; running or jumping on compliant terrain introduces additional challenges (higher impact forces)
- Does not address granular terrain (sand, gravel) where terrain flows and shifts under the robot
- How accurately does MuJoCo's soft contact model represent real-world compliant materials?
- Can explicit terrain compliance estimation (via CPTE-like modules) improve upon purely implicit inference?
- What is the interaction between robot joint compliance (SEAs) and terrain compliance for bipedal stability?
