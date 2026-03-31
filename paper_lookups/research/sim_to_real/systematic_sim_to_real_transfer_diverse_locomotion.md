# Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Locomotion Behaviors

**Authors:** Various (ETH Zurich)
**Year:** 2025 | **Venue:** arXiv 2025
**Links:** https://arxiv.org/abs/2509.06342

---

## Abstract Summary
Proposes a systematic framework for sim-to-real transfer using physics-grounded energy models and minimal parameter sets for motor/actuator identification. Validates bottom-up identification protocols across multiple robot platforms, achieving robust transfer without exhaustive domain randomization. The approach prioritizes interpretable, physically motivated system identification over black-box randomization strategies.

## Core Contributions
- Introduces physics-grounded energy models for actuator identification that capture motor dynamics with minimal parameters
- Develops a bottom-up identification protocol that systematically characterizes each subsystem (motors, gearboxes, joints, links)
- Demonstrates that accurate system ID can replace or significantly reduce the need for extensive domain randomization
- Validates the approach across multiple robot platforms (quadrupeds, bipeds) with diverse locomotion behaviors
- Provides open-source identification tools and protocols for reproducibility
- Shows that energy-based models generalize better than torque-based models for motor characterization
- Achieves robust zero-shot transfer for diverse gaits including walking, trotting, bounding, and galloping

## Methodology Deep-Dive
The core methodology centers on a bottom-up system identification approach that starts from individual actuator characterization and builds up to full-body dynamics. For each motor/actuator, the paper develops energy-based models that capture the input-output relationship through power flow analysis rather than direct torque mapping. This energy perspective naturally accounts for losses due to friction, heat dissipation, and mechanical compliance, yielding more physically accurate models with fewer parameters.

The identification protocol consists of three phases. Phase 1 involves isolated actuator testing where each motor is driven through sinusoidal trajectories at varying frequencies and amplitudes while recording electrical power input and mechanical power output. The energy balance equation is then fit to identify motor constants, viscous friction, Coulomb friction, and thermal parameters. Phase 2 extends to coupled joint dynamics, where multi-joint coordination patterns are used to identify coupling effects, gravity compensation terms, and link inertial parameters. Phase 3 validates the full-body model against recorded locomotion data.

A key insight is that the energy model formulation dramatically reduces the parameter space compared to traditional system ID approaches. Instead of identifying full rigid-body dynamics matrices, the energy approach requires only 5-8 parameters per actuator (motor constant, gear ratio, viscous friction, Coulomb friction, thermal resistance, spring constant for SEAs) plus standard link parameters. This minimal set is both identifiable from limited data and sufficient for accurate simulation.

The paper demonstrates that with accurate system ID, domain randomization can be reduced to a narrow band around identified values (±5-10%) rather than the wide ranges (±50-100%) typically used. This narrower randomization produces policies that are both more performant (less conservative) and more robust (better matched to real dynamics) than broadly randomized alternatives.

The validation spans multiple platforms and behaviors, with each platform going through the same identification protocol. Results show consistent transfer success across morphologies, suggesting the approach is general rather than robot-specific.

## Key Results & Numbers
- Energy-model system ID requires only 5-8 parameters per actuator vs. 20+ for traditional approaches
- Narrow-band randomization (±5-10%) matches or exceeds transfer success of wide-band (±50-100%)
- Zero-shot transfer success rate of 92% across tested locomotion behaviors
- Validated on 3+ robot platforms including quadrupeds and bipeds
- Motor model prediction error < 3% of measured energy consumption
- Full identification protocol can be completed in < 2 hours per robot
- Policies trained with identified models show 20-30% less conservative behavior vs. broadly randomized

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
The systematic actuator identification approach is directly applicable to Mini Cheetah's proprietary motors. Mini Cheetah uses custom direct-drive actuators where accurate motor modeling is critical for sim-to-real transfer. The energy-based identification protocol can be applied to characterize each of the 12 actuators, capturing the motor constants, friction profiles, and thermal behavior. The finding that narrow-band randomization with accurate ID outperforms wide-band randomization suggests the current domain randomization strategy for Mini Cheetah could be significantly improved. The 2-hour identification timeline is practical for lab settings. The multigait validation (walk, trot, bound, gallop) aligns perfectly with Mini Cheetah's expected gait repertoire.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
Cassie's Series Elastic Actuators (SEAs) present a particularly challenging system ID problem due to the compliance and damping characteristics of the series springs. The energy-based identification approach is well-suited for SEAs because the energy model naturally captures the spring's storage and dissipation characteristics—parameters that are notoriously difficult to identify with torque-based methods. Accurate SEA models would improve simulation fidelity across all hierarchy levels, from the Controller level (PD tracking) to the Safety level (LCBF constraint verification). The minimal parameter set also simplifies the RSSM/Dreamer world model's task of predicting dynamics. The cross-platform validation on bipeds specifically increases confidence in applicability to Cassie.

## What to Borrow / Implement
- Apply the 3-phase identification protocol to Mini Cheetah's 12 actuators (isolated → coupled → full-body)
- Implement energy-based motor models in MuJoCo for more accurate Mini Cheetah simulation
- Use the energy model formulation to characterize Cassie's SEAs (spring + motor subsystem)
- Replace broad domain randomization with narrow-band randomization around identified parameter values
- Integrate the identification tools into the training pipeline for periodic re-calibration
- Use the energy prediction error as a diagnostic metric for simulation fidelity

## Limitations & Open Questions
- Energy-based models may not capture all nonlinear effects (e.g., magnetic saturation, cogging torque)
- Identification requires access to the physical robot and dedicated testing time
- The protocol assumes quasi-static or slow-varying thermal conditions during identification
- Cross-coupling effects between distant joints may not be fully captured in Phase 2
- How to handle actuator degradation over time (wear, aging) without re-identification
- Extension to soft or compliant robots with distributed actuation remains unexplored
