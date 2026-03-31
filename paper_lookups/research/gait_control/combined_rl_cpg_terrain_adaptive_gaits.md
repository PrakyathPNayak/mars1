# Combined Reinforcement Learning and CPG Algorithm to Generate Terrain-Adaptive Locomotion

**Authors:** Various
**Year:** 2023 | **Venue:** MDPI Actuators
**Links:** https://www.mdpi.com/2076-0825/12/4/157

---

## Abstract Summary
This paper proposes a combined RL-CPG framework where RL shapes the high-level locomotion strategy while CPGs generate smooth, rhythmic leg motions. The RL agent learns to modulate CPG parameters for different terrains, achieving stable locomotion across flat, inclined, and rough surfaces. The approach was tested on multi-legged platforms demonstrating smooth gait transitions.

## Core Contributions
- Develops a hierarchical RL-CPG framework separating strategy (RL) from execution (CPG) for terrain-adaptive locomotion
- Demonstrates smooth gait transitions across terrain types without abrupt parameter changes
- Shows that CPG-generated base gaits provide stability guarantees while RL adds terrain adaptability
- Compares RL-only, CPG-only, and RL-CPG hybrid approaches, demonstrating the hybrid's superior performance
- Achieves stable locomotion on flat, inclined (up to 25°), and rough surfaces with a single framework
- Provides analysis of emergent gait patterns showing biologically plausible terrain adaptations

## Methodology Deep-Dive
The framework consists of two coupled components: a CPG oscillator network that generates rhythmic leg trajectories and an RL agent that modulates CPG parameters based on terrain and task requirements. The CPG provides the "how" of locomotion (smooth, coordinated leg movements), while the RL agent provides the "what" (which gait to use, how aggressive to be, how to adapt to terrain).

The CPG network uses Matsuoka oscillators, a biologically inspired model that produces stable limit cycle oscillations. Each oscillator is parameterized by intrinsic frequency, mutual inhibition weights, and adaptation time constants. The oscillator outputs drive joint position references through a mapping function that converts oscillator phase into foot trajectories (swing and stance phases). The coupling between oscillators determines the gait pattern — in-phase coupling produces bounding, alternating coupling produces trotting, and sequential coupling produces walking.

The RL agent operates at a lower frequency than the CPG (typically 10-20 Hz vs. 100-500 Hz for the CPG). At each RL step, the agent observes the robot's proprioceptive state, terrain feedback, and current CPG state, then outputs modifications to CPG parameters. These modifications include frequency scaling, amplitude adjustment, phase bias, and duty cycle changes. The RL agent is trained with PPO using a reward function that balances forward progress, stability, and energy efficiency.

A critical design choice is the parameter modulation range. The RL agent can only modify CPG parameters within a bounded range around their nominal values. This prevents the RL agent from producing configurations that break the CPG's stability properties. The nominal parameters are set based on known stable gaits, and the RL agent learns to make targeted adjustments for terrain adaptation.

Training uses a two-phase approach. First, the CPG parameters are initialized using manually designed gaits for flat terrain. Second, the RL agent is trained in simulation with terrain randomization, learning to modulate CPG parameters for various conditions. The pre-initialized CPG provides a safe starting point — even before RL training converges, the robot can maintain basic locomotion using the default CPG parameters.

## Key Results & Numbers
- RL-CPG hybrid outperforms RL-only by 25% in locomotion stability across mixed terrain
- RL-CPG hybrid outperforms CPG-only by 40% in terrain adaptability metrics
- Smooth gait transitions with <10ms latency when terrain changes are detected
- Stable locomotion on inclines up to 25° and rough terrain with 10cm height variations
- Energy consumption reduced by 18% compared to pure RL due to CPG's rhythmic efficiency
- Training convergence 2-4x faster than pure RL baselines due to reduced action space
- Emergent gait patterns match biological observations: wider stance on slopes, slower frequency on rough terrain

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: Medium**
The RL-CPG framework provides an alternative locomotion architecture for Mini Cheetah that prioritizes gait smoothness and stability. The hierarchical separation of strategy (RL) and execution (CPG) reduces the burden on the RL policy and could improve sim-to-real transfer by leveraging CPG's inherent smoothness. For Mini Cheetah's 12-DoF system, the CPG could generate coordinated leg movements while RL handles terrain adaptation. However, Mini Cheetah's existing pure RL pipeline may already achieve sufficient performance, making the CPG layer an optional enhancement rather than a necessity.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: High**
The RL-CPG integration directly informs the Neural ODE Gait Phase design in Project B. The paper's hierarchical separation (RL strategy → CPG execution) mirrors Project B's hierarchy (Planner/Primitives → Controller with Neural ODE Gait Phase). The Matsuoka oscillator dynamics provide a concrete mathematical framework that can be implemented as a Neural ODE with limit cycle attractors. The frequency and amplitude modulation by RL corresponds to Project B's architecture where higher hierarchy levels modulate gait timing through phase variable control. The bounded parameter modulation range is an important safety feature that can be adopted in Project B — the Controller can modulate gait phase within safe bounds defined by the Primitives level.

## What to Borrow / Implement
- Use Matsuoka oscillator dynamics as the basis for the Neural ODE Gait Phase module in Project B
- Implement bounded CPG parameter modulation to ensure gait phase stability in the hierarchical controller
- Adopt the two-phase training approach: initialize with stable gaits, then train RL for terrain adaptation
- Apply the hierarchical frequency design to both projects: RL at 10-20 Hz, execution at 500 Hz
- Use the RL-CPG comparison framework to benchmark pure RL vs. structured approaches for Mini Cheetah
- Implement the coupling structure analysis to ensure left-right leg coordination for Cassie's bipedal gait
- Borrow the emergent gait pattern analysis methodology to verify biologically plausible locomotion

## Limitations & Open Questions
- Matsuoka oscillators require manual tuning of intrinsic parameters (mutual inhibition, adaptation constants)
- The bounded modulation range limits the policy's ability to handle extreme terrain conditions
- Two-phase training requires good initial CPG parameters, which need domain expertise to set
- The framework is validated on multi-legged platforms — bipedal adaptation requires significant modifications
- Gait transition smoothness depends on the RL policy frequency relative to CPG frequency
- Non-periodic locomotion (standing, turning in place, recovery from falls) requires additional mechanisms outside the CPG
- The comparison between RL-CPG and pure RL may not generalize to all robot morphologies
