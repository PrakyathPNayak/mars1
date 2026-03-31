# Learning-Based Locomotion Controllers for Quadruped Robots in Indoor Environments

**Authors:** Various
**Year:** 2024 | **Venue:** IEEE (2024)
**Links:** https://ieeexplore.ieee.org/document/10594976

---

## Abstract Summary
This paper develops reinforcement learning-based locomotion controllers specifically optimized for quadruped robots operating in indoor environments. The controllers handle terrain features characteristic of buildings including stairs, ledges, smooth floors, and transitions between surface types. The approach incorporates stair detection and terrain awareness into the policy observation space, enabling robust indoor navigation and deployment on real quadruped hardware.

## Core Contributions
- Develops RL locomotion controllers specifically optimized for indoor terrain features
- Incorporates explicit stair detection and terrain classification into policy observations
- Handles smooth floor, carpet, stair, ledge, and ramp surfaces in a unified policy
- Demonstrates robust stair climbing and descent on real quadruped hardware
- Introduces terrain-aware reward shaping for indoor-specific locomotion challenges
- Achieves surface-type transitions (e.g., floor to stairs) without separate mode switching
- Provides detailed sim-to-real transfer results for indoor deployment scenarios

## Methodology Deep-Dive
The observation space design is central to the approach. Beyond standard proprioceptive observations (joint angles, velocities, IMU orientation, body velocity), the policy receives terrain-aware inputs: a local height map from depth sensors, terrain classification features (floor/stairs/ramp/ledge), and stair-specific parameters (step height, step depth, number of remaining steps). These terrain features are extracted from onboard sensors using lightweight perception modules that run in real-time.

The RL training uses PPO in Isaac Gym with procedurally generated indoor environments. Training environments include hallways, rooms with varying floor types, staircases with different dimensions, door thresholds, and ramp transitions. Domain randomization covers surface friction (0.3-1.0), stair dimensions (height: 10-20cm, depth: 20-35cm), robot mass (±15%), motor strength (±20%), and sensor noise. The curriculum starts with flat floors and progressively introduces more challenging indoor features.

Reward shaping is tailored for indoor locomotion. Key reward components include: velocity tracking (following desired velocity commands), energy efficiency (penalizing unnecessary joint torques—important for indoor operation where battery life matters), foot clearance over obstacles (encouraging high stepping for stair climbing), body orientation stability (penalizing excessive pitch/roll especially on stairs), and surface-appropriate gait (encouraging slower, more stable gaits on stairs vs. faster gaits on flat floors). A terrain-adaptive reward weighting increases stair-related rewards when stair features are detected in observations.

The stair-specific training protocol deserves special attention. Stair climbing is decomposed into approach, ascent, and descent phases. During approach, the policy must detect stairs and adjust gait preemptively. During ascent/descent, foot placement accuracy becomes critical. The training curriculum introduces stairs gradually: first low single steps, then multiple steps of uniform height, then varied-height stairs, and finally stairs with worn or uneven edges. A contact-based reward encourages full foot contact on stair surfaces to prevent slipping.

Sim-to-real transfer uses the standard domain randomization approach supplemented with actuator network adaptation. The actuator network models the real motor dynamics (including backlash, friction, and thermal effects) and is fine-tuned with a small amount of real-world motor data before deployment. System identification of the floor friction is performed using a short calibration walk before autonomous operation.

## Key Results & Numbers
- Successful stair climbing on stairs with 12-18cm step height, 25-30cm depth
- Stair descent success rate: 85% on standard indoor staircases
- Flat floor velocity tracking: <0.1 m/s error at walking speeds (0.3-0.8 m/s)
- Surface type transitions handled within 2-3 steps without falling
- Real-world indoor navigation tested over 500+ meters of mixed terrain
- Energy efficiency improved 15% over non-terrain-aware baselines on flat floors
- Sim-to-real transfer success rate: 90%+ for flat floors, 75% for stairs on first deployment
- Training time: ~4 hours on Isaac Gym with 4096 parallel environments

## Relevance to Project A — Mini Cheetah Quadruped RL
**Rating: High**
This paper is directly applicable to Mini Cheetah locomotion in building environments. The indoor-specific terrain observation design (height maps, stair parameters) can be incorporated into the MuJoCo training pipeline. The terrain-aware reward shaping provides a template for curriculum learning rewards specific to indoor deployment scenarios. The stair climbing methodology—decomposed into approach, ascent, and descent with contact-based rewards—directly applies to Mini Cheetah's 12 DoF controller. The sim-to-real transfer techniques (actuator network, friction calibration) complement the existing domain randomization strategy. The indoor curriculum progression (flat→threshold→ramp→stairs) maps to the curriculum learning framework.

## Relevance to Project B — Cassie Bipedal Hierarchical RL
**Rating: Medium**
While focused on quadrupeds, the indoor terrain classification and observation design are applicable to Cassie's deployment in building environments. The stair detection and parameterization can inform Cassie's CPTE (Contrastive Pre-trained Terrain Encoder) design for indoor terrains. The decomposed stair-climbing approach (approach/ascent/descent) maps to the Primitives level where terrain-specific locomotion skills are needed. The terrain-adaptive reward weighting concept could be applied across Cassie's hierarchy levels. However, bipedal stair climbing has fundamentally different dynamics (single-support phases, higher center of mass), so the specific reward shapes would need significant modification.

## What to Borrow / Implement
- Incorporate stair detection features into Mini Cheetah observation space
- Adopt the terrain-aware reward shaping framework for indoor curriculum learning
- Use the stair-climbing curriculum (flat→threshold→ramp→stairs) for progressive training
- Apply contact-based stair rewards for stable foot placement during climbing
- Implement actuator network adaptation for improved sim-to-real transfer
- Use terrain classification features to inform CPTE design in Project B
- Adapt the surface-appropriate gait concept for velocity-dependent gait selection

## Limitations & Open Questions
- Stair descent success rate (85%) leaves room for improvement, especially on steep stairs
- Depth sensor dependency may fail in poor lighting conditions common indoors
- No handling of dynamic obstacles (people, doors closing) common in indoor environments
- Terrain classification accuracy not quantified; errors could lead to inappropriate gaits
- Open question: How to handle transparent/reflective surfaces (glass doors, polished floors) that confuse depth sensors?
- Battery life during continuous stair climbing not evaluated
- No multi-floor navigation planning; focuses only on locomotion control
