# Research Insights — Synthesized

Synthesized from key papers on quadruped locomotion RL.

## Papers Analyzed
1. Kim et al. 2019 - "MIT Cheetah 3: Design and Control" (hardware, dynamics)
2. Kumar et al. 2021 - "RMA: Rapid Motor Adaptation" (sim2real, adaptation)
3. Lee et al. 2020 - "Learning Quadrupedal Locomotion over Challenging Terrain"
4. DreamWaQ 2023 - "Robust Quadrupedal Locomotion via DRL" (blind locomotion)
5. Hwangbo et al. 2019 - "Learning Agile Motor Skills for Legged Robots" (ANYmal)
6. Peng et al. 2020 - "Learning Agile Robotic Locomotion Skills from Animals"

## Observation Space (consensus: DreamWaQ, RMA, Hwangbo 2019)
- Joint positions (12 DoF)
- Joint velocities (12 DoF)
- Base linear velocity (3)
- Base angular velocity (3)
- Gravity vector in body frame (3)
- Previous action (12)
- Command (velocity x, velocity y, yaw rate) (3)
- **Total: 48 proprioceptive dims (no exteroception for baseline)**

## Action Space (consensus from multiple papers)
- Target joint positions (12 DoF, position control via PD)
- PD controller tracks targets at high frequency (500Hz physics, 50Hz policy)
- Action = delta from default stance, clipped to ±0.5 rad

## Reward Function (synthesized: DreamWaQ, Hwangbo, Kumar)
- Linear velocity tracking (x, y): exp(-||v_cmd - v_base||² / 0.25)
- Yaw rate tracking: exp(-||w_cmd - w_yaw||² / 0.25)
- Joint torque penalty: -0.0001 × Σ(τ²)
- Action smoothness: -0.01 × Σ((a_t - a_{t-1})²)
- Upright orientation bonus: dot(body_z, world_z)
- Survival bonus: +1.0 per step alive

## Network Architecture (consensus: MLP works well for proprioceptive)
- Actor: MLP [512, 256, 128] → tanh → 12
- Critic: MLP [512, 256, 128] → 1
- Optional: GRU(256) for history encoding

## Training Algorithm
- PPO with clip=0.2, entropy coeff=0.01
- Learning rate: 3e-4 with linear decay
- 8-24 parallel environments, 2048 steps per rollout
- Domain randomization: mass (±20%), friction (0.4–1.2), motor strength (±20%)

## Physical Parameters (MIT Mini Cheetah — Kim et al. 2019)
- Total mass: ~9 kg (body ~6 kg)
- Body dimensions: 0.40 × 0.10 × 0.05 m
- Upper leg (thigh): 0.209 m
- Lower leg (calf): 0.175 m
- Hip abduction range: ±30°
- Hip flexion range: ±60°
- Knee range: -154.5° to -30°
- Max joint torque: 17 Nm
- 12 actuated DoF (3 per leg: abduction, hip flex, knee)

## Locomotion Gaits
- Walk: 4-beat gait, ~0.5 m/s
- Trot: diagonal pairs, ~1.5 m/s (most stable for RL)
- Bound: front-rear pairs, ~2.5 m/s
- Gallop: fast asymmetric, ~3.0+ m/s

## Sim-to-Real Transfer
- Domain randomization on mass, friction, motor gains
- Observation noise injection (2% std)
- Action delay randomization (0-1 steps)
- Random pushes during training
