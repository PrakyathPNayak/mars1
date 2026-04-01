# MIT Mini Cheetah RL Locomotion Environment

Complete deep reinforcement learning environment for the MIT Mini Cheetah quadruped robot.
Train locomotion policies with PPO, control via keyboard, and deploy autonomous exploration.

## Quick Start

```bash
# Install dependencies
make install

# Generate robot model (MuJoCo MJCF)
make model

# Run tests
make test

# Interactive keyboard-controlled demo
make demo

# Train a PPO policy (5M steps, ~8 parallel envs)
make train

# Quick training run (500K steps)
make train-quick

# Evaluate best checkpoint
make eval

# Generate training dashboard plot
make dashboard
```

## Features

- **MuJoCo physics** with accurate Mini Cheetah model (12 DoF, PD control at 500 Hz)
- **Keyboard control**: walk, trot, run, jump, crouch, turn, strafe
- **Autonomous exploration**: heading-based and waypoint navigation
- **PPO training pipeline** with domain randomization and curriculum learning
- **Visualization**: MuJoCo viewer, video recording, training dashboard

## Keyboard Controls

| Key | Action |
|-----|--------|
| W / ↑ | Forward |
| S / ↓ | Backward |
| A / ← | Strafe left |
| D / → | Strafe right |
| Q / E | Turn left / right |
| SHIFT + dir | Run speed |
| CTRL | Toggle crouch |
| J | Jump |
| SPACE | Stop |
| 1 / 2 / 3 | Walk / Trot / Run mode |
| X | Toggle exploration mode |

## Exploration Mode

```python
from src.control.exploration_policy import ExplorationPolicy
import math

policy = ExplorationPolicy()
policy.set_target_heading(math.pi / 4)   # Navigate 45° NE
vx, vy, wz = policy.get_command(current_yaw)

# Or waypoint-based:
policy.set_target_waypoint(5.0, 3.0, current_x=0.0, current_y=0.0)
```

## Research Foundation

Design choices synthesized from:
- **DreamWaQ** (2023) — proprioceptive blind locomotion
- **RMA** (Kumar et al. 2021) — rapid motor adaptation
- **Hwangbo et al. 2019** — ANYmal sim-to-real transfer
- **Kim et al. 2019** — MIT Cheetah 3 hardware and control

See [.state/RESEARCH_INSIGHTS.md](.state/RESEARCH_INSIGHTS.md) for full synthesis.

## Reward Design

The `AdvancedTerrainEnv` uses a 14-term weighted reward function, designed following
legged_gym (Rudin et al. 2022), DreamWaQ, and CHRL conventions. All velocity-based
terms are computed in **body frame** for sim-to-real consistency.

| Term | Weight | Description |
|------|--------|-------------|
| `lin_vel_tracking` | +1.0 | Exponential kernel tracking of body-frame vx/vy vs command |
| `ang_vel_tracking` | +0.5 | Exponential kernel tracking of yaw rate vs command |
| `height` | -1.0 | Squared deviation from skill-dependent height target |
| `orientation` | -1.0 | Projected gravity xy² (penalises tilt) |
| `lin_vel_z` | -2.0 | Vertical velocity² (suppresses bouncing) |
| `ang_vel_xy` | -0.05 | Body-frame roll/pitch rate² |
| `torque` | -2e-5 | Torque² × skill energy weight |
| `action_rate` | -0.02 | Action delta² (smoothness) |
| `joint_acc` | -2.5e-7 | Joint acceleration² (suppresses jitter) |
| `joint_limit` | -1.0 | Proximity to joint limits (0.1 rad margin) |
| `contact` | -0.2 | Gait-dependent foot contact pattern penalty |
| `terrain` | +0.2 | Forward velocity × terrain difficulty |
| `collision` | -0.5 | Trunk/thigh touching ground |
| `stumble` | -0.5 | Foot lateral hit on non-floor obstacle |

Net reward at convergence (trot, 1.5 m/s, flat): ~1.5–2.0 per step.

## Training Configuration

Hardware-tuned config in [`training_config.json`](training_config.json):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parallel envs | 64 | Saturates 24 CPU cores via SubprocVecEnv |
| Batch size | 8192 | 2× n_steps for stable gradient estimates |
| n_steps | 4096 | ~82s rollout per env (2048 steps × 0.02s × 2) |
| Learning rate | 3e-4 → 1e-5 | Cosine decay schedule |
| Entropy coef | 0.01 → 0.001 | Linear decay for exploration → exploitation |
| Episode length | 2000 steps | 40s at 50 Hz control rate |
| Push magnitude | 50 N | ~5.2 m/s² for 9.57 kg robot (CHRL: up to 80 N) |
| PD gains | kp=80, kd=2.0 | Kim et al. 2019; kd raised from 1.0 to suppress oscillation |
| Max torque | 17 Nm | Mini Cheetah hardware limit |

Domain randomization ranges:
- Body mass ±15%, floor friction 0.3–1.5, foot friction 0.8–2.0
- Joint damping ±20%, armature ±20%, PD gains ±15%, motor strength ±10%
- Observation noise σ=0.02

## Documentation

- [Architecture](docs/architecture.md) — detailed system design
- [Design Decisions](.state/DECISIONS.md) — rationale for all choices
- [Research Insights](.state/RESEARCH_INSIGHTS.md) — paper synthesis

## Training Results

Trained with PPO for 1M timesteps (8 parallel envs, ~17 min on CPU):

| Model | Mean Reward | Std | Survival Rate | Mean Ep Length |
|-------|------------|-----|---------------|----------------|
| **Best checkpoint** (200K steps) | **3112** | 455 | **100%** | 1001 |
| Final model (1M steps) | 2168 | 585 | 90% | 930 |

Best checkpoint evaluated over 20 episodes with `randomize_domain=False`.
Video recording available at `logs/best_model_rollout.mp4`.

```bash
# Evaluate a checkpoint
python3 scripts/evaluate.py --checkpoint checkpoints/best/best_model.zip --episodes 20

# Record a video
python3 scripts/record_video.py --checkpoint checkpoints/best/best_model.zip --output logs/rollout.mp4
```
