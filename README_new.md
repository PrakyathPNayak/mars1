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

## Documentation

- [Architecture](docs/architecture.md) — detailed system design
- [Design Decisions](.state/DECISIONS.md) — rationale for all choices
- [Research Insights](.state/RESEARCH_INSIGHTS.md) — paper synthesis
