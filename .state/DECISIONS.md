# Design Decisions
All major design decisions with rationale.

## Simulator Choice
- **Choice**: MuJoCo 3.4.0
- **Reason**: MuJoCo installed via pip; best accuracy for legged locomotion, fast contact solving
- **Alternatives considered**: PyBullet (less accurate contacts), Isaac Gym (requires NVIDIA GPU + special install)

## RL Framework
- **Choice**: Stable-Baselines3 2.7.1 with PPO
- **Reason**: Mature, well-tested PPO implementation; easy vectorized envs; good SB3/Gymnasium integration
