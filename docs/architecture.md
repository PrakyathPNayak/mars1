# MIT Mini Cheetah RL Locomotion — Architecture

## Overview
Complete RL environment for training and deploying locomotion policies
on the MIT Mini Cheetah quadruped. Supports keyboard control, autonomous
exploration, and sim-to-real transfer via domain randomization.

## Package Structure
```
.
├── run.py                       # Main entrypoint
├── Makefile                     # Convenience targets
├── assets/
│   ├── mini_cheetah.xml         # MuJoCo MJCF model
│   └── mini_cheetah_params.json # Physical parameters
├── src/
│   ├── env/
│   │   └── cheetah_env.py       # Gymnasium environment
│   ├── control/
│   │   ├── keyboard_controller.py
│   │   └── exploration_policy.py
│   ├── training/
│   │   ├── train.py             # PPO training loop
│   │   └── curriculum.py        # Terrain curriculum
│   └── visualization/
│       ├── viewer.py            # Interactive demo
│       └── stats_dashboard.py   # Training dashboard
├── tests/
│   └── test_env.py
├── scripts/
│   ├── generate_model.py        # MJCF model generator
│   └── evaluate.py              # Policy evaluation
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard + monitor logs
└── .state/                      # Session tracking
    ├── RESEARCH_INSIGHTS.md
    ├── DECISIONS.md
    └── PROGRESS.md
```

## Architectures

### Baseline: MLP Policy (train.py)
- Network: Actor/Critic MLP [512, 256, 128] with tanh activation
- PPO: clip=0.2, lr=3e-4, batch=256, epochs=10, gamma=0.99

### Advanced: Hierarchical Transformer + MoE (train_advanced.py)
Paper-inspired architecture combining insights from 55+ papers:

| Component | Inspiration | What it does |
|-----------|------------|--------------|
| Sensory Group Encoder | MSTA (2409.03332) | Encodes joint/vel/IMU/cmd groups independently, fuses via cross-attention |
| Temporal Transformer | SET (2410.13496), TERT (2212.07740) | 3-layer causal transformer over 16-step history |
| Morphological Symmetry | MS-PPO (2512.00727) | Bilateral leg reflection with precomputed indices |
| Mixture of Experts | MoE-Loco (2503.08564) | 4 terrain-specialized experts with load-balanced gating |
| World Model Head | DWL (2408.14472) | Auxiliary next-state prediction for representation quality |
| Adaptive Curriculum | LP-ACRL (2601.17428) | Learning progress-based terrain difficulty adjustment |
| Obs Normalization | Standard practice | Running mean/std normalization (Welford's algorithm) |
| Action Smoothing | DiffuseLoco (2404.19264) | EMA filter (α=0.8) for temporal action coherence |
| LR Warmup+Cosine | TERT, SET | 5% linear warmup then cosine decay |

**Parameters:** ~757K trainable (vs ~580K baseline MLP)
**Files:** `src/training/advanced_policy.py`, `src/training/sb3_integration.py`, `src/training/train_advanced.py`

## Observation Space (48 dims)
| Range   | Content                               |
|---------|---------------------------------------|
| [0:12]  | Joint positions (rad)                 |
| [12:24] | Joint velocities (rad/s)              |
| [24:27] | Base linear velocity (m/s, body frame)|
| [27:30] | Base angular velocity (rad/s, body)   |
| [30:33] | Gravity vector in body frame          |
| [33:45] | Previous action                       |
| [45:48] | Velocity command (vx, vy, wz)         |

## Action Space (12 dims)
Delta joint positions from default stance, clipped to ±0.5 rad.
Applied via PD controller (kp=80, kd=1) at 500 Hz physics simulation.

## Reward Function
- **Velocity tracking**: `exp(-||v_err||² / 0.25)` for x and y
- **Yaw tracking**: `exp(-||w_err||² / 0.25)`
- **Torque penalty**: `−0.0001 × ||τ||²`
- **Smoothness**: `−0.01 × ||Δa||²`
- **Upright bonus**: `dot(gravity_body_z, -1)`
- **Survival**: `+1.0 per step`

## Training (PPO)
- Network: Actor/Critic MLP [512, 256, 128] with tanh activation
- PPO: clip=0.2, lr=3e-4, batch=256, epochs=10, gamma=0.99
- 8 parallel environments, 2048 steps per rollout
- Domain randomization: mass ±20%, friction 0.5-1.5x, obs noise 2%

## Physical Parameters (MIT Mini Cheetah)
- Total mass: ~9 kg, Body: 0.40 × 0.10 × 0.05 m
- Upper leg: 0.209 m, Lower leg: 0.175 m
- 12 actuated DoF: 3 per leg (abduction, hip flex, knee)
- Max joint torque: 17 Nm
