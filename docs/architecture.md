# MIT Mini Cheetah RL Locomotion — Architecture

## Overview
Complete RL environment for training and deploying locomotion policies
on the MIT Mini Cheetah quadruped. Supports keyboard control, autonomous
exploration, and sim-to-real transfer via domain randomization.

## Package Structure
```
.
├── run.py                       # Main entrypoint
├── Makefile                     # Convenience targets (test, test-full, report)
├── assets/
│   ├── mini_cheetah.xml         # MuJoCo MJCF model (capsule collision geometry)
│   └── mini_cheetah_params.json # Physical parameters
├── src/
│   ├── env/
│   │   ├── cheetah_env.py       # Base Gymnasium environment (48-dim obs)
│   │   └── terrain_env.py       # Advanced terrain environment (57-dim obs)
│   ├── control/
│   │   ├── keyboard_controller.py
│   │   └── exploration_policy.py
│   ├── training/
│   │   ├── train.py             # MLP baseline training
│   │   ├── train_advanced.py    # Transformer training on base env
│   │   ├── advanced_policy.py   # Hierarchical Transformer + MoE + CPG + Terrain
│   │   ├── sb3_integration.py   # SB3 wrappers for transformer policy
│   │   └── curriculum.py        # TerrainCurriculum + AdvancedTerrainCurriculum
│   └── visualization/
│       ├── viewer.py            # Interactive demo
│       └── stats_dashboard.py   # Training dashboard
├── scripts/
│   ├── train_terrain.py         # Terrain-aware transformer training
│   ├── generate_model.py        # MJCF model generator
│   └── evaluate.py              # Policy evaluation
├── tests/
│   ├── test_env.py              # Basic environment tests
│   ├── test_final_validation.py # Transformer validation
│   └── test_suite.py            # Comprehensive test suite (39 tests + HTML report)
├── reports/                     # Auto-generated test reports (HTML + JSON)
├── checkpoints/                 # Saved model weights
└── logs/                        # TensorBoard + monitor logs
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
| Gait Phase Oscillator | DeepGait/CPG | Learned periodic leg phases with speed modulation |
| Terrain Estimator | DreamWaQ++, GenTe | Conv1d proprioceptive terrain inference from IMU+joint history |
| Contrastive Temporal Head | CPC/BYOL | Auxiliary temporal coherence loss for representation learning |
| Privileged Encoder | TERT, DreamWaQ++ | Sim-to-real teacher-student distillation pipeline |
| Feature Fusion | Multi-modal fusion | Combines transformer + terrain + phase features |

**Parameters:** ~968K standalone, ~821K SB3 integrated
**Files:** `src/training/advanced_policy.py`, `src/training/sb3_integration.py`, `src/training/train_advanced.py`

## Environments

### Base Environment: MiniCheetahEnv (cheetah_env.py)
- Observation: 54 dimensions (49 base + 5 skill one-hot)
- Flat ground, velocity command tracking, domain randomization
- 5 skill modes: stand, walk, run, crouch, jump

### Terrain Environment: AdvancedTerrainEnv (terrain_env.py)
- Observation: 57 dimensions (base 48 + 8 terrain encoding + 1 phase)
- Procedural terrain: flat, rough, slopes, stairs, gaps, stepping stones, random blocks, mixed
- Multi-skill: walk, trot, run, jump, crouch, stand (each with tuned reward profile)
- Foot contact detection, push perturbations, skill-dependent rewards
- Curriculum: AdvancedTerrainCurriculum (terrain type + difficulty + skill progression)

## Observation Space (54 base / 57 terrain)
| Range   | Content                               |
|---------|---------------------------------------|
| [0:12]  | Joint positions (rad)                 |
| [12:24] | Joint velocities (rad/s)              |
| [24:27] | Base linear velocity (m/s, body frame)|
| [27:30] | Base angular velocity (rad/s, body)   |
| [30:33] | Gravity vector in body frame          |
| [33:45] | Previous action                       |
| [45:49] | Command (vx, vy, wz, target_height)   |
| [49:54] | Skill one-hot encoding (5 modes)      |
| [48:56] | Terrain encoding (terrain env only)   |
| [56]    | Episode phase (terrain env only)      |

## Action Space (12 dims)
Delta joint positions from default stance, clipped to ±0.5 rad.
Applied via PD controller (kp=100, kd=0; MuJoCo joint damping=2)
at 500 Hz physics (n_substeps=10 at 50 Hz control).

## Reward Function (v6)

Reward v6 — Gait-focused redesign based on training log analysis at 10M steps.
Informed by legged_gym (Rudin 2022), Walk These Ways (Margolis 2023),
DreamWaQ (Nahrendra 2023), and SLIP (Spring-Loaded Inverted Pendulum) models.

### Key v6 Changes from v5
| Issue (from 10M-step logs) | Root Cause | Fix |
|-------|-----------|-----|
| r_smooth = -0.15 to -0.19/step (hurting gait) | L2 penalty (-0.05) quadratically punishes big leg swings | L1 penalty at -0.01 (5× reduction, linear) |
| r_gait only 0.35-0.53/step | clearance_target=5cm too low; binary trot symmetry; weak sub-components | clearance→8cm, sqrt trot symmetry, +stride_freq sub-reward, scale 2.0→3.5 |
| std frozen at 0.637 throughout training | LR decayed to 1.4e-6; log_std_init=-0.5 already close to final value | log_std_init=-1.0, LR floor=1e-5 |
| clip_fraction → 0 at end of training | LR linear decay to 0 | Added min_lr=1e-5 floor |
| Only 40 gradient steps per rollout | batch_size=4096, n_epochs=5 | batch_size=2048, n_epochs=10 → 160 grad steps/rollout |
| r_dof_vel constant -0.315/step | Scale -2e-4 too aggressive for joint oscillation | Halved to -1e-4 |

### Reward Components (13 terms + survival multiplier)

| Component | Base Scale | Formula | Purpose |
|-----------|-----------|---------|---------|
| r_linvel | 5.0 | exp(-‖v_xy_err‖²/0.25) | Velocity tracking |
| r_yaw | 1.5 | exp(-w_z_err²/0.25) | Yaw rate tracking |
| r_gait | 3.5 | 0.3×air_time + 0.25×trot_sym + 0.25×clearance + 0.2×stride_freq | Combined gait quality |
| r_posture | 2.0 | exp(-joint_err/0.5) | Joint angle targets per mode |
| r_body_height | 2.0 | exp(-Δh²/0.02) − 1.0 | Height tracking (0 at target) |
| r_stillness | 1.0 | exp(-(joint²×0.001+body²)/0.5) | Stand/crouch stillness |
| r_jump_phase | 3.0 | FSM: crouch→launch→airborne→land | Jump behavior |
| r_orientation | −1.0 | Σ(gravity_body_xy²) | Anti-tilt |
| r_torque | −5e-6 | Σ(tau²) | Energy |
| r_smooth | −0.01 | Σ\|Δaction\| (L1) | Action rate (was L2 at -0.05) |
| r_joint_limit | −2.0 | Proximity squared | Joint limits |
| r_lin_vel_z | −0.5 | vz² | Anti-bounce |
| r_dof_vel | −1e-4 | Σ(joint_vel²) | Joint velocity (halved) |

**Survival multiplier**: total_reward × min(1 + √(t/T) × 2, 3.0)

### Gait Sub-Components (v6)
| Sub-Component | Weight | Description |
|---------------|--------|-------------|
| Air time reward | 0.30 | Σ(air_time − 0.15s) at first contact |
| Trot symmetry | 0.25 | √(\|diag_FR+RL − diag_FL+RR\| / 2) — soft gradient |
| Foot clearance | 0.25 | mean(swing_height / 0.08m) — raised from 5cm |
| Stride frequency | 0.20 | exp(-0.5×(n_touchdowns − 2)²) — peaks at trot cadence |

### Per-Mode Reward Multipliers (Walk These Ways style)
| Mode | r_linvel | r_gait | r_posture | r_smooth | r_stillness |
|------|----------|--------|-----------|----------|-------------|
| stand | 0.3 | 0.0 | 3.0 | 1.0 | 3.0 |
| walk | 1.2 | 2.0 | 1.0 | 1.0 | 0.0 |
| run | 2.0 | 1.5 | 0.5 | 0.5 | 0.0 |
| crouch | 0.3 | 0.0 | 3.0 | 1.0 | 2.0 |
| jump | 0.2 | 0.0 | 0.5 | 1.0 | 0.0 |

### Command Mode Randomization (v5+)
During training, `command_mode` is randomly sampled at each reset and
every 200 steps (4s at 50 Hz) with mode-appropriate velocity ranges:
- stand (20%): 50% zero velocity, 50% low velocity
- walk (25%): moderate velocity (0.2–1.0 m/s)
- run (20%): high velocity only (1.0–2.5 m/s)
- crouch (15%): low velocity only (−0.1–0.2 m/s)
- jump (20%): forward (0.0–0.5 m/s)

### Diagnostic Validation
With zero action (200 steps):
- Stand mode: +5.30/step (r_stand_still now 16% of total, was 1.7%)
- Crouch mode: +0.83/step at wrong posture → +5.58/step at correct posture
- Oscillation (±0.2): 0.0/step (clipped; r_dof_vel = −98/step dominates)
- Walking: r_dof_vel = −0.06/step (1600× less than oscillation)

## Training (PPO)
- Network: Actor/Critic MLP [2048, 1024, 512] with ELU activation
- PPO: clip=0.2, lr=3e-4 → min 1e-5 (linear decay with floor), batch=2048, epochs=10, gamma=0.99
- 8 parallel environments, 4096 steps per rollout
- Gradient budget: 16 minibatches × 10 epochs = 160 grad steps/rollout → ~48.8K total at 10M steps
- log_std_init=-1.0 (initial std≈0.37)
- Domain randomization: mass ±15%, friction 0.8–2.0×, obs noise 2%
- MLP expert: 10M steps default
- Hierarchical (BC + Transformer PPO): 100 BC epochs + 15M PPO steps

## Meta-Learning Assessment

Meta-learning approaches (MAML, RMA) were evaluated for applicability:

| Approach | Applicability | Status |
|----------|-------------|--------|
| MAML (Finn 2017) | Not suited — task distribution too narrow (single robot) | Not implemented |
| RMA (Kumar 2021) | Adaptation module useful but Transformer temporal context provides similar functionality via history encoding | Partially covered by existing architecture |
| Learned reward shaping | Automatic reward scale tuning could help but adds complexity | Future work — hyperparameter sweep preferred |
| Domain randomization | Already implemented; serves as implicit meta-training across dynamics variations | ✅ Implemented |

The current architecture already provides meta-learning benefits: the Transformer's
16-step temporal context acts as an implicit adaptation module (similar to RMA's
adaptation head), and domain randomization during training ensures robustness to
parameter variation. Full MAML or learned-objective approaches would add complexity
without clear benefit given the single-robot, single-terrain setup.

## Physical Parameters (Unitree Go1)
- Total mass: ~12.9 kg (trunk: 5.204 kg, 4 legs: ~7.7 kg)
- Trunk: 0.3762 × 0.0935 × 0.114 m
- Thigh: 0.213 m, Calf+foot: 0.213 m (total leg: ~0.426 m)
- Standing height: 0.27 m (from go1.xml worldbody position)
- 12 actuated DoF: 3 per leg (abduction ±0.863 rad, hip −0.686–4.501, knee −2.818–−0.888)
- Max motor torque: 33.5 Nm (ctrlrange)
- Joint damping: 2 Ns/m, armature: 0.01, frictionloss: 0.2
