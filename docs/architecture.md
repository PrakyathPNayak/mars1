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
- Observation: 48 dimensions
- Flat ground, velocity command tracking, domain randomization

### Terrain Environment: AdvancedTerrainEnv (terrain_env.py)
- Observation: 57 dimensions (base 48 + 8 terrain encoding + 1 phase)
- Procedural terrain: flat, rough, slopes, stairs, gaps, stepping stones, random blocks, mixed
- Multi-skill: walk, trot, run, jump, crouch, stand (each with tuned reward profile)
- Foot contact detection, push perturbations, skill-dependent rewards
- Curriculum: AdvancedTerrainCurriculum (terrain type + difficulty + skill progression)

## Observation Space (48 base / 57 terrain)
| Range   | Content                               |
|---------|---------------------------------------|
| [0:12]  | Joint positions (rad)                 |
| [12:24] | Joint velocities (rad/s)              |
| [24:27] | Base linear velocity (m/s, body frame)|
| [27:30] | Base angular velocity (rad/s, body)   |
| [30:33] | Gravity vector in body frame          |
| [33:45] | Previous action                       |
| [45:48] | Velocity command (vx, vy, wz)         |
| [48:56] | Terrain encoding (terrain env only)   |
| [56]    | Episode phase (terrain env only)      |

## Action Space (12 dims)
Delta joint positions from default stance, clipped to ±0.5 rad.
Applied via PD controller (kp=100, kd=0; MuJoCo joint damping=2)
at 500 Hz physics (n_substeps=10 at 50 Hz control).

## Reward Function (v3)

Reward design informed by legged_gym (Rudin 2022), Humanoid-Gym (Gu 2024),
and diagnostic analysis of stand/crouch failure modes.

### Key v3 Changes from v2
| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Shaking on stand | No joint velocity penalty | Added `r_dof_vel` (sum of joint_vel²) |
| Reward increased while shaking | Penalties too weak vs tracking reward | 6× `r_smooth`, 10× `r_body_acc`, 4× `r_action_jerk` |
| Crouch not working | r_body_height always positive (exp kernel) | Centered at 0: `exp(-err²/σ) - 1.0` |
| No crouch gradient | r_posture exp kernel too narrow (σ=0.5) | Flipped to negative squared-error penalty |
| Crouch conflicts | r_stand_still penalized joint deviation in crouch mode | Gated by `mode not in (crouch, jump)` |
| Model never sees crouch | command_mode always "stand" during training | Randomized mode every reset + resample |

### Positive Rewards (~6/step at convergence)
| Component | Scale | Formula | Purpose |
|-----------|-------|---------|---------|
| r_linvel | 4.0 | exp(-‖v_xy_err‖²/0.25) | Velocity tracking (combined xy) |
| r_yaw | 2.0 | exp(-w_z_err²/0.25) | Yaw rate tracking |
| r_feet_air_time | 2.0 | Σ(air_time − 0.15) @ first contact | Gait frequency |
| r_gait_phase | 1.5 | |diag1 − diag2| / 2 | Trot symmetry |
| r_foot_clearance | 0.5 | mean(swing_height / 0.05) | Foot lift |
| r_body_height | 2.0 | exp(-Δh²/0.02) − 1.0 | Height tracking (0 at target) |
| r_jump_phase | 1.0 | FSM phase reward | Jump behavior |
| r_alive | 0.2 | constant | Survival tie-breaker |

### Penalties
| Component | Scale | Formula | Purpose |
|-----------|-------|---------|---------|
| r_posture | −1.5 | Σ(hip_err² + knee_err²) | Joint angle targets per mode |
| r_dof_vel | −0.1 | Σ(joint_vel²) | **Anti-shake** (key addition) |
| r_stand_still | −3.0 | Σ|q − default| if standing, mode ≠ crouch | Stand stability |
| r_height | −5.0 | (base_z − target)² | Height penalty |
| r_orientation | −2.0 | Σ(gravity_body_xy²) | Anti-tilt |
| r_lin_vel_z | −2.0 | vz² | Anti-bounce |
| r_smooth | −0.3 | Σ(Δaction²) | Action rate (6× from v2) |
| r_body_acc | −0.005 | Σ(body_acc²) | Anti-shake (10× from v2) |
| r_action_jerk | −0.02 | Σ(action_jerk²) | 2nd-order smoothness (4× from v2) |
| r_cmd_vel_error | −1.0 | lin_vel_error | Velocity error |
| r_crouch_penalty | −5.0 | Height window detection | Anti-unwanted-crouch |
| r_joint_limit | −5.0 | Proximity squared | Joint limits |
| r_abduction | −2.0 | Σ(abd_q²) | Anti-splay |
| r_hip_excess | −3.0 | Σ(clip(hip − 1.3, 0)²) | Anti-belly-sit |
| r_torque | −5e-6 | Σ(tau²) | Energy |
| r_power | −1e-5 | Σ|tau·qvel| | Mechanical power |
| r_ang_vel_xy | −0.01 | Σ(wx² + wy²) | Roll/pitch rate |
| r_joint_acc | −5e-8 | Σ(joint_acc²) | Joint acceleration |
| r_foot_strike_vel | −0.005 | Σ(foot_z_vel²) @ contact | Impact |

### Command Mode Randomization (v3)
During training, `command_mode` is randomly sampled at each reset and
every 200 steps (4s at 50 Hz) with mode-appropriate velocity ranges:
- stand (15%): 50% zero velocity, 50% low velocity
- walk (15%), trot (25%), explore (10%): full range
- run (15%): high velocity only (0.5–2.0 m/s)
- crouch (20%): low velocity only (−0.1–0.3 m/s)

### Diagnostic Validation
With zero action (200 steps):
- Stand mode: +5.30/step (r_stand_still now 16% of total, was 1.7%)
- Crouch mode: +0.83/step at wrong posture → +5.58/step at correct posture
- Oscillation (±0.2): 0.0/step (clipped; r_dof_vel = −98/step dominates)
- Walking: r_dof_vel = −0.06/step (1600× less than oscillation)

## Training (PPO)
- Network: Actor/Critic MLP [2048, 1024, 512] with ELU activation
- PPO: clip=0.2, lr=3e-4 (linear decay), batch=4096, epochs=5, gamma=0.99
- 8 parallel environments, 4096 steps per rollout
- Domain randomization: mass ±15%, friction 0.5–1.5×, obs noise 2%
- MLP expert: 3M steps default
- Hierarchical (BC + Transformer PPO): 100 BC epochs + 10M PPO steps

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
