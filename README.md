# Unitree Go1 RL Locomotion Environment

Deep reinforcement learning environment for the **Unitree Go1** quadruped (MuJoCo Menagerie).
Trains locomotion policies with PPO via a two-stage hierarchical pipeline:
MLP expert → Transformer+MoE fine-tuning.

## Quick Start

```bash
# Install dependencies
just install        # or: pip install -r requirements.txt

# Run tests (10 core tests, ~8s)
just test

# Smoke-test the full pipeline (~5 min CPU)
just train-quick

# Full training pipeline (MLP 3M steps → Hierarchical 10M steps)
just train

# Evaluate best checkpoint (20 episodes)
just eval

# Interactive keyboard demo
just demo
```

> **Prerequisite:** [just](https://just.systems) — `cargo install just` (or see install docs).
> Python 3.10+ and `pip install gymnasium stable-baselines3[extra] mujoco` are required.

## Training Pipeline

Training runs in two stages:

```
Stage 1 — MLP PPO expert
    src/training/train.py
    Network: MLP [2048, 1024, 512], ELU, n_epochs=10
    Duration: 3M env steps (8 parallel envs)
    Output: checkpoints/mlp_best.zip, vec_normalize.pkl

Stage 2 — Hierarchical BC → Transformer+MoE PPO
    src/training/train_hierarchical.py
    Phase 1: Collect expert rollouts (SubprocVecEnv, 4 workers)
    Phase 2: Behavioural cloning on Transformer encoder (100 epochs, GPU)
    Phase 3: PPO fine-tuning (n_epochs=15, 10M steps)
    Output: checkpoints/hierarchical_best.zip

Pipeline orchestrator: scripts/pipeline.py
    — Runs Stage 1 then Stage 2 automatically
    — All artifacts saved to runs/{run_id}/
```

### Just Recipes

```bash
just train                   # Full pipeline (3M + 10M steps)
just train-gpu               # Same, on GPU
just train-quick             # Smoke test (200K + 500K steps, ~5 min)
just train-mlp               # Stage 1 only — MLP PPO
just train-hier              # Stage 2 only — needs --expert path
just train-hier-only expert=path/to/expert.zip   # Skip Stage 1

# Custom step counts
just train-steps mlp_steps=5000000 hier_steps=15000000

# Advanced CLI
python3 scripts/pipeline.py --mlp-steps 3000000 --mlp-epochs 10 \
    --hier-steps 10000000 --hier-epochs 15 --bc-epochs 100 --device auto
```

All pipeline outputs go to `runs/{run_id}/`:
```
runs/20260403_120000/
├── mlp_final.zip
├── mlp_best.zip
├── mlp_vec_normalize.pkl
├── hierarchical_final.zip
├── hierarchical_best.zip
└── training_summary.json
```

### Direct Training Scripts

```bash
# MLP PPO
python3 src/training/train.py --total-steps 3000000 --n-envs 8 --n-epochs 10

# Hierarchical
python3 src/training/train_hierarchical.py \
    --expert checkpoints/best/best_model.zip \
    --bc-epochs 100 \
    --total-steps 10000000 \
    --n-epochs 15

# Advanced single-env Transformer
python3 src/training/train_advanced.py --total-steps 5000000
```

## Evaluation & Visualization

```bash
# Evaluate a checkpoint (20 episodes, prints mean reward + survival)
just eval
just eval checkpoint=runs/20260403_120000/hierarchical_best.zip

# Interactive demo — MuJoCo viewer + keyboard control
just demo
just demo checkpoint=runs/20260403_120000/hierarchical_best.zip

# Demo without policy (keyboard → pure physics)
just demo-no-policy

# Record video
just record output=logs/rollout.mp4

# Reward component diagnostic (prints per-term breakdown, 200 steps)
just diagnose mode=stand
just diagnose mode=crouch

# TensorBoard
just tensorboard logdir=runs/20260403_120000/logs_mlp
```

## Keyboard Controls (demo mode)

| Key | Action |
|-----|--------|
| W / ↑ | Forward |
| S / ↓ | Backward |
| A / ← | Strafe left |
| D / → | Strafe right |
| Q / E | Turn left / right |
| 1 / 2 / 3 | Walk / Trot / Run speed |
| C | Toggle crouch (low body height) |
| J | Jump |
| SPACE | Stop motion / clear command |
| ESC ESC | Quit (press twice within 500ms) |

See [INTERACTIVE_GUIDE.md](INTERACTIVE_GUIDE.md) for terrain selection,
difficulty levels, and the full list of supported terrain types
(`flat`, `rough`, `slope_up`, `slope_down`, `stairs_up`, `stairs_down`,
`gaps`, `stepping_stones`, `random_blocks`, `mixed`).

## Reward Design (v23)

22-component reward function for the Go1 (see `REWARD_SCALES` in
`src/env/cheetah_env.py`). Per-mode multipliers in `MODE_REWARD_MULTIPLIERS`
swap which terms are active for `stand`, `walk`, `run`, `jump`. A final
`MODE_REWARD_SCALE` (stand=0.40, walk/run=1.0, jump=0.35) equalises return
magnitudes across modes (v31p).

**Tracking (exp kernel `exp(-err²/σ²)`, command-active gated):**
- `r_vel_x` × 2.0 — forward velocity tracking, σ adaptive
- `r_vel_y` × 2.0 — lateral velocity tracking
- `r_yaw` × 2.5 — yaw rate tracking
- `r_fwd_vel` × 10.0 — EMA-smoothed forward speed bonus

**Gait & posture:**
- `r_gait` × 2.5 — air-time + diagonal symmetry + clearance + stride freq
- `r_posture` × 1.0 — height-interpolated hip/knee targets
- `r_body_height` × 1.5 — terrain-relative trunk height (key for slopes)
- `r_foot_clearance` × 0.5 — swing foot above local terrain (0.08m target)

**Penalties (bounded or small):**
- `r_orientation` × −2.0 — terrain-normal-relative tilt
- `r_joint_limit` × −10.0 — soft barrier at 0.1 rad from limits
- `r_lin_vel_z` × −2.0 — vertical bounce
- `r_smooth` × −0.02 — action L2 rate
- `r_standstill` × −3.0 — quadratic on EMA speed deficit

**Mode-specific:**
- `r_stillness` (stand only) — joint + body motion penalty
- `r_jump_phase` × 5.0 (jump only) — crouch/launch/flight/landing FSM
- `r_terrain_progress` × 1.0 — velocity projected onto commanded direction

See [docs/architecture.md](docs/architecture.md) for the full table.

## Robot: Unitree Go1

| Parameter | Value |
|-----------|-------|
| XML model | `assets/go1.xml` (mujoco_menagerie) |
| Total mass | ~12.9 kg (trunk 5.2 kg) |
| Standing height | 0.27 m |
| Abduction range | ±0.863 rad |
| Hip range | −0.686 to 4.501 rad |
| Knee range | −2.818 to −0.888 rad |
| Max torque | 33.5 Nm |
| Control rate | 50 Hz (PD: kp=100, kd=0, joint damping=2) |

## Training Configuration

Full config in [`training_config.json`](training_config.json).

| Parameter | MLP Stage | Hierarchical Stage |
|-----------|-----------|-------------------|
| Total steps | 3M | 10M |
| n_envs | 8 | 8 |
| n_epochs | 10 | 15 |
| batch_size | 4096 | 256 |
| n_steps | 4096 | 2048 |
| BC epochs | — | 100 |
| Network | MLP [2048,1024,512] | Transformer d=256, 3L, 4-expert MoE |
| Observation | 196-dim | 196-dim × 16-step history |

## Project Structure

```
src/
├── env/
│   ├── cheetah_env.py       # Go1 Gymnasium environment (196-dim obs, reward v23)
│   └── terrain_env.py       # Terrain variant
├── training/
│   ├── train.py             # Stage 1: MLP PPO
│   ├── train_hierarchical.py# Stage 2: BC → Transformer+MoE PPO
│   ├── train_advanced.py    # Standalone Transformer training
│   ├── advanced_policy.py   # Transformer + MoE + CPG + World Model
│   └── sb3_integration.py   # SB3 policy wrappers
├── control/
│   ├── keyboard_controller.py
│   └── exploration_policy.py
scripts/
├── pipeline.py              # Full 2-stage training pipeline
├── diagnose_reward.py       # Reward component diagnostic
├── evaluate.py              # Policy evaluation
├── record_video.py          # Video rollout recording
└── interactive_control.py   # Keyboard → velocity commands
assets/
└── go1.xml                  # Unitree Go1 MuJoCo model
docs/
└── architecture.md          # Architecture + reward design detail
justfile                     # Task runner (replaces Makefile)
training_config.json         # Reference hyperparameters
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — detailed reward design, meta-learning assessment, Go1 parameters
- [training_config.json](training_config.json) — hyperparameter reference

## Development

```bash
# Run tests
just test

# Run full test suite with HTML report
just test-full           # → reports/test_report.html

# Clean up
just clean               # Remove __pycache__ / .pytest_cache
just clean-all           # Remove runs/, logs/ (preserves checkpoints/best/)
```

