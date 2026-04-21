# MARS Go1 RL Locomotion — Usage Guide

Comprehensive guide covering setup, training, evaluation, visualization, and how to pause/resume from any point.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Project Structure](#2-project-structure)
3. [Setup](#3-setup)
4. [Training](#4-training)
   - [Full Pipeline (Recommended)](#41-full-pipeline-recommended)
   - [Stage 1 Only — MLP PPO](#42-stage-1-only--mlp-ppo)
   - [Stage 2 Only — Hierarchical](#43-stage-2-only--hierarchical)
   - [Terrain Training](#44-terrain-training)
5. [Where Models Are Saved](#5-where-models-are-saved)
   - [Pipeline runs/](#51-pipeline-runs)
   - [Standalone train.py](#52-standalone-trainpy)
   - [Hierarchical](#53-hierarchical-trainpy)
   - [Terrain](#54-terrain-training)
6. [Pausing and Resuming](#6-pausing-and-resuming)
   - [Stopping safely](#61-stopping-safely)
   - [Resuming train.py](#62-resuming-trainpy)
   - [Resuming from pipeline mid-run](#63-resuming-from-pipeline-mid-run)
   - [Resuming Stage 2 only (skip-mlp)](#64-resuming-stage-2-only-skip-mlp)
   - [Resuming terrain training](#65-resuming-terrain-training)
7. [Evaluation](#7-evaluation)
8. [Interactive Control (Keyboard Demo)](#8-interactive-control-keyboard-demo)
9. [Recording Video](#9-recording-video)
10. [TensorBoard](#10-tensorboard)
11. [Reward Diagnostics](#11-reward-diagnostics)
12. [Training Configuration Reference](#12-training-configuration-reference)
13. [Checkpoint Reference](#13-checkpoint-reference)
14. [Common Workflows](#14-common-workflows)

---

## 1. Quick Start

```bash
# 1. Install dependencies
just install         # or: pip install -r requirements.txt

# 2. Run tests
just test-quick

# 3. Smoke-test training (tiny run — ~5 min)
just train-quick

# 4. Evaluate pretrained model
just eval

# 5. Interactive demo with keyboard
just control
```

---

## 2. Project Structure

```
miniproject/
├── run.py                       # Top-level dispatcher (train/eval/demo/explore)
├── justfile                     # All task recipes (use: just <recipe>)
├── Makefile                     # Legacy recipes (use justfile instead)
├── training_config.json         # Canonical hyperparameter reference
│
├── scripts/
│   ├── pipeline.py              # Full 2-stage training orchestrator ← main entry
│   ├── evaluate.py              # Policy evaluation + JSON report
│   ├── interactive_control.py   # Keyboard demo
│   ├── record_video.py          # MP4 rollout recorder
│   ├── train_terrain.py         # Terrain-specific training
│   └── diagnose_reward.py       # Reward component debugger
│
├── src/
│   ├── env/
│   │   ├── cheetah_env.py       # Core Go1 Gymnasium environment (Reward v7)
│   │   └── terrain_env.py       # Terrain-randomized environment
│   └── training/
│       ├── train.py             # Stage 1: MLP PPO training
│       ├── train_hierarchical.py# Stage 2: BC → Transformer+MoE PPO
│       ├── sb3_integration.py   # Policy classes, wrappers, callbacks
│       └── curriculum.py        # Terrain/skill curriculum
│
├── checkpoints/                 # Default output for standalone training
│   ├── best/best_model.zip      # Best eval model (auto-saved)
│   ├── vec_normalize.pkl        # VecNormalize stats (required for loading)
│   └── cheetah_ppo_*_steps.zip  # Periodic snapshots every 100K steps
│
├── runs/                        # Pipeline outputs (one folder per run)
│   └── {YYYYMMDD_HHMMSS}/       # e.g. runs/20260405_143000/
│       ├── mlp_final.zip
│       ├── mlp_best.zip
│       ├── hierarchical_final.zip
│       └── training_summary.json
│
└── logs/
    ├── training/                # TensorBoard + monitor CSVs
    └── eval_*.json              # Evaluation results
```

---

## 3. Setup

### Install dependencies

```bash
# With just
just install

# Manually
pip install gymnasium stable-baselines3[extra] tensorboard mujoco pynput imageio scipy matplotlib torch

# Dev extras (pytest, HTML reports)
just install-dev
```

### Activate virtualenv

```bash
source /home/prakyathpnayak/venv/bin/activate
```

### Verify

```bash
just test          # 10 unit tests
just test-full     # 39 tests (unit + integration + performance + regression)
```

---

## 4. Training

### 4.1 Full Pipeline (Recommended)

Runs Stage 1 (MLP PPO, 100M steps) then Stage 2 (BC warm-start + Transformer PPO, 50M steps) end-to-end. All outputs go to a single timestamped `runs/` folder.

```bash
# Default full run (CPU, i9-14900K)
python3 scripts/pipeline.py

# GPU
python3 scripts/pipeline.py --device cuda

# Custom run ID (makes the output folder easy to find)
python3 scripts/pipeline.py --run-id my_experiment

# Custom step counts
python3 scripts/pipeline.py --mlp-steps 50000000 --hier-steps 25000000

# Smoke test (fast, just validates everything runs)
python3 scripts/pipeline.py --mlp-steps 200000 --hier-steps 500000 --bc-epochs 5 --run-id smoke_test
```

**All pipeline args:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | `YYYYMMDD_HHMMSS` | Output folder name under `runs/` |
| `--mlp-steps` | 100,000,000 | Stage 1 total env steps |
| `--mlp-epochs` | 10 | PPO gradient epochs per rollout (Stage 1) |
| `--skip-mlp` | — | Skip Stage 1 entirely; requires `--expert` |
| `--expert` | — | Pre-trained `.zip` to use as Stage 2 expert |
| `--vec-normalize` | — | `vec_normalize.pkl` for a skipped Stage 1 |
| `--hier-steps` | 50,000,000 | Stage 2 total env steps |
| `--hier-epochs` | 10 | PPO gradient epochs per rollout (Stage 2) |
| `--n-expert-episodes` | 200 | BC data collection episodes |
| `--bc-epochs` | 100 | Behavioral cloning training epochs |
| `--bc-lr` | 5e-4 | BC learning rate |
| `--bc-batch` | 256 | BC batch size |
| `--d-model` | 256 | Transformer hidden dimension |
| `--n-layers` | 3 | Number of Transformer layers |
| `--n-experts` | 4 | MoE expert count |
| `--history-len` | 16 | Observation history window |
| `--n-envs` | 24 | Parallel environments |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--verbose` | — | Extra logging |

---

### 4.2 Stage 1 Only — MLP PPO

Trains the flat-terrain MLP policy independently. Output goes to `checkpoints/` by default.

```bash
python3 src/training/train.py \
    --total-steps 100000000 \
    --n-envs 24 \
    --n-epochs 10 \
    --device cpu

# Custom output dirs
python3 src/training/train.py \
    --total-steps 100000000 \
    --ckpt-dir my_run/checkpoints \
    --log-dir my_run/logs
```

**All train.py args:**

| Flag | Default | Description |
|------|---------|-------------|
| `--total-steps` | 100,000,000 | Total env steps |
| `--n-envs` | 24 | Parallel environments |
| `--n-epochs` | 10 | PPO gradient epochs per rollout |
| `--resume` | — | Path to `.zip` to continue from (see §6.2) |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--ckpt-dir` | `checkpoints` | Where to save model checkpoints |
| `--log-dir` | `logs/training` | Where to save TensorBoard / monitor logs |

---

### 4.3 Stage 2 Only — Hierarchical

Trains the Transformer + MoE policy using a pre-trained MLP policy as expert.

```bash
python3 src/training/train_hierarchical.py \
    --expert checkpoints/best/best_model.zip \
    --total-steps 50000000 \
    --n-envs 24

# With VecNormalize stats from Stage 1
python3 src/training/train_hierarchical.py \
    --expert checkpoints/best/best_model.zip \
    --vec-normalize checkpoints/vec_normalize.pkl \
    --total-steps 50000000

# GPU
python3 src/training/train_hierarchical.py \
    --expert checkpoints/best/best_model.zip \
    --device cuda
```

**Three internal phases** (run automatically, no user intervention needed):

1. **Expert data collection** — runs `--n-expert-episodes 200` episodes with the MLP expert
2. **Behavioral Cloning (BC)** — warms up Transformer weights from expert trajectories
3. **PPO fine-tuning** — full RL training with BC-initialized weights

**All train_hierarchical.py args:**

| Flag | Default | Description |
|------|---------|-------------|
| `--expert` | *(required)* | Path to Stage 1 `.zip` |
| `--vec-normalize` | — | VecNormalize stats `.pkl` |
| `--n-expert-episodes` | 200 | Episodes for BC data collection |
| `--bc-epochs` | 100 | BC training epochs |
| `--bc-lr` | 5e-4 | BC learning rate |
| `--bc-batch` | 256 | BC batch size |
| `--total-steps` | 50,000,000 | Stage 2 RL training steps |
| `--n-envs` | 24 | Parallel environments |
| `--n-epochs` | 10 | PPO epochs per rollout |
| `--d-model` | 256 | Transformer hidden dimension |
| `--n-layers` | 3 | Transformer layers |
| `--n-experts` | 4 | MoE expert count |
| `--history-len` | 16 | Obs history window |
| `--n-collect-envs` | 4 | Parallel envs for BC data collection |
| `--ckpt-dir` | `checkpoints/hierarchical` | Where to save checkpoints |
| `--log-dir` | `logs/training_hierarchical` | TensorBoard/monitor log dir |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--verbose` | — | Extra logging |

---

### 4.4 Terrain Training

Trains a Transformer policy on randomized terrain with curriculum learning.

```bash
# Full terrain run
python3 scripts/train_terrain.py --total-steps 5000000 --n-envs 8

# Quick smoke test
python3 scripts/train_terrain.py --quick
# (equivalent to: --total-steps 500000 --n-envs 4)
```

Outputs: `checkpoints/terrain/terrain_transformer_final.zip`, `terrain_config.json`

---

## 5. Where Models Are Saved

### 5.1 Pipeline `runs/`

Every pipeline run creates:

```
runs/{run_id}/
├── mlp_training/                        # Stage 1 working directory
│   ├── cheetah_ppo_100000_steps.zip     # snapshot every ~100K steps
│   ├── cheetah_ppo_200000_steps.zip
│   ├── ...
│   ├── best/
│   │   └── best_model.zip              # lowest eval loss so far
│   ├── vec_normalize.pkl
│   └── training_config.json
├── logs_mlp/
│   ├── PPO_1/events.out.tfevents.*     # TensorBoard data
│   ├── eval/evaluations.npz            # EvalCallback history
│   ├── monitor.csv                     # episode-level stats
│   └── reward_components.csv           # per-term reward breakdown
│
├── hierarchical_training/               # Stage 2 working directory
│   ├── hierarchical_100000_steps.zip
│   ├── ...
│   ├── best/
│   │   └── best_model.zip
│   └── training_config.json
├── logs_hierarchical/
│
├── mlp_final.zip                        # copy of Stage 1 final
├── mlp_best.zip                         # copy of Stage 1 best
├── mlp_vec_normalize.pkl                # copy of VecNormalize
├── hierarchical_final.zip               # copy of Stage 2 final
├── hierarchical_best.zip                # copy of Stage 2 best
└── training_summary.json                # metadata: paths, timestamps, elapsed
```

**The "best" model** is saved whenever the mean eval reward exceeds the previous best. Evaluation runs every ~50K / n_envs steps, but only after 200K steps (to avoid saving an untrained deterministic policy).

**Periodic snapshots** (`cheetah_ppo_{N}_steps.zip`) are saved every `max(100_000 // n_envs, 1)` rollouts. With 24 envs at 4096 steps/rollout they appear roughly every 4 rollouts.

### 5.2 Standalone `train.py`

```
checkpoints/
├── cheetah_ppo_{N}_steps.zip    # every ~100K steps
├── best/
│   └── best_model.zip
├── cheetah_final.zip            # after learn() completes
├── vec_normalize.pkl
└── training_config.json

logs/training/
├── monitor.csv
├── reward_components.csv
├── PPO_N/events.out.tfevents.*
└── eval/evaluations.npz
```

### 5.3 Hierarchical `train_hierarchical.py`

```
checkpoints/hierarchical/
├── hierarchical_{N}_steps.zip
├── best/best_model.zip
└── hierarchical_final.zip
```

### 5.4 Terrain Training

```
checkpoints/terrain/
├── terrain_transformer_{N}_steps.zip
├── best/best_model.zip
├── terrain_transformer_final.zip
└── terrain_config.json
```

---

## 6. Pausing and Resuming

### 6.1 Stopping Safely

You can press **Ctrl+C** at any time. SB3 will finish the current rollout, save state only if a `CheckpointCallback` or `EvalCallback` fires, then exit.

**To maximise progress saved before stopping:** wait for a "Saving model checkpoint" log line to appear (every ~100K steps), then Ctrl+C. The last periodic checkpoint will be intact.

To force an immediate safe save, you can also set very frequent checkpoints:
```python
# In your own wrapper: just re-run with small --mlp-steps, pointing at the zip
python3 src/training/train.py --resume checkpoints/cheetah_ppo_500000_steps.zip \
    --total-steps 501000  # run just enough to checkpoint, ctrl+C, then resume again
```

---

### 6.2 Resuming `train.py`

Use `--resume <checkpoint.zip>`. The trainer loads the model weights, attaches the live env, and continues training from the step count encoded in the zip.

```bash
# Resume from the latest periodic checkpoint
python3 src/training/train.py \
    --resume checkpoints/cheetah_ppo_500000_steps.zip \
    --total-steps 100000000 \
    --n-envs 24

# Resume from the best model
python3 src/training/train.py \
    --resume checkpoints/best/best_model.zip \
    --total-steps 100000000
```

> **VecNormalize note:** When resuming, `vec_normalize.pkl` is loaded automatically if it exists in the same directory as the `.zip`. If the paths differ, the normalizer starts from scratch — observation statistics will drift for a few million steps before stabilising. This is acceptable but suboptimal.

---

### 6.3 Resuming from Pipeline Mid-Run

The pipeline itself has **no built-in `--resume` flag**. Depending on where it stopped:

**If Stage 1 (MLP) finished but Stage 2 (Hierarchical) did not:**

```bash
python3 scripts/pipeline.py \
    --skip-mlp \
    --expert runs/20260405_143000/mlp_best.zip \
    --vec-normalize runs/20260405_143000/mlp_vec_normalize.pkl \
    --run-id 20260405_143000_resumed     # keep logs together
```

**If Stage 1 partially completed (need to continue it first):**

```bash
# Step 1: Find the latest checkpoint in the run's mlp_training dir
ls -lt runs/20260405_143000/mlp_training/*.zip

# Step 2: Resume Stage 1 with train.py directly
python3 src/training/train.py \
    --resume runs/20260405_143000/mlp_training/cheetah_ppo_4800000_steps.zip \
    --total-steps 100000000 \
    --ckpt-dir runs/20260405_143000/mlp_training \
    --log-dir runs/20260405_143000/logs_mlp

# Step 3: Once Stage 1 finishes, run Stage 2 via pipeline with --skip-mlp
python3 scripts/pipeline.py \
    --skip-mlp \
    --expert runs/20260405_143000/mlp_training/cheetah_final.zip \
    --vec-normalize runs/20260405_143000/mlp_training/vec_normalize.pkl \
    --run-id 20260405_143000_hier
```

**If Stage 2 partially completed:**

```bash
# Find latest hierarchical checkpoint
ls -lt runs/20260405_143000/hierarchical_training/*.zip

# Resume Stage 2 directly
python3 src/training/train_hierarchical.py \
    --expert runs/20260405_143000/mlp_best.zip \
    --vec-normalize runs/20260405_143000/mlp_vec_normalize.pkl \
    --total-steps 50000000 \
    --ckpt-dir runs/20260405_143000/hierarchical_training \
    --log-dir runs/20260405_143000/logs_hierarchical
```

> **Note:** Hierarchical stage 2 does not have a `--resume` flag like train.py — it re-runs BC warm-start and then continues PPO. To truly resume from a hierarchical checkpoint, load it directly as `--expert` to bypass BC and jump straight to PPO. This means BC initialisation is skipped but the loaded weights already encode the BC knowledge.

---

### 6.4 Resuming Stage 2 Only (`--skip-mlp`)

Use this when you already have a trained MLP model and want to run (or re-run) Stage 2:

```bash
python3 scripts/pipeline.py \
    --skip-mlp \
    --expert checkpoints/best/best_model.zip \
    --vec-normalize checkpoints/vec_normalize.pkl \
    --hier-steps 50000000 \
    --n-envs 24
```

Or directly:
```bash
python3 src/training/train_hierarchical.py \
    --expert checkpoints/best/best_model.zip \
    --vec-normalize checkpoints/vec_normalize.pkl \
    --total-steps 50000000 \
    --n-envs 24
```

---

### 6.5 Resuming Terrain Training

Terrain training (`train_terrain.py`) has no `--resume` flag. Start it again pointing to the last checkpoint via the `--expert` flag (it will re-do the BC phase using the checkpoint as expert):

```bash
python3 scripts/train_terrain.py \
    --total-steps 5000000 \
    --n-envs 8
```

Or re-launch with a custom ckpt dir that already has snapshots — the `CheckpointCallback` will start writing new ones but won't overwrite old ones (different step numbers in filenames).

---

## 7. Evaluation

Run a saved policy deterministically across multiple episodes and get a JSON report:

```bash
# Evaluate best model (default)
python3 scripts/evaluate.py

# Custom checkpoint, more episodes, with viewer
python3 scripts/evaluate.py \
    --checkpoint runs/20260405_143000/mlp_best.zip \
    --episodes 50 \
    --render

# Via run.py
python3 run.py eval --checkpoint checkpoints/best/best_model.zip --episodes 20
```

**Output:** `logs/eval_{unix_timestamp}.json`

```json
{
  "mean_reward": 4231.5,
  "std_reward": 812.3,
  "max_reward": 5809.0,
  "min_reward": 2115.0,
  "mean_length": 987.4,
  "survival_rate": 0.72
}
```

`survival_rate` = fraction of episodes reaching ≥ 999 steps (out of 2000 max).

---

## 8. Interactive Control (Keyboard Demo)

Opens the MuJoCo viewer and maps keyboard input to locomotion commands in real time. **Type commands in the terminal, not the viewer window.**

```bash
# Using best model (auto-detected)
python3 scripts/interactive_control.py

# Specific checkpoint
python3 scripts/interactive_control.py --checkpoint runs/20260405_143000/mlp_best.zip

# PD standing only (no policy) — good for debugging env/physics
python3 scripts/interactive_control.py --no-policy

# Via justfile
just control
```

### Keyboard Controls

| Key | Action | Note |
|-----|--------|------|
| `W` / `↑` | Move forward | Speed adjustable with 1/2/3 |
| `S` / `↓` | Move backward | |
| `A` / `←` | Strafe left | |
| `D` / `→` | Strafe right | |
| `Q` | Turn left | |
| `E` | Turn right | |
| `1` | Walk speed (0.5 m/s) | |
| `2` | Trot speed (1.2 m/s) | |
| `3` | Run speed (2.5 m/s) | |
| `J` | Jump | Triggers jump FSM sequence |
| `C` | Toggle crouch | Hold: crouched; release: stand |
| `Space` | Stop (zero velocity) | Mode stays the same |
| `Esc` | Quit | |

**Status is printed every 50 steps** (~1 second at 50 Hz) showing current height, velocity, reward, and skill mode.

**Auto-reset:** the episode resets automatically when terminated (fallen) or truncated (2000 steps). The robot re-spawns at standing height without restarting the script.

---

## 9. Recording Video

Records an MP4 rollout using `rgb_array` render mode (headless, no viewer window):

```bash
python3 scripts/record_video.py \
    --checkpoint checkpoints/best/best_model.zip \
    --output logs/my_rollout.mp4 \
    --episodes 1

# Default output: logs/rollout.mp4
python3 scripts/record_video.py
```

---

## 10. TensorBoard

```bash
# Default training run
tensorboard --logdir logs/training

# Specific pipeline run
tensorboard --logdir runs/20260405_143000/logs_mlp

# All runs overlaid
tensorboard --logdir runs/

# Hierarchical logs
tensorboard --logdir logs/training_hierarchical
```

**Key metrics to watch:**

| Metric | Good sign |
|--------|-----------|
| `rollout/ep_len_mean` | Increasing (robot surviving longer) |
| `rollout/ep_rew_mean` | Increasing |
| `train/clip_fraction` | 0.05–0.15 (above 0.3 = LR too high) |
| `train/entropy_loss` | Not collapsing to 0 (exploration alive) |
| `train/approx_kl` | < 0.02 (stable updates) |
| `train/explained_variance` | Approaching 1.0 (value function calibrated) |

Training PPO_1, PPO_2, ..., PPO_N folders correspond to successive `model.learn()` calls (each new training session increments the SB3 internal counter).

---

## 11. Reward Diagnostics

Inspect individual reward components for a specific skill mode without running full training:

```bash
python3 scripts/diagnose_reward.py --mode stand
python3 scripts/diagnose_reward.py --mode crouch
python3 scripts/diagnose_reward.py --mode walk
python3 scripts/diagnose_reward.py --mode run
```

This runs a few hundred steps in the given mode and prints the mean, min, max of each reward term, helping identify scale issues or broken terms.

For CSVs from an existing training run:

```python
import pandas as pd
df = pd.read_csv('logs/training/reward_components.csv')
print(df.describe())
print(df.tail(1000).mean())   # late-training averages
```

---

## 12. Training Configuration Reference

The canonical hyperparameters live in [training_config.json](../training_config.json).

### Environment

| Parameter | Value |
|-----------|-------|
| Robot | Unitree Go1 (mujoco_menagerie) |
| Sim frequency | 500 Hz (0.002 s physics_dt) |
| Control frequency | 50 Hz (0.02 s dt, 10 substeps) |
| Episode length | 2000 steps (40 s) |
| Observation dim | 54 (12 jpos + 12 jvel + 3 linvel + 3 angvel + 3 gravity + 12 prev_action + 4 command + 5 skill one-hot) |
| Action dim | 12 (delta joint position targets, ±0.5 rad) |
| PD gains | kp=100, kd=0 (passive damping in XML) |
| Max torque | 33.5 Nm |
| Standing height | 0.27 m |

### PPO (Stage 1 — MLP)

| Param | Value |
|-------|-------|
| Algorithm | PPO (SB3) |
| Policy | MlpPolicy `[2048, 1024, 512]`, ELU |
| Learning rate | 3e-4 → 1e-5 (linear decay) |
| n_envs | 24 |
| n_steps | 4096 |
| batch_size | 4096 |
| n_epochs | 10 |
| clip_range | 0.2 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| ent_coef | 0.01 |
| max_grad_norm | 1.0 |
| log_std_init | -1.0 (std ≈ 0.37) |
| Total steps | 100M |

### Hierarchical Stage (Stage 2)

| Param | Value |
|-------|-------|
| Policy | TransformerActorCriticPolicy + MoE |
| d_model | 256 |
| n_layers | 3 |
| n_experts | 4 |
| history_len | 16 |
| n_steps | 2048 |
| batch_size | 256 |
| n_epochs | 10 |
| max_grad_norm | 0.5 |
| BC epochs | 100 |
| BC lr | 5e-4 (cosine annealing) |
| Total steps | 50M |

### Reward Scales (v7)

| Term | Scale | Mode multipliers |
|------|-------|-----------------|
| r_alive | +0.5 | — (constant per step) |
| r_linvel | +1.5 | stand=0, crouch=0, run=2.5 |
| r_yaw | +0.5 | stand=0, crouch=0 |
| r_gait | +1.0 | — |
| r_posture | +1.0 | — |
| r_body_height | +1.0 | stand=1.5, crouch=2.0 |
| r_stillness | +1.5 | active only in stand/crouch |
| r_jump_phase | +2.0 | active only in jump |
| r_orientation | -2.0 | stand=2x, crouch=2x |
| r_ang_vel_xy | -0.05 | stand=2x |
| r_torque | -1e-5 | run=0.5x |
| r_smooth | -0.01 | run=0.5x |
| r_joint_limit | -10.0 | — |
| r_lin_vel_z | -2.0 | — |
| r_dof_vel | -5e-5 | — |

---

## 13. Checkpoint Reference

### Auto-saved models (what to use for what)

| File | When created | Use for |
|------|-------------|---------|
| `checkpoints/best/best_model.zip` | Whenever eval reward improves | Evaluation, demos, Stage 2 expert |
| `checkpoints/cheetah_final.zip` | End of `train.py` | Most up-to-date weights |
| `checkpoints/cheetah_ppo_{N}_steps.zip` | Every ~100K steps | Rollback to any point |
| `checkpoints/vec_normalize.pkl` | End of training | **Required** alongside any checkpoint for correct obs normalization |
| `runs/{id}/hierarchical_best.zip` | Stage 2 best eval | Best hierarchical demo/eval |
| `runs/{id}/training_summary.json` | End of pipeline | Audit log of the full run |

### Loading a model manually (Python)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from src.env.cheetah_env import MiniCheetahEnv

# Create env
env = DummyVecEnv([lambda: MiniCheetahEnv(render_mode="human")])

# Load VecNormalize stats (obs statistics — must match training)
env = VecNormalize.load("checkpoints/vec_normalize.pkl", env)
env.training = False    # don't update stats during inference
env.norm_reward = False

# Load policy
model = PPO.load("checkpoints/best/best_model.zip", env=env)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

---

## 14. Common Workflows

### "I want to train from scratch"

```bash
python3 scripts/pipeline.py --run-id full_run_v1 --n-envs 24
```

### "I want to quickly test everything works"

```bash
just train-quick   # 200K + 500K steps, takes ~10 min
just eval
just control
```

### "I want to continue a training run that was interrupted mid-Stage 1"

```bash
# Find where it stopped
ls -lt checkpoints/*.zip

# Resume from latest snapshot
python3 src/training/train.py \
    --resume checkpoints/cheetah_ppo_8200000_steps.zip \
    --total-steps 100000000 \
    --n-envs 24
```

### "I have a Stage 1 model, I want to train Stage 2"

```bash
python3 scripts/pipeline.py \
    --skip-mlp \
    --expert checkpoints/best/best_model.zip \
    --vec-normalize checkpoints/vec_normalize.pkl \
    --hier-steps 50000000
```

### "I want to evaluate a specific checkpoint"

```bash
python3 scripts/evaluate.py \
    --checkpoint checkpoints/best/best_model.zip \
    --episodes 20

# Or view it live:
python3 scripts/evaluate.py \
    --checkpoint checkpoints/best/best_model.zip \
    --render --episodes 5
```

### "I want to see the robot walk interactively"

```bash
python3 scripts/interactive_control.py \
    --checkpoint checkpoints/best/best_model.zip
```

### "I want to record a video of the best policy"

```bash
python3 scripts/record_video.py \
    --checkpoint checkpoints/best/best_model.zip \
    --output logs/demo.mp4
```

### "I want to monitor training in real time"

```bash
# In a second terminal:
tensorboard --logdir logs/training
# Open: http://localhost:6006
```

### "Something looks wrong with the reward — how do I debug it?"

```bash
# Check each reward term for a specific mode
python3 scripts/diagnose_reward.py --mode stand

# Inspect training CSV for patterns
python3 -c "
import pandas as pd
df = pd.read_csv('logs/training/reward_components.csv')
print(df[['r_linvel','r_body_height','r_stillness','r_torque','r_total']].tail(100).mean())
"
```

### "I want to clean up old logs but keep models"

```bash
just clean-logs     # removes logs/training/ and logs/training_hierarchical/
                    # keeps checkpoints/ and runs/ intact
```

> **Warning:** Do NOT run `make clean-all` (Makefile) — it deletes all of `checkpoints/`. Use `just clean-all` instead, which only removes `runs/` and `logs/` but keeps `checkpoints/best/`.
