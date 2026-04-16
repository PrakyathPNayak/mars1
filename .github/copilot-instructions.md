# Copilot Instructions — Unitree Go1 RL Locomotion

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt
# Or with just (task runner): just install

# Run core tests (~8s, 10 tests)
python3 -m pytest tests/test_env.py -v

# Run a single test
python3 -m pytest tests/test_env.py -v -k test_step

# Full test suite with HTML report
python3 tests/test_suite.py --output reports/test_report.html

# Smoke-test the full training pipeline (~5 min CPU)
just train-quick
```

There is no linter or type checker configured.

## Architecture

This is a deep RL project that trains locomotion policies for the **Unitree Go1** quadruped robot in MuJoCo. Training uses a **two-stage hierarchical pipeline** orchestrated by `scripts/pipeline.py`:

**Stage 1 — MLP PPO expert** (`src/training/train.py`):
Standard PPO with a large MLP `[2048, 1024, 512]` + ELU. Trains from scratch using `stable-baselines3` with `SubprocVecEnv` + `VecNormalize`. Outputs `mlp_best.zip`.

**Stage 2 — Hierarchical BC → Transformer+MoE** (`src/training/train_hierarchical.py`):
Three phases: (1) collect expert rollouts from Stage 1, (2) behavioral cloning to warm-start a Transformer encoder, (3) PPO fine-tuning with the full hierarchical policy. The advanced policy (`src/training/advanced_policy.py`) includes sensory-group attention, mixture-of-experts, bilateral symmetry encoding, CPG oscillator, and an auxiliary world-model head.

**Environment** (`src/env/cheetah_env.py`):
Custom Gymnasium environment (v23). 196-dim observation, 12-dim action (delta joint targets ±0.2 rad). Multi-mode command system (stand/walk/run/jump) with terrain-aware heightmap observations. The reward function has 20+ terms versioned as `REWARD_SCALES` in the env file — this is the authoritative source for reward weights.

**Key data flow:**
- `run.py` — CLI entrypoint dispatching to train/demo/eval/test
- `scripts/pipeline.py` — orchestrates Stage 1 → Stage 2, writes all artifacts to `runs/{run_id}/`
- `src/training/sb3_integration.py` — wraps the Transformer policy as an SB3 `ActorCriticPolicy`
- `training_config.json` — reference hyperparameters (not loaded at runtime; the code uses argparse defaults)

## Key Conventions

- **Task runner:** `just` (justfile) is the primary task runner. The `Makefile` exists but is older and has diverged — prefer `just` recipes.
- **Python path:** Scripts use `sys.path.insert(0, project_root)` for imports. The Dockerfile sets `PYTHONPATH=/workspace`. Imports use `from src.env.cheetah_env import MiniCheetahEnv` style (not relative).
- **Environment naming:** The env class is `MiniCheetahEnv` (historical name from Mini Cheetah; the robot is now Unitree Go1). Don't rename it — it's referenced everywhere.
- **Reward versioning:** Reward weights are tracked by version (currently v7 in `training_config.json`, v23 in the env). Changes to rewards should be documented in the `REWARD_SCALES` dict and `training_config.json`.
- **Observation layout:** The 196-dim observation vector has a strict layout documented at the top of `cheetah_env.py` and mirrored in `SENSORY_GROUPS` in `advanced_policy.py`. These must stay in sync.
- **Checkpoint format:** Models are saved as SB3 `.zip` files. `VecNormalize` stats are saved separately as `.pkl`. Both are needed for inference.
- **Device convention:** MLP PPO runs on CPU (faster for small networks with SB3). GPU (`--device cuda`) is used for the Transformer+MoE stage.
- **Rendering:** MuJoCo uses EGL for headless rendering (`MUJOCO_GL=egl`). Pass `render_mode="none"` in tests.
- **Test pattern:** Tests in `test_env.py` follow a function-per-test pattern with a `run_all_tests()` collector. They also work with pytest (`-m pytest tests/test_env.py`).
