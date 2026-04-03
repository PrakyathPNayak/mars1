# justfile — Unitree Go1 RL Locomotion (reward v3)
# Install: cargo install just   |   https://just.systems
# Run:     just <recipe>        |   just --list  (show all recipes)

set shell := ["bash", "-c"]

PYTHON := "python3"
VENV_PYTHON := if path_exists("/home/prakyathpnayak/venv/bin/python3") == "true" { "/home/prakyathpnayak/venv/bin/python3" } else { "python3" }

# ── Default: show available recipes ────────────────────────────────────────────
default:
    @just --list

# ── Setup ──────────────────────────────────────────────────────────────────────

# Install all Python dependencies
install:
    pip install gymnasium stable-baselines3[extra] tensorboard mujoco pynput imageio scipy matplotlib torch

# Install dependencies for development (includes testing tools)
install-dev: install
    pip install pytest pytest-html

# ── Testing ────────────────────────────────────────────────────────────────────

# Run core environment tests (fast, ~8s)
test:
    {{PYTHON}} -m pytest tests/test_env.py -v

# Run full test suite with HTML report
test-full:
    {{PYTHON}} tests/test_suite.py --output reports/test_report.html

# Quick test suite (subset)
test-quick:
    {{PYTHON}} tests/test_suite.py --quick --output reports/test_report_quick.html

# ── Training ───────────────────────────────────────────────────────────────────

# Full pipeline: MLP expert (3M steps) → Hierarchical Transformer+MoE (10M steps)
train:
    {{PYTHON}} scripts/pipeline.py

# Full pipeline with GPU
train-gpu:
    {{PYTHON}} scripts/pipeline.py --device cuda

# Full pipeline with custom steps
train-steps mlp_steps="3000000" hier_steps="10000000":
    {{PYTHON}} scripts/pipeline.py --mlp-steps {{mlp_steps}} --hier-steps {{hier_steps}}

# Quick smoke-test run (~5 min CPU) to verify the pipeline works end-to-end
train-quick:
    {{PYTHON}} scripts/pipeline.py \
        --mlp-steps 200000 \
        --hier-steps 500000 \
        --bc-epochs 5 \
        --run-id smoke_test

# MLP PPO only (Stage 1)
train-mlp steps="3000000" envs="8":
    {{PYTHON}} src/training/train.py --total-steps {{steps}} --n-envs {{envs}}

train-mlp-gpu steps="3000000":
    {{PYTHON}} src/training/train.py --total-steps {{steps}} --device cuda

# Hierarchical BC → Transformer+MoE (Stage 2, needs an expert checkpoint)
train-hier expert="checkpoints/best/best_model.zip":
    {{PYTHON}} src/training/train_hierarchical.py --expert {{expert}}

train-hier-gpu expert="checkpoints/best/best_model.zip":
    {{PYTHON}} src/training/train_hierarchical.py --expert {{expert}} --device cuda

# Pipeline with a pre-trained expert (skip Stage 1)
train-hier-only expert="checkpoints/best/best_model.zip":
    {{PYTHON}} scripts/pipeline.py --skip-mlp --expert {{expert}}

# ── Evaluation ─────────────────────────────────────────────────────────────────

# Evaluate the best MLP checkpoint (20 episodes)
eval checkpoint="checkpoints/best/best_model.zip":
    {{PYTHON}} scripts/evaluate.py --checkpoint {{checkpoint}} --episodes 20

# Evaluate a specific run's hierarchical model
eval-hier run_id:
    {{PYTHON}} scripts/evaluate.py \
        --checkpoint runs/{{run_id}}/hierarchical_best.zip \
        --episodes 20

# ── Visualization ──────────────────────────────────────────────────────────────

# Interactive demo (MuJoCo viewer + keyboard control)
demo checkpoint="checkpoints/best/best_model.zip":
    {{PYTHON}} run.py demo --checkpoint {{checkpoint}}

# Demo without policy (pure keyboard control in physics sim)
demo-no-policy:
    {{PYTHON}} run.py demo --no-policy

# Record a video rollout
record checkpoint="checkpoints/best/best_model.zip" output="logs/rollout.mp4":
    {{PYTHON}} scripts/record_video.py --checkpoint {{checkpoint}} --output {{output}}

# Launch TensorBoard to inspect training logs
tensorboard logdir="logs/training":
    tensorboard --logdir {{logdir}}

# ── Utilities ──────────────────────────────────────────────────────────────────

# Run a quick reward diagnostic (200-step rollout, prints per-component reward breakdown)
# mode: stand, crouch, trot, run, walk
diagnose mode="stand":
    {{PYTHON}} scripts/diagnose_reward.py --mode {{mode}}

# Interactive on-robot command control (keyboard → velocity commands)
control checkpoint="checkpoints/best/best_model.zip":
    {{PYTHON}} scripts/interactive_control.py --checkpoint {{checkpoint}}

# ── Cleanup ────────────────────────────────────────────────────────────────────

# Remove Python cache files and pytest artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache/
    @echo "Cleaned cache files."

# Remove logs and checkpoints (keeps models in checkpoints/best/)
clean-logs:
    rm -rf logs/training/ logs/training_hierarchical/
    @echo "Cleaned training logs."

# Remove all generated outputs (runs/, logs/, checkpoints/ except best/)
clean-all: clean clean-logs
    rm -rf runs/
    @echo "Cleaned all outputs (runs/, logs/). checkpoints/best/ preserved."
