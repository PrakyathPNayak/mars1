#!/usr/bin/env python3
"""
Generate a self-contained Colab notebook for the Mini Cheetah RL project.

Embeds source files as base64+zlib compressed blobs so the notebook stays
compact and clean while being fully self-contained.

Usage: python3 create_colab_notebook.py
Output: mini_cheetah_colab.ipynb
"""
import json
import os
import base64
import zlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read_src(relpath):
    """Read a project file and return its content."""
    with open(os.path.join(PROJECT_ROOT, relpath), "r") as f:
        return f.read()


def compress_file(relpath):
    """Read a file and return base64+zlib compressed string."""
    raw = read_src(relpath).encode()
    return base64.b64encode(zlib.compress(raw, 9)).decode()


def make_cell(cell_type, source, metadata=None, collapsed=False):
    """Create an ipynb cell dict."""
    if isinstance(source, str):
        lines = []
        for i, line in enumerate(source.split("\n")):
            if i < len(source.split("\n")) - 1:
                lines.append(line + "\n")
            else:
                if line:
                    lines.append(line)
        source = lines

    meta = metadata or {}
    if collapsed:
        meta["jupyter"] = {"source_hidden": True}

    cell = {
        "cell_type": cell_type,
        "metadata": meta,
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    cells = []

    # ── Title ──
    cells.append(make_cell("markdown", """# MIT Mini Cheetah — RL Locomotion (Colab)

**Quadruped locomotion with PPO + MuJoCo**

This notebook is fully self-contained. Just run all cells top-to-bottom:
1. Install dependencies
2. Write source files to disk (auto-generated from project repo)
3. Train PPO policy (MLP [512, 256, 128])
4. Evaluate and visualize results

> Runtime → Change runtime type → **T4 GPU** recommended (training uses CPU for MLP, but GPU helps PyTorch ops)
"""))

    # ── Install deps ──
    cells.append(make_cell("markdown", "## 1. Install Dependencies"))

    cells.append(make_cell("code",
        "!pip install -q gymnasium stable-baselines3[extra] mujoco tensorboard matplotlib scipy"))

    # ── Project setup: single compact cell with base64-compressed sources ──
    cells.append(make_cell("markdown",
        "## 2. Project Setup\nCreate directory structure and write all source files (auto-generated)."))

    # Build the compressed source blobs
    source_files = {
        "assets/mini_cheetah.xml": compress_file("assets/mini_cheetah.xml"),
        "src/env/cheetah_env.py": compress_file("src/env/cheetah_env.py"),
        "src/training/train.py": compress_file("src/training/train.py"),
        "src/training/reward_logger.py": compress_file("src/training/reward_logger.py"),
        "scripts/evaluate.py": compress_file("scripts/evaluate.py"),
    }

    # Format the dict as a Python literal
    blob_lines = []
    for path, blob in source_files.items():
        blob_lines.append(f'    "{path}":\n        "{blob}",')
    blobs_str = "\n".join(blob_lines)

    cells.append(make_cell("code", f"""import os, sys, base64, zlib, pathlib

# Create directory structure
for d in ["assets", "src/env", "src/training", "src/control",
          "src/robot", "src/utils", "src/visualization",
          "scripts", "logs/training/eval", "checkpoints/best"]:
    os.makedirs(d, exist_ok=True)
for d in ["src", "src/env", "src/training", "src/control",
          "src/robot", "src/utils", "src/visualization"]:
    pathlib.Path(d, "__init__.py").touch()

# Decode and write compressed source files
_SOURCES = {{
{blobs_str}
}}

for path, blob in _SOURCES.items():
    pathlib.Path(path).write_bytes(zlib.decompress(base64.b64decode(blob)))
    print(f"  {{path}}")

sys.path.insert(0, os.getcwd())
print("\\nProject files ready.")"""))

    # ── Training section ──
    cells.append(make_cell("markdown", """## 3. Train PPO Policy

Configure training hyperparameters below. Default: **2M steps** with 8 parallel envs.

| Parameter | Value |
|-----------|-------|
| Architecture | MLP [512, 256, 128] (actor & critic) |
| Algorithm | PPO (clip=0.2, lr=3e-4, batch=256) |
| Reward | Velocity tracking + penalties (no survival bonus) |
| Domain rand. | Mass ±20%, friction ±50% |

Adjust `TOTAL_STEPS` and `N_ENVS` as needed.
"""))

    cells.append(make_cell("code", """import types
from src.training.train import train

TOTAL_STEPS = 2_000_000  # 2M ≈ 25 min on Colab
N_ENVS = 8

train(types.SimpleNamespace(total_steps=TOTAL_STEPS, n_envs=N_ENVS, resume=None))
print("\\nTraining complete!")"""))

    # ── TensorBoard ──
    cells.append(make_cell("markdown", "## 4. Monitor Training\nLaunch TensorBoard inline to see reward curves."))

    cells.append(make_cell("code", """%load_ext tensorboard
%tensorboard --logdir logs/training"""))

    # ── Evaluation section ──
    cells.append(make_cell("markdown", """## 5. Evaluate Trained Policy

Run the best checkpoint on 20 episodes and print statistics.
"""))

    cells.append(make_cell("code", """import os
import numpy as np
from stable_baselines3 import PPO
from src.env.cheetah_env import MiniCheetahEnv

# Load best checkpoint (fall back to final if best doesn't exist)
ckpt_path = "checkpoints/best/best_model.zip"
if not os.path.exists(ckpt_path):
    ckpt_path = "checkpoints/cheetah_final.zip"

policy = PPO.load(ckpt_path)
env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=1000)

rewards, lengths = [], []
for ep in range(20):
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_r += reward
        steps += 1
        if done or truncated:
            break
    rewards.append(total_r)
    lengths.append(steps)
    print(f"Episode {ep+1:3d}: reward={total_r:8.2f}, steps={steps}")

env.close()

print(f"\\n{'='*50}")
print(f"Mean reward:   {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
print(f"Mean length:   {np.mean(lengths):8.1f}")
print(f"Survival rate: {sum(1 for l in lengths if l >= 999)/len(lengths)*100:.0f}%")"""))

    # ── Visualization: render video ──
    cells.append(make_cell("markdown", """## 6. Visualize Policy (Video)

Render the trained policy and display as an inline video.
"""))

    cells.append(make_cell("code", """import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

policy = PPO.load(ckpt_path)
env = MiniCheetahEnv(render_mode="rgb_array", randomize_domain=False, episode_length=500)

frames = []
obs, _ = env.reset()
for _ in range(500):
    action, _ = policy.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    if done or truncated:
        break
env.close()

if frames:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])
    def update(i):
        im.set_data(frames[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
    plt.close(fig)
    display(HTML(ani.to_jshtml()))
else:
    print("No frames rendered — check MuJoCo rgb_array support")"""))

    # ── Reward component analysis ──
    cells.append(make_cell("markdown", """## 7. Reward Component Analysis

Analyze the reward breakdown across a rollout.
"""))

    cells.append(make_cell("code", """policy = PPO.load(ckpt_path)
env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=500)

component_history = {k: [] for k in [
    "r_linvel", "r_yaw", "r_orientation", "r_lin_vel_z",
    "r_ang_vel_xy", "r_height", "r_torque", "r_smooth", "r_total"
]}

obs, _ = env.reset()
for _ in range(500):
    action, _ = policy.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if "reward_components" in info:
        for k, v in info["reward_components"].items():
            if k in component_history:
                component_history[k].append(v)
    if done or truncated:
        break
env.close()

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Reward Components Over Episode", fontsize=14)
for ax, (name, values) in zip(axes.flat, component_history.items()):
    ax.plot(values, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Step")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("\\nAverage reward components:")
for name, values in component_history.items():
    if values:
        print(f"  {name:20s}: {np.mean(values):+.4f}")"""))

    # ── Download checkpoint ──
    cells.append(make_cell("markdown", "## 8. Download Trained Model\nDownload the best checkpoint to your local machine."))

    cells.append(make_cell("code", """from google.colab import files

if os.path.exists("checkpoints/best/best_model.zip"):
    files.download("checkpoints/best/best_model.zip")
    print("Downloaded best_model.zip")
elif os.path.exists("checkpoints/cheetah_final.zip"):
    files.download("checkpoints/cheetah_final.zip")
    print("Downloaded cheetah_final.zip")
else:
    print("No checkpoint found — train first!")"""))

    # ── Assemble notebook ──
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    out_path = os.path.join(PROJECT_ROOT, "mini_cheetah_colab.ipynb")
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook written: {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
