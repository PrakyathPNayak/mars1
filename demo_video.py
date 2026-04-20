import imageio
import numpy as np
import sys
from stable_baselines3 import PPO

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'terrain_testing'))

from terrain_testing.envs.base_terrain_wrapper import BaseTerrainWrapper

# ---------- SETTINGS ----------
MODEL_PATH = "runs/best_model.zip"
TERRAIN = "trench_crossing"   # change to flat / stairs / frozen
STEPS = 300
VIDEO_PATH = f"demo_{TERRAIN}.mp4"

# ---------- LOAD MODEL ----------
model = PPO.load(MODEL_PATH)

# ---------- CREATE ENV ----------
env = BaseTerrainWrapper(
    terrain_name=TERRAIN,
    render_mode="rgb_array"   # IMPORTANT
)

obs, _ = env.reset()

frames = []

# ---------- RUN ----------
for _ in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        break

env.close()

# ---------- SAVE VIDEO ----------
imageio.mimsave(VIDEO_PATH, frames, fps=30)

print(f"✅ Video saved as {VIDEO_PATH}")
