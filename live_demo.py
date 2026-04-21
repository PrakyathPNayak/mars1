import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from stable_baselines3 import PPO
import time

# ---------- SETTINGS ----------
MODEL_PATH = "checkpoints/best/best_model.zip"
STEPS = 500

# ---------- LOAD MODEL ----------
model = PPO.load(
    MODEL_PATH,
    custom_objects={
        "learning_rate": 0.0003,
        "lr_schedule": lambda _: 0.0003,
        "clip_range": lambda _: 0.2,
    },
    device="cpu"
)

# ---------- CREATE ENV (LIVE WINDOW) ----------
from src.env.cheetah_env import MiniCheetahEnv
env = MiniCheetahEnv(render_mode="human")

# ---------- RESET ----------
obs, _ = env.reset()

# ---------- RUN ----------
for _ in range(STEPS):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    # slow down for visualization
    time.sleep(0.02)

    # reset if episode ends
    if terminated or truncated:
        obs, _ = env.reset()

time.sleep(0.5)
env.close()
time.sleep(0.5)
