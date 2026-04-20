from stable_baselines3 import PPO
#from src.env.terrain_env import AdvancedTerrainEnv
import time

# ---------- SETTINGS ----------
MODEL_PATH = "runs/best_model.zip"
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

# 🔥 IMPORTANT: slice to match model input (45)
obs = obs[:45]

# ---------- RUN ----------
for _ in range(STEPS):

    # ✅ ALWAYS pass correct-sized observation
    action, _ = model.predict(obs, deterministic=True)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    # 🔥 IMPORTANT: slice again after step
    obs = obs[:45]

    # slow down for visualization
    time.sleep(0.02)

    # reset if episode ends
    if terminated or truncated:
        obs, _ = env.reset()
        obs = obs[:45]

time.sleep(0.5)
env.close()
time.sleep(0.5)
