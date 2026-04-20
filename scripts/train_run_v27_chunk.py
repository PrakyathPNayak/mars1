"""Run training v27 in 500K chunks with resume support."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from src.env.cheetah_env import MiniCheetahEnv
import numpy as np

def make_env(rank, mode):
    def _init():
        env = MiniCheetahEnv(render_mode="none", forced_mode=mode, episode_length=2000)
        env.reset(seed=rank)
        return env
    return _init

N_ENVS = 8
MODE = "run"
CHUNK = 500_000
CKPT_DIR = "checkpoints/run_v27_chunks"
os.makedirs(CKPT_DIR, exist_ok=True)

# Find latest chunk
existing = sorted(glob.glob(f"{CKPT_DIR}/chunk_*.zip"))
if existing:
    latest = existing[-1]
    chunk_num = int(latest.split("chunk_")[1].split(".")[0]) + 1
    total_so_far = chunk_num * CHUNK
    print(f"Resuming from {latest}, chunk {chunk_num}, {total_so_far:,} steps so far", flush=True)
else:
    latest = None
    chunk_num = 0
    total_so_far = 0
    print(f"Starting fresh, chunk 0", flush=True)

base_env = DummyVecEnv([make_env(i, MODE) for i in range(N_ENVS)])
env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True, clip_obs=10.0)

vecnorm_path = f"{CKPT_DIR}/vecnorm.pkl"
if latest:
    model = PPO.load(latest, env=env, device="cpu")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, DummyVecEnv([make_env(i, MODE) for i in range(N_ENVS)]))
        model.set_env(env)
    print(f"Loaded model, training chunk {chunk_num} ({CHUNK:,} steps)...", flush=True)
else:
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=4096, batch_size=4096, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0001,
        verbose=1, device="cpu",
        policy_kwargs=dict(log_std_init=-2.0, net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    print(f"Fresh model, training chunk 0 ({CHUNK:,} steps)...", flush=True)

model.learn(total_timesteps=CHUNK)

save_path = f"{CKPT_DIR}/chunk_{chunk_num:03d}"
model.save(save_path)
env.save(vecnorm_path)
total_so_far += CHUNK
print(f"Chunk {chunk_num} done. Saved to {save_path}. Total: {total_so_far:,} steps.", flush=True)
