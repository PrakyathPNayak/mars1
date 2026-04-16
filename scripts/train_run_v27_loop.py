"""Run training v27 in 250K chunks with resume. Bulletproof saves via callback + signal handler."""
import sys, os, glob, signal, atexit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from src.env.cheetah_env import MiniCheetahEnv
import numpy as np

def make_env(rank, mode="run"):
    def _init():
        env = MiniCheetahEnv(render_mode="none", forced_mode=mode, episode_length=2000)
        env.reset(seed=rank)
        return env
    return _init

N_ENVS = 4
CHUNK = 250_000
TARGET = 3_000_000
CKPT_DIR = "checkpoints/run_v27_loop"
os.makedirs(CKPT_DIR, exist_ok=True)

# Find latest chunk
existing = sorted(glob.glob(f"{CKPT_DIR}/chunk_*.zip"))
if existing:
    latest = existing[-1]
    chunk_num = int(os.path.basename(latest).split("_")[1].split(".")[0]) + 1
    total_so_far = chunk_num * CHUNK
else:
    latest = None
    chunk_num = 0
    total_so_far = 0

if total_so_far >= TARGET:
    print(f"Already at {total_so_far:,} >= {TARGET:,}. Done!", flush=True)
    sys.exit(0)

print(f"Chunk {chunk_num}: {total_so_far:,}/{TARGET:,} steps. Training {CHUNK:,} more...", flush=True)

base_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
vecnorm_path = f"{CKPT_DIR}/vecnorm.pkl"

if latest and os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, VecMonitor(base_env))
    env.training = True
    model = PPO.load(latest, env=env, device="cpu")
    print(f"Resumed from {latest}", flush=True)
elif latest:
    env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True)
    model = PPO.load(latest, env=env, device="cpu")
    print(f"Resumed model (no vecnorm)", flush=True)
else:
    env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=2048, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0001,
        verbose=1, device="cpu",
        policy_kwargs=dict(log_std_init=-2.0, net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    print("Fresh model", flush=True)

# Emergency save on any signal
save_path = f"{CKPT_DIR}/chunk_{chunk_num:03d}"
_saved = [False]

def emergency_save(sig=None, frame=None):
    if _saved[0]:
        return
    _saved[0] = True
    signame = signal.Signals(sig).name if sig else "atexit"
    print(f"\n{signame}: emergency save to {save_path}...", flush=True)
    try:
        model.save(save_path)
        env.save(vecnorm_path)
        print(f"{signame}: saved OK", flush=True)
    except Exception as e:
        print(f"{signame}: save FAILED: {e}", flush=True)
    if sig:
        sys.exit(0)

signal.signal(signal.SIGTERM, emergency_save)
signal.signal(signal.SIGINT, emergency_save)
atexit.register(emergency_save)

# Periodic checkpoint callback (every 50K steps)
class PeriodicSave(BaseCallback):
    def __init__(self, save_freq=50_000):
        super().__init__()
        self.save_freq = save_freq
        self.last_save = 0
    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_freq:
            model.save(save_path)
            env.save(vecnorm_path)
            self.last_save = self.num_timesteps
            print(f"  [periodic save at {self.num_timesteps:,} steps]", flush=True)
        return True

try:
    model.learn(total_timesteps=CHUNK, reset_num_timesteps=True, callback=PeriodicSave(50_000))
    print("Training complete.", flush=True)
except Exception as e:
    print(f"Training error: {e}", flush=True)
    import traceback; traceback.print_exc()

# Final save
if not _saved[0]:
    _saved[0] = True
    print(f"Saving to {save_path}...", flush=True)
    model.save(save_path)
    env.save(vecnorm_path)
    total_so_far += CHUNK
    print(f"Chunk {chunk_num} done. Total: {total_so_far:,}/{TARGET:,}", flush=True)
