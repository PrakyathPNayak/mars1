"""Run-only training v27 — robust version with signal handling, no eval callback."""
import sys, os, signal, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

def signal_handler(signum, frame):
    signame = signal.Signals(signum).name
    with open("logs/run_v27_signal.log", "a") as f:
        f.write(f"Received signal {signame} ({signum})\n")
        traceback.print_stack(frame, file=f)
    print(f"\n!!! SIGNAL {signame} received !!!", flush=True)
    if signum in (signal.SIGTERM, signal.SIGINT):
        # Save what we have
        try:
            model.save("checkpoints/run_v27_interrupted")
            env.save("checkpoints/run_v27_interrupted_vecnorm.pkl")
            print("Saved interrupted model", flush=True)
        except: pass
    sys.exit(128 + signum)

for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(sig, signal_handler)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
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
TOTAL_STEPS = 3_000_000

print(f"Training {MODE}-only v27-robust, {N_ENVS} envs, {TOTAL_STEPS:,} steps...", flush=True)
base_env = DummyVecEnv([make_env(i, MODE) for i in range(N_ENVS)])
env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4, n_steps=4096, batch_size=4096, n_epochs=5,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0001,
    verbose=1, device="cpu",
    policy_kwargs=dict(log_std_init=-2.0, net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

# Simple checkpoint every 200K steps, no eval callback
ckpt_cb = CheckpointCallback(save_freq=25_000, save_path="checkpoints/run_v27_ckpts/",
                              name_prefix="run_v27", verbose=0)

try:
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt_cb)
    print("Training completed normally!", flush=True)
except Exception as e:
    print(f"Training exception: {e}", flush=True)
    traceback.print_exc()

model.save("checkpoints/run_v27_final")
env.save("checkpoints/run_v27_final_vecnorm.pkl")
print("Saved final model", flush=True)
