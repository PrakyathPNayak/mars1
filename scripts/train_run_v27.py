"""Run-only training v27: speed-first reward, reduced penalties."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
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

print(f"Training {MODE}-only v27, {N_ENVS} envs, {TOTAL_STEPS:,} steps...")
print("Config: batch=4096, ent=0.0001, log_std=-2.0, action_scale=0.8, speed [0.5, 2.0]")
print("v27 reward: speed-first, 6x reduced ang_vel/lin_vel penalties")
import sys; sys.stdout.flush()

base_env = DummyVecEnv([make_env(i, MODE) for i in range(N_ENVS)])
env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True, clip_obs=10.0)

eval_base = DummyVecEnv([make_env(100, MODE)])
eval_env = VecNormalize(VecMonitor(eval_base), norm_obs=False, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=4096,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0001,
    verbose=1,
    device="cpu",
    policy_kwargs=dict(
        log_std_init=-2.0,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)

eval_callback = EvalCallback(
    eval_env, best_model_save_path="checkpoints/run_v27_best/",
    eval_freq=50_000, n_eval_episodes=5, deterministic=True, verbose=1,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)

model.save("checkpoints/run_v27")
env.save("checkpoints/run_v27_vecnorm.pkl")
print("Saved to checkpoints/run_v27*")
sys.stdout.flush()

# Quick eval
print("\n=== DETERMINISTIC EVAL (10 episodes) ===")
eval_raw = MiniCheetahEnv(render_mode="none", forced_mode=MODE, episode_length=2000)
rewards, ep_lens = [], []
for ep in range(10):
    obs, _ = eval_raw.reset(seed=100+ep)
    total_r = 0.0
    for step in range(2000):
        action, _ = model.predict(obs.astype(np.float32), deterministic=True)
        obs, r, term, trunc, info = eval_raw.step(action)
        total_r += r
        if term or trunc: break
    rewards.append(total_r)
    ep_lens.append(step+1)
    print(f"  ep{ep}: steps={step+1}, r={total_r:.1f}")
    sys.stdout.flush()

print(f"\nDeterministic: {np.mean(rewards):.1f}+/-{np.std(rewards):.1f}, {np.mean(ep_lens):.0f}+/-{np.std(ep_lens):.0f} steps")

# Zero-action baseline
zr, zs = [], []
for ep in range(5):
    obs, _ = eval_raw.reset(seed=200+ep)
    total_r = 0.0
    for step in range(2000):
        obs, r, term, trunc, info = eval_raw.step(np.zeros(12, dtype=np.float32))
        total_r += r
        if term or trunc: break
    zr.append(total_r)
    zs.append(step+1)
print(f"Zero-action: {np.mean(zr):.1f}+/-{np.std(zr):.1f}, {np.mean(zs):.0f}+/-{np.std(zs):.0f} steps")
eval_raw.close()
env.close()
