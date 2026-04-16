"""Walk-only training v24g: NO VecNormalize — raw observations.
Key insight: VecNormalize causes train/eval mismatch. Raw obs are in [-3, 3] naturally.
Changes from v24f:
  - norm_obs=False (no observation normalization)
  - batch_size=4096 (40 updates/rollout, proven in v24f)
  - ent_coef=0.0001 (near-zero entropy)
  - log_std_init=-2.0 (std=0.135)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
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
MODE = "walk"
TOTAL_STEPS = 5_000_000

print(f"Training {MODE}-only v24g, {N_ENVS} envs, {TOTAL_STEPS:,} steps...")
print("Config: NO VecNormalize, batch=4096, ent_coef=0.0001, log_std=-2.0")

# NO VecNormalize — raw observations
env = VecMonitor(DummyVecEnv([make_env(i, MODE) for i in range(N_ENVS)]))
eval_env = VecMonitor(DummyVecEnv([make_env(100, MODE)]))

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
    eval_env, best_model_save_path="checkpoints/walk_v24g_best/",
    eval_freq=50_000, n_eval_episodes=10, deterministic=True, verbose=1,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)

model.save("checkpoints/walk_v24g")
print("Saved to checkpoints/walk_v24g")

# Evaluate — no normalization needed!
print("\n=== EVALUATION (10 episodes, deterministic) ===")
eval_raw = MiniCheetahEnv(render_mode="none", forced_mode=MODE, episode_length=2000)
for label, deterministic in [("deterministic", True), ("stochastic", False)]:
    rewards, ep_lens = [], []
    for ep in range(10):
        obs, _ = eval_raw.reset(seed=100+ep)
        total_r = 0.0
        for step in range(2000):
            action, _ = model.predict(obs.astype(np.float32), deterministic=deterministic)
            obs, r, term, trunc, info = eval_raw.step(action)
            total_r += r
            if term or trunc: break
        rewards.append(total_r)
        ep_lens.append(step+1)
    print(f"{label}: r={np.mean(rewards):.1f}±{np.std(rewards):.1f}, "
          f"steps={np.mean(ep_lens):.0f}±{np.std(ep_lens):.0f}, "
          f"r/step={np.mean(rewards)/np.mean(ep_lens):.3f}")

# Zero-action baseline
print("\n=== ZERO-ACTION BASELINE ===")
zr, zs = [], []
for ep in range(10):
    obs, _ = eval_raw.reset(seed=100+ep)
    total_r = 0.0
    for step in range(2000):
        obs, r, term, trunc, info = eval_raw.step(np.zeros(12, dtype=np.float32))
        total_r += r
        if term or trunc: break
    zr.append(total_r)
    zs.append(step+1)
print(f"zero-action: r={np.mean(zr):.1f}±{np.std(zr):.1f}, "
      f"steps={np.mean(zs):.0f}±{np.std(zs):.0f}, "
      f"r/step={np.mean(zr)/np.mean(zs):.3f}")

eval_raw.close()
env.close()
