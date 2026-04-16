"""Walk-only training with SAC — testing if off-policy fixes deterministic gap.
PPO deterministic eval plateaus at ~716 (barely above zero-action 671).
SAC should produce better deterministic policy via explicit entropy-reward tradeoff.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import SAC
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

MODE = "walk"
TOTAL_STEPS = 2_000_000  # SAC is more sample-efficient

print(f"Training {MODE}-only SAC, 1 env, {TOTAL_STEPS:,} steps...")
print("Config: SAC auto-entropy, no VecNormalize, 256x256")

env = make_env(0, MODE)()
eval_env = VecMonitor(DummyVecEnv([make_env(100, MODE)]))

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=500_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    device="cpu",
    policy_kwargs=dict(
        net_arch=[256, 256],
    ),
)

eval_callback = EvalCallback(
    eval_env, best_model_save_path="checkpoints/walk_sac_best/",
    eval_freq=25_000, n_eval_episodes=10, deterministic=True, verbose=1,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)

model.save("checkpoints/walk_sac")
print("Saved to checkpoints/walk_sac")

# Evaluate
print("\n=== EVALUATION (10 episodes) ===")
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
      f"steps={np.mean(zs):.0f}±{np.std(zs):.0f}")

eval_raw.close()
env.close()
