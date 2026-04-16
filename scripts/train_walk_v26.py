"""Walk training v27d: calibrated reference + Gaussian tracking.
Reference scales amplitude by vx_cmd (0.65*cmd+0.22, clamped [0.3,0.9]).
16 SubprocVecEnv on i9-14900K. norm_reward=True, norm_obs=False, batch=4096.
v27d: track(5.0,s=0.25) + vx_lin(2.0) + gait(0.5) - orient(0.1).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from src.env.cheetah_env import MiniCheetahEnv
import numpy as np

def make_env(rank, mode):
    def _init():
        env = MiniCheetahEnv(render_mode="none", forced_mode=mode, episode_length=2000)
        env.reset(seed=rank)
        return env
    return _init

N_ENVS = 16
MODE = "walk"
TOTAL_STEPS = 10_000_000

print(f"Training {MODE}-only v27d, {N_ENVS} SubprocVecEnv, {TOTAL_STEPS:,} steps...")
print("Calibrated reference (0.65*cmd+0.22), 16 parallel envs on i9-14900K")
print("Reward: track(5.0,s=0.25) + vx_lin(2.0) + gait(0.5) - orient(0.1)")

base_env = SubprocVecEnv([make_env(i, MODE) for i in range(N_ENVS)])
env = VecNormalize(VecMonitor(base_env), norm_obs=False, norm_reward=True, clip_obs=10.0)

eval_base = DummyVecEnv([make_env(100, MODE)])
eval_env = VecNormalize(VecMonitor(eval_base), norm_obs=False, norm_reward=False, clip_obs=10.0)

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
    eval_env, best_model_save_path="checkpoints/walk_v27d_best/",
    eval_freq=50_000, n_eval_episodes=10, deterministic=True, verbose=1,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback)

model.save("checkpoints/walk_v27d")
print("Saved to checkpoints/walk_v27d")

# Evaluate with distance measurement
print("\n=== EVALUATION (10 episodes) ===")
eval_raw = MiniCheetahEnv(render_mode="none", forced_mode=MODE, episode_length=2000)
std_val = np.exp(model.policy.log_std.detach().numpy()).mean()
print(f"Final std: {std_val:.4f}")

for label, deterministic in [("deterministic", True), ("stochastic", False)]:
    rewards, ep_lens, distances = [], [], []
    for ep in range(10):
        obs, _ = eval_raw.reset(seed=100+ep)
        start_x = float(eval_raw.data.qpos[0])
        total_r = 0.0
        for step in range(2000):
            action, _ = model.predict(obs.astype(np.float32), deterministic=deterministic)
            obs, r, term, trunc, info = eval_raw.step(action)
            total_r += r
            if term or trunc: break
        end_x = float(eval_raw.data.qpos[0])
        rewards.append(total_r)
        ep_lens.append(step+1)
        distances.append(end_x - start_x)
    print(f"{label}: r={np.mean(rewards):.1f}±{np.std(rewards):.1f}, "
          f"steps={np.mean(ep_lens):.0f}±{np.std(ep_lens):.0f}, "
          f"dist={np.mean(distances):.2f}±{np.std(distances):.2f}m, "
          f"speed={np.mean(distances)/np.mean(ep_lens)/0.02:.2f} m/s")

# Zero-action baseline (same seeds)
print("\n=== ZERO-ACTION BASELINE ===")
zr, zs, zd = [], [], []
for ep in range(10):
    obs, _ = eval_raw.reset(seed=100+ep)
    start_x = float(eval_raw.data.qpos[0])
    total_r = 0.0
    for step in range(2000):
        obs, r, term, trunc, info = eval_raw.step(np.zeros(12, dtype=np.float32))
        total_r += r
        if term or trunc: break
    end_x = float(eval_raw.data.qpos[0])
    zr.append(total_r)
    zs.append(step+1)
    zd.append(end_x - start_x)
print(f"zero-action: r={np.mean(zr):.1f}±{np.std(zr):.1f}, "
      f"steps={np.mean(zs):.0f}±{np.std(zs):.0f}, "
      f"dist={np.mean(zd):.2f}±{np.std(zd):.2f}m")

# Command-specific speed tracking test
print("\n=== COMMAND TRACKING TEST ===")
import math
for vx_cmd in [0.30, 0.50, 0.75, 1.00, 1.20]:
    speeds = []
    for ep in range(5):
        obs, _ = eval_raw.reset(seed=200+ep)
        x0 = float(eval_raw.data.qpos[0])
        for step in range(500):
            eval_raw.command[0] = vx_cmd
            action, _ = model.predict(obs.astype(np.float32), deterministic=True)
            obs, r, term, trunc, info = eval_raw.step(action)
            if term or trunc: break
        dx = float(eval_raw.data.qpos[0]) - x0
        speeds.append(dx / ((step+1)*0.02))
    spd = np.mean(speeds)
    err = abs(spd - vx_cmd)
    print(f"  cmd={vx_cmd:.2f}: speed={spd:.2f} err={err:.2f} track={math.exp(-err**2/0.25):.3f}")

eval_raw.close()
env.close()
