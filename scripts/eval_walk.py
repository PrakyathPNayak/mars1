#!/usr/bin/env python3
"""Quick evaluation of the best model for walking behavior."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.env.cheetah_env import MiniCheetahEnv
from src.training.sb3_integration import ActionSmoothingWrapper
from stable_baselines3 import PPO

env_raw = MiniCheetahEnv(render_mode=None, terrain_type='flat', forced_mode='walk')
env = ActionSmoothingWrapper(env_raw, alpha=0.8)
model = PPO.load('checkpoints/advanced/best/best_model')

print("=== DETERMINISTIC ===")
for ep in range(6):
    obs, _ = env.reset()
    r = 0; vxs = []; s = 0
    while s < 400:
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, t, tr, _ = env.step(a)
        r += rew; vxs.append(float(env_raw.data.qvel[0])); s += 1
        if t or tr: break
    print(f"D{ep} s={s} r={r:.0f} vx={np.mean(vxs):.3f} ema={env_raw._vx_ema:.3f} cmd={env_raw.command[0]:.2f} fell={t}")

print("\n=== STOCHASTIC ===")
for ep in range(6):
    obs, _ = env.reset()
    r = 0; vxs = []; s = 0
    while s < 400:
        a, _ = model.predict(obs, deterministic=False)
        obs, rew, t, tr, _ = env.step(a)
        r += rew; vxs.append(float(env_raw.data.qvel[0])); s += 1
        if t or tr: break
    print(f"S{ep} s={s} r={r:.0f} vx={np.mean(vxs):.3f} ema={env_raw._vx_ema:.3f} cmd={env_raw.command[0]:.2f} fell={t}")

env.close()
