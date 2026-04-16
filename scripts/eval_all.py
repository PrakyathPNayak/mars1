"""Evaluate all saved models across all modes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from src.env.cheetah_env import MiniCheetahEnv

def eval_model(model_path, mode, n_eps=10, deterministic=True):
    """Evaluate a model on a specific mode."""
    model = PPO.load(model_path)
    env = MiniCheetahEnv(render_mode="none", forced_mode=mode, episode_length=2000)
    
    rewards, ep_lens, max_vx_list = [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=100+ep)
        total_r = 0.0
        max_vx = 0.0
        for step in range(2000):
            action, _ = model.predict(obs.astype(np.float32), deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            max_vx = max(max_vx, abs(float(env.data.qvel[0])))
            if term or trunc: break
        rewards.append(total_r)
        ep_lens.append(step+1)
        max_vx_list.append(max_vx)
    
    env.close()
    return {
        'reward': f"{np.mean(rewards):.0f}±{np.std(rewards):.0f}",
        'steps': f"{np.mean(ep_lens):.0f}±{np.std(ep_lens):.0f}",
        'r_per_step': f"{np.mean(rewards)/max(np.mean(ep_lens),1):.2f}",
        'max_vx': f"{np.mean(max_vx_list):.2f}",
    }

def eval_zero_action(mode, n_eps=10):
    """Zero-action baseline."""
    env = MiniCheetahEnv(render_mode="none", forced_mode=mode, episode_length=2000)
    rewards, ep_lens = [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=200+ep)
        total_r = 0.0
        for step in range(2000):
            obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
            total_r += r
            if term or trunc: break
        rewards.append(total_r)
        ep_lens.append(step+1)
    env.close()
    return {
        'reward': f"{np.mean(rewards):.0f}±{np.std(rewards):.0f}",
        'steps': f"{np.mean(ep_lens):.0f}±{np.std(ep_lens):.0f}",
        'r_per_step': f"{np.mean(rewards)/max(np.mean(ep_lens),1):.2f}",
        'max_vx': 'N/A',
    }

# Discover models
models = {}
for name in ['walk_v24d_best/best_model', 'walk_v24f_best/best_model',
             'walk_v24h_best/best_model', 'jump_v24_best/best_model',
             'run_v25_best/best_model', 'stand_v24_best/best_model']:
    path = f"checkpoints/{name}"
    if os.path.exists(path + ".zip"):
        models[name.split('/')[0]] = path

print(f"Found models: {list(models.keys())}")
print()

# Evaluate each model on its intended mode
for mode in ["stand", "walk", "run", "jump"]:
    print(f"=== {mode.upper()} MODE ===")
    
    # Zero-action baseline
    za = eval_zero_action(mode, n_eps=10)
    print(f"  Zero-action: reward={za['reward']}, steps={za['steps']}, r/step={za['r_per_step']}")
    
    # Check for matching model
    matching = [k for k in models if mode in k]
    for m in matching:
        for det in [True, False]:
            label = "deterministic" if det else "stochastic"
            result = eval_model(models[m], mode, n_eps=10, deterministic=det)
            print(f"  {m} ({label}): reward={result['reward']}, steps={result['steps']}, "
                  f"r/step={result['r_per_step']}, max_vx={result['max_vx']}")
    
    if not matching:
        print(f"  No trained model found for {mode}")
    print()
