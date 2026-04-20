#!/usr/bin/env python3
"""Evaluate v31q checkpoint across all modes."""
import numpy as np, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.env.cheetah_env import MiniCheetahEnv
from scipy.spatial.transform import Rotation

def make_env():
    env = MiniCheetahEnv(render_mode="none")
    env.randomize_commands = False
    return env

def eval_mode(model, venv, mode, cmd, n_steps=500, warmup=50, target_height=None):
    """Evaluate a single mode. Returns dict of metrics."""
    obs = venv.reset()
    raw_envs = venv.get_attr('unwrapped')
    raw_env = raw_envs[0]
    
    raw_env.command = np.array(cmd, dtype=np.float32)
    raw_env.command_mode = mode
    if target_height is not None:
        raw_env._effective_target_height = target_height
        raw_env.target_height = target_height
    
    vx_list, vy_list, wz_list, heights, rewards = [], [], [], [], []
    
    for step in range(warmup + n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        
        if step < warmup:
            continue
        
        # Get body-frame velocity
        quat = raw_env.data.qpos[3:7]
        world_vel = raw_env.data.qvel[:3]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        body_vel = rot.inv().apply(world_vel)
        
        vx_list.append(body_vel[0])
        vy_list.append(body_vel[1])
        wz_list.append(raw_env.data.qvel[5])
        heights.append(raw_env.data.qpos[2])
        rewards.append(reward[0])
    
    return {
        'vx_mean': np.mean(vx_list),
        'vy_mean': np.mean(vy_list),
        'wz_mean': np.mean(wz_list),
        'height_mean': np.mean(heights),
        'reward_mean': np.mean(rewards),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to model .zip')
    parser.add_argument('--vecnorm', default=None, help='Path to vec_normalize.pkl')
    args = parser.parse_args()
    
    if args.vecnorm is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        args.vecnorm = os.path.join(ckpt_dir, 'vec_normalize.pkl')
    
    venv = DummyVecEnv([make_env])
    if os.path.exists(args.vecnorm):
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False
    
    model = PPO.load(args.ckpt, env=venv)
    
    tests = {
        'walk_fwd':  ('walk', [0.5, 0.0, 0.0], 'vx', 0.5, None),
        'walk_back': ('walk', [-0.3, 0.0, 0.0], 'vx', -0.3, None),
        'lat_R':     ('walk', [0.0, 0.3, 0.0], 'vy', 0.3, None),
        'lat_L':     ('walk', [0.0, -0.3, 0.0], 'vy', -0.3, None),
        'yaw_R':     ('walk', [0.0, 0.0, 0.8], 'wz', 0.8, None),
        'yaw_L':     ('walk', [0.0, 0.0, -0.8], 'wz', -0.8, None),
        'stand':     ('stand', [0.0, 0.0, 0.0], 'vx', 0.0, None),
        'crouch':    ('walk', [0.0, 0.0, 0.0], 'height', 0.08, 0.08),
        'run_fwd':   ('run', [2.0, 0.0, 0.0], 'vx', 2.0, None),
        'jump':      ('jump', [0.0, 0.0, 0.0], 'height', 0.45, None),
    }
    
    ckpt_name = os.path.basename(args.ckpt)
    print(f"\n{'='*60}")
    print(f"EVAL: {ckpt_name}")
    print(f"{'='*60}")
    
    for name, (mode, cmd, metric_key, target, th) in tests.items():
        result = eval_mode(model, venv, mode, cmd, target_height=th)
        
        if metric_key == 'height' and name == 'crouch':
            actual = result['height_mean']
            pct = max(0, 1.0 - abs(actual - target) / 0.20) * 100
        elif metric_key == 'height' and name == 'jump':
            actual = result['height_mean']
            pct = max(0, (actual - 0.28) / (0.45 - 0.28)) * 100
        elif target == 0.0:
            actual = result[f'{metric_key}_mean']
            pct = max(0, (1.0 - abs(actual) / 0.1)) * 100
        else:
            actual = result[f'{metric_key}_mean']
            pct = (actual / target) * 100 if target != 0 else 0
        
        print(f"  {name:12s}: actual={actual:+.3f}  target={target:+.3f}  pct={pct:.0f}%  r={result['reward_mean']:.2f}")
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
