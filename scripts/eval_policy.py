"""Evaluate a trained policy across standard scenarios.

Usage:
    python3 scripts/eval_policy.py checkpoints/v15b/cheetah_ppo_2000000_steps.zip
    python3 scripts/eval_policy.py checkpoints/v13/cheetah_final.zip --trials 20
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import re
import numpy as np
from src.env.cheetah_env import MiniCheetahEnv
from stable_baselines3 import PPO


SCENARIOS = [
    # (name,     mode_str, [vx, vy, wz])
    ("Stand",          "stand",  [0, 0, 0]),
    ("Walk fwd 0.5",   "walk",   [0.5, 0, 0]),
    ("Walk fwd 0.3",   "walk",   [0.3, 0, 0]),
    ("Walk lat +0.3",  "walk",   [0, 0.3, 0]),
    ("Walk lat -0.3",  "walk",   [0, -0.3, 0]),
    ("Walk yaw +0.5",  "walk",   [0, 0, 0.5]),
    ("Walk yaw -0.5",  "walk",   [0, 0, -0.5]),
    ("Walk combo",     "walk",   [0.3, 0.2, 0.3]),
    ("Run 1.0",        "run",    [1.0, 0, 0]),
    ("Run 2.0",        "run",    [2.0, 0, 0]),
    ("Run 1.0 lat",    "run",    [1.0, 0.3, 0]),
    ("Run 1.0 yaw",    "run",    [1.0, 0, 0.3]),
    ("Crouch",         "crouch", [0, 0, 0]),
    ("Jump",           "jump",   [0, 0, 0]),
]


SKILL_MODES = ["stand", "walk", "run", "crouch", "jump"]


def evaluate(model_path: str, n_trials: int = 10, episode_len: int = 500):
    env = MiniCheetahEnv(render_mode=None, randomize_domain=False)
    env.randomize_commands = False  # Prevent mid-episode re-randomization
    model = PPO.load(model_path, device="cpu")
    print(f"Model: {model_path}")
    print(f"{'Scenario':20s} | {'avg_vx':>8s} {'avg_vy':>8s} {'avg_wz':>8s} {'avg_h':>7s} {'dist':>6s} {'hdg°':>6s} {'surv':>5s}")
    print("-" * 80)

    for name, mode_str, cmd in SCENARIOS:
        all_vx, all_vy, all_wz, all_h, all_surv, all_dist, all_heading = [], [], [], [], [], [], []

        for trial in range(n_trials):
            obs, _ = env.reset()
            # v22: Use set_command API (handles height ramp, jump trajectory, etc.)
            env.unwrapped.set_command(float(cmd[0]), float(cmd[1]), float(cmd[2]), mode_str)
            obs = env.unwrapped._get_obs()

            vx_hist, vy_hist, wz_hist, h_hist = [], [], [], []
            start_pos = env.unwrapped.data.qpos[:2].copy()
            for step in range(episode_len):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                d = env.unwrapped.data
                # Use body-frame velocities (like reward function does)
                quat = d.qpos[3:7]
                base_linvel = env.unwrapped._quat_rotate_inv(quat, d.qvel[:3])
                vx_hist.append(base_linvel[0])
                vy_hist.append(base_linvel[1])
                wz_hist.append(d.qvel[5])  # yaw rate is frame-independent
                h_hist.append(d.qpos[2])
                if term or trunc:
                    break

            # Average over last 50% of episode (after transients settle)
            half = max(len(vx_hist) // 2, 1)
            all_vx.append(np.mean(vx_hist[half:]))
            all_vy.append(np.mean(vy_hist[half:]))
            all_wz.append(np.mean(wz_hist[half:]))
            all_h.append(np.mean(h_hist[half:]))
            all_surv.append(len(vx_hist))
            # Track displacement
            end_pos = env.unwrapped.data.qpos[:2].copy()
            disp = end_pos - start_pos
            all_dist.append(np.linalg.norm(disp))
            all_heading.append(np.degrees(np.arctan2(disp[1], disp[0])))

        print(f"{name:20s} | {np.mean(all_vx):+8.3f} {np.mean(all_vy):+8.3f} "
              f"{np.mean(all_wz):+8.3f} {np.mean(all_h):7.3f} {np.mean(all_dist):6.2f} "
              f"{np.mean(all_heading):+6.1f} {np.mean(all_surv):5.0f}")

    env.close()


def resolve_model_path(path: str) -> str:
    """If path is a directory, find the latest checkpoint."""
    if os.path.isdir(path):
        ckpts = glob.glob(os.path.join(path, "cheetah_ppo_*_steps.zip"))
        if not ckpts:
            # Try final
            final = os.path.join(path, "cheetah_final.zip")
            if os.path.exists(final):
                return final
            raise FileNotFoundError(f"No checkpoints in {path}")
        ckpts.sort(key=lambda p: int(re.search(r"(\d+)_steps", p).group(1)))
        return ckpts[-1]
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model .zip path or checkpoint directory")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    evaluate(model_path, n_trials=args.trials, episode_len=args.steps)
