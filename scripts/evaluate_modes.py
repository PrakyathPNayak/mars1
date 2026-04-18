#!/usr/bin/env python3
"""
Per-mode evaluation of trained policy with specific commands.

Tests each mode (stand/walk/run/crouch/jump) with targeted velocity commands
and reports tracking accuracy, survival, and reward.

Uses load_policy_for_inference for proper VecNormalize handling.

Usage:
    python3 scripts/evaluate_modes.py --checkpoint checkpoints/v8b/best/best_model.zip
    python3 scripts/evaluate_modes.py  # uses default checkpoint search
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.cheetah_env import MiniCheetahEnv
from src.utils.policy_loader import load_policy_for_inference


# Test scenarios: (mode, vx_cmd, vy_cmd, wz_cmd, description)
TEST_SCENARIOS = [
    ("stand",  0.0,  0.0,  0.0, "stand still"),
    ("walk",   0.5,  0.0,  0.0, "walk forward 0.5"),
    ("walk",   0.0,  0.3,  0.0, "walk lateral 0.3"),
    ("walk",   0.0, -0.3,  0.0, "walk lateral -0.3"),
    ("walk",   0.0,  0.0,  0.5, "walk yaw 0.5"),
    ("walk",   0.0,  0.0, -0.5, "walk yaw -0.5"),
    ("walk",   0.5,  0.3,  0.3, "walk combined"),
    ("walk",   0.8,  0.0,  0.0, "walk fast 0.8"),
    ("run",    1.5,  0.0,  0.0, "run forward 1.5"),
    ("run",    2.0,  0.0,  0.0, "run fast 2.0"),
    ("run",    1.0,  0.3,  0.0, "run + lateral"),
    ("jump",   0.2,  0.0,  0.0, "jump"),
]

HEIGHT_TARGETS = {
    "stand": 0.27, "walk": 0.27, "run": 0.27, "jump": 0.35,
}


def evaluate_scenario(env, policy, normalize_fn, mode, vx, vy, wz, n_steps=500):
    """Run one scenario and return metrics."""
    obs, _ = env.reset()
    env.command_mode = mode
    env.randomize_commands = False
    env.command = np.array([vx, vy, wz], dtype=np.float32)
    env.target_height = HEIGHT_TARGETS[mode]

    # Re-encode skill one-hot in observation after mode change.
    # In v23 env, skill one-hot is at obs[49:53] (4 modes: stand/walk/run/jump).
    skill_idx = {"stand": 0, "walk": 1, "run": 2, "jump": 3}[mode]
    obs[49:53] = 0.0
    obs[49 + skill_idx] = 1.0
    obs[45:49] = [vx, vy, wz, env.target_height]

    total_r = 0.0
    vx_vals, vy_vals, wz_vals, hz_vals = [], [], [], []
    survived = 0

    for step in range(n_steps):
        norm_obs = normalize_fn(obs)
        action, _ = policy.predict(norm_obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        total_r += reward
        survived += 1

        # Body-frame velocities from obs
        vx_vals.append(float(obs[24]))
        vy_vals.append(float(obs[25]))
        wz_vals.append(float(obs[29]))
        hz_vals.append(float(info.get("base_height", obs[2] if len(obs) > 2 else 0)))

        if done:
            break

    n = max(survived, 1)
    # Skip first 50 steps (transient) for tracking metrics
    skip = min(50, n // 2)
    vx_track = np.array(vx_vals[skip:]) if len(vx_vals) > skip else np.array(vx_vals)
    vy_track = np.array(vy_vals[skip:]) if len(vy_vals) > skip else np.array(vy_vals)
    wz_track = np.array(wz_vals[skip:]) if len(wz_vals) > skip else np.array(wz_vals)

    return {
        "survived": survived,
        "reward": total_r,
        "vx_mean": float(np.mean(vx_track)) if len(vx_track) > 0 else 0,
        "vy_mean": float(np.mean(vy_track)) if len(vy_track) > 0 else 0,
        "wz_mean": float(np.mean(wz_track)) if len(wz_track) > 0 else 0,
        "vx_err": float(np.mean(np.abs(vx_track - vx))) if len(vx_track) > 0 else 0,
        "vy_err": float(np.mean(np.abs(vy_track - vy))) if len(vy_track) > 0 else 0,
        "wz_err": float(np.mean(np.abs(wz_track - wz))) if len(wz_track) > 0 else 0,
        "h_mean": float(np.mean(hz_vals)) if hz_vals else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3, help="Repeat each scenario N times")
    args = parser.parse_args()

    policy, normalize_fn = load_policy_for_inference(args.checkpoint)
    if policy is None:
        print("ERROR: No policy found. Provide --checkpoint path.")
        sys.exit(1)

    env = MiniCheetahEnv(render_mode="none", randomize_domain=False, episode_length=args.steps + 100)

    print(f"{'Mode':8s} {'Description':22s} {'Surv':>5s} {'Reward':>8s} "
          f"{'vx_cmd':>7s} {'vx_act':>7s} {'vx_err':>7s} "
          f"{'vy_cmd':>7s} {'vy_act':>7s} {'vy_err':>7s} "
          f"{'wz_cmd':>7s} {'wz_act':>7s} {'wz_err':>7s} "
          f"{'height':>7s}")
    print("-" * 140)

    for mode, vx, vy, wz, desc in TEST_SCENARIOS:
        all_results = []
        for rep in range(args.repeats):
            r = evaluate_scenario(env, policy, normalize_fn, mode, vx, vy, wz, args.steps)
            all_results.append(r)

        # Average across repeats
        avg = {k: np.mean([r[k] for r in all_results]) for k in all_results[0]}

        # Grade tracking quality
        vx_grade = "OK" if avg["vx_err"] < 0.15 else ("WEAK" if avg["vx_err"] < 0.3 else "FAIL")
        vy_grade = "OK" if avg["vy_err"] < 0.1 else ("WEAK" if avg["vy_err"] < 0.2 else "FAIL")
        wz_grade = "OK" if avg["wz_err"] < 0.15 else ("WEAK" if avg["wz_err"] < 0.3 else "FAIL")

        print(f"{mode:8s} {desc:22s} {avg['survived']:5.0f} {avg['reward']:8.1f} "
              f"{vx:+7.2f} {avg['vx_mean']:+7.3f} {avg['vx_err']:7.3f}{vx_grade:>5s} "
              f"{vy:+7.2f} {avg['vy_mean']:+7.3f} {avg['vy_err']:7.3f}{vy_grade:>5s} "
              f"{wz:+7.2f} {avg['wz_mean']:+7.3f} {avg['wz_err']:7.3f}{wz_grade:>5s} "
              f"{avg['h_mean']:7.3f}")

    env.close()


if __name__ == "__main__":
    main()
