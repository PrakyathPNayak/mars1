"""Reward component diagnostic — run 200 steps with zero action and print per-component breakdown.

Usage:
    python3 scripts/diagnose_reward.py
    python3 scripts/diagnose_reward.py --mode crouch
    python3 scripts/diagnose_reward.py --mode stand --steps 500
    just diagnose mode=crouch
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.cheetah_env import HEIGHT_TARGETS, MiniCheetahEnv


def run_diag(mode: str, steps: int = 200) -> None:
    env = MiniCheetahEnv(render_mode="none", randomize_domain=False)
    env.randomize_commands = False
    env.command_mode = mode
    env.target_height = HEIGHT_TARGETS.get(mode, 0.27)
    env.command = np.zeros(3, dtype=np.float32)
    env.reset(seed=42)

    accum: dict[str, list[float]] = {}
    survived = 0
    for i in range(steps):
        _, _, done, _, info = env.step(np.zeros(12, dtype=np.float32))
        survived = i + 1
        for k, v in info.get("reward_components", {}).items():
            accum.setdefault(k, []).append(float(v))
        if done:
            break

    base_z = float(env.data.qpos[2])
    jv = float(np.linalg.norm(env.data.qvel[6:18]))
    print(f"\n=== Reward Diagnostic: mode={mode!r}  steps={survived}/{steps} ===")
    print(f"  base_z = {base_z:.4f} m    joint_vel_mag = {jv:.4f} rad/s")
    print(f"  {'component':<28}  mean/step     sum")
    print("  " + "-" * 56)

    sorted_keys = sorted(accum, key=lambda k: -abs(np.mean(accum[k])))
    for k in sorted_keys:
        vals = accum[k]
        print(f"  {k:<28}  {np.mean(vals):+.4f}      {np.sum(vals):+.2f}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward component diagnostic")
    parser.add_argument(
        "--mode",
        type=str,
        default="stand",
        choices=list(HEIGHT_TARGETS.keys()),
        help="Command mode to test (default: stand)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps (default: 200)",
    )
    args = parser.parse_args()
    run_diag(args.mode, args.steps)


if __name__ == "__main__":
    main()
