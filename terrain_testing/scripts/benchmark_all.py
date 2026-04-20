"""
Benchmark all registered terrains and save results as CSV.

Usage:
    # Fast smoke-test (random policy, 200 steps per terrain):
    python terrain_testing/scripts/benchmark_all.py --steps 200

    # Full benchmark with random policy:
    python terrain_testing/scripts/benchmark_all.py --steps 2000 --episodes 5

    # Full benchmark with trained checkpoint:
    python terrain_testing/scripts/benchmark_all.py \
        --steps 2000 \
        --episodes 5 \
        --checkpoint checkpoints/best_model.zip \
        --output results/full_benchmark.csv

    # Only paper terrains:
    python terrain_testing/scripts/benchmark_all.py --source paper --steps 500

    # Only custom terrains:
    python terrain_testing/scripts/benchmark_all.py --source custom --steps 500
"""

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TERRAIN_TESTING)
for _p in [_TERRAIN_TESTING, _REPO_ROOT, os.path.join(_REPO_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from maps.map_registry import REGISTRY, list_terrains
from envs.base_terrain_wrapper import BaseTerrainWrapper


_REWARD_COMPONENT_KEYS = [
    "r_linvel", "r_yaw", "r_height", "r_orientation",
    "r_lin_vel_z", "r_ang_vel_xy", "r_torque", "r_action_rate",
    "r_joint_acc", "r_joint_limit", "r_contact", "r_terrain",
    "r_collision", "r_stumble", "r_total",
]


def load_policy(checkpoint_path):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 not installed. Falling back to random policy.")
        return None
    print(f"  Loading checkpoint: {checkpoint_path}")
    return PPO.load(checkpoint_path)


def run_episodes(env, policy, n_episodes, max_steps):
    """Run n_episodes on env, return aggregated metrics dict."""
    ep_rewards, ep_lengths, ep_survived = [], [], []
    ep_fwd_vels = []
    component_sums = {k: [] for k in _REWARD_COMPONENT_KEYS}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        step = 0
        fwd_vels = []
        comp_ep = {k: 0.0 for k in _REWARD_COMPONENT_KEYS}
        terminated = False

        for step in range(max_steps):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward

            # Forward velocity from base_pos delta approximation or command
            cmd = info.get("command", [0.0, 0.0, 0.0])
            fwd_vels.append(cmd[0])

            for k in _REWARD_COMPONENT_KEYS:
                comp_ep[k] += info.get("reward_components", {}).get(k, 0.0)

            if terminated or truncated:
                break

        n_steps = step + 1
        ep_rewards.append(total_r)
        ep_lengths.append(n_steps)
        ep_survived.append(not terminated)
        ep_fwd_vels.append(np.mean(fwd_vels) if fwd_vels else 0.0)
        for k in _REWARD_COMPONENT_KEYS:
            component_sums[k].append(comp_ep[k] / max(n_steps, 1))

    metrics = {
        "mean_ep_reward": float(np.mean(ep_rewards)),
        "std_ep_reward": float(np.std(ep_rewards)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "survival_rate": float(np.mean(ep_survived)),
        "fall_rate": float(1.0 - np.mean(ep_survived)),
        "mean_forward_vel_cmd": float(np.mean(ep_fwd_vels)),
    }
    for k in _REWARD_COMPONENT_KEYS:
        metrics[f"comp_{k}"] = float(np.mean(component_sums[k]))

    return metrics


def print_table(results: list[dict]):
    """Pretty-print benchmark results as a table."""
    cols = ["terrain", "source", "survival_rate", "mean_ep_reward",
            "mean_ep_length", "fall_rate"]
    col_w = [26, 8, 14, 16, 16, 10]
    header = "".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
    print("\n" + "=" * sum(col_w))
    print(header)
    print("─" * sum(col_w))
    for r in results:
        row = [
            r.get("terrain", ""),
            r.get("source", ""),
            f"{r.get('survival_rate', 0):.2%}",
            f"{r.get('mean_ep_reward', 0):+.2f}",
            f"{r.get('mean_ep_length', 0):.0f}",
            f"{r.get('fall_rate', 0):.2%}",
        ]
        print("".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    print("=" * sum(col_w))


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all terrain environments")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per terrain")
    parser.add_argument("--source", default=None, choices=["paper", "custom"],
                        help="Filter terrains by source")
    parser.add_argument("--terrains", nargs="+", default=None,
                        help="Specific terrain names (default: all)")
    parser.add_argument("--difficulty", type=float, default=0.5,
                        help="Fixed difficulty for all terrains")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SB3 PPO checkpoint (.zip)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/benchmark_<timestamp>.csv)")
    parser.add_argument("--skill", default=None,
                        choices=["walk", "trot", "run", "jump", "crouch", "stand"],
                        help="Lock skill mode for all terrains")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-domain-rand", action="store_true",
                        help="Disable domain randomization")
    args = parser.parse_args()

    terrain_names = args.terrains or list_terrains(args.source)

    if args.output is None:
        ts = int(time.time())
        args.output = os.path.join(_TERRAIN_TESTING, "results",
                                   f"benchmark_{ts}.csv")

    print(f"\nBenchmarking {len(terrain_names)} terrains")
    print(f"Policy    : {'checkpoint' if args.checkpoint else 'random'}")
    print(f"Episodes  : {args.episodes}  |  Max steps: {args.steps}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Output    : {args.output}\n")

    policy = load_policy(args.checkpoint) if args.checkpoint else None

    results = []
    for idx, name in enumerate(terrain_names):
        entry = REGISTRY[name]
        print(f"[{idx + 1:02d}/{len(terrain_names)}] {name} ({entry.source}) ... ", end="", flush=True)
        t0 = time.time()
        try:
            env = BaseTerrainWrapper(
                terrain_name=name,
                difficulty=args.difficulty,
                fixed_difficulty=True,
                fixed_skill=args.skill,
                render_mode="none",
                randomize_domain=not args.no_domain_rand,
            )
            metrics = run_episodes(env, policy, args.episodes, args.steps)
            env.close()
            elapsed = time.time() - t0
            print(f"surv={metrics['survival_rate']:.0%}  "
                  f"rew={metrics['mean_ep_reward']:+.1f}  "
                  f"len={metrics['mean_ep_length']:.0f}  "
                  f"({elapsed:.1f}s)")
            row = {"terrain": name, "source": entry.source,
                   "paper": entry.paper, **metrics}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            row = {"terrain": name, "source": entry.source,
                   "paper": entry.paper, "error": str(e)}

        results.append(row)

    print_table([r for r in results if "error" not in r])

    if any("error" in r for r in results):
        print("\nFailed terrains:")
        for r in results:
            if "error" in r:
                print(f"  {r['terrain']}: {r['error']}")

    save_csv(results, args.output)


if __name__ == "__main__":
    main()
