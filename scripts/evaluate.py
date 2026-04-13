"""
Evaluate a trained policy on N episodes.
Usage: python3 scripts/evaluate.py --checkpoint checkpoints/best/best_model.zip
"""
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(checkpoint_path, n_episodes=20, render=False):
    from src.env.cheetah_env import MiniCheetahEnv
    from src.utils.policy_loader import load_policy_for_inference

    env = MiniCheetahEnv(
        render_mode="human" if render else "none",
        use_terrain=False,
        episode_length=2000,
    )
    policy, normalize_fn = load_policy_for_inference(checkpoint_path)

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        steps = 0
        while True:
            action, _ = policy.predict(normalize_fn(obs), deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_r += reward
            steps += 1
            if done or truncated:
                break
        results.append({"episode": ep, "reward": total_r, "steps": steps})
        print(f"Episode {ep + 1:3d}: reward={total_r:8.2f}, steps={steps}")

    env.close()

    rewards = [r["reward"] for r in results]
    lengths = [r["steps"] for r in results]
    summary = {
        "checkpoint": checkpoint_path,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_length": float(np.mean(lengths)),
        "survival_rate": float(sum(1 for l in lengths if l >= 999) / len(lengths)),
    }

    print(f"\n{'=' * 50}")
    print("EVALUATION SUMMARY")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    out_path = Path("logs") / f"eval_{int(time.time())}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render)
