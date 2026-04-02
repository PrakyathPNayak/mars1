"""
Record a video of the trained policy.
Usage: python3 scripts/record_video.py --checkpoint checkpoints/best/best_model.zip
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def record(checkpoint_path, output_path="logs/rollout.mp4", n_episodes=1, fps=50):
    from src.env.cheetah_env import MiniCheetahEnv
    from src.utils.policy_loader import load_policy_for_inference
    import imageio

    env = MiniCheetahEnv(
        render_mode="rgb_array",
        randomize_domain=False,
        episode_length=1000,
    )
    policy, normalize_fn = load_policy_for_inference(checkpoint_path)

    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        steps = 0
        while True:
            action, _ = policy.predict(normalize_fn(obs), deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            total_r += reward
            steps += 1
            if done or truncated:
                break
        print(f"Episode {ep+1}: reward={total_r:.2f}, steps={steps}")

    env.close()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=fps)
    print(f"Video saved: {out} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best/best_model.zip")
    parser.add_argument("--output", default="logs/rollout.mp4")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    record(args.checkpoint, args.output, args.episodes)
