"""
Run a single terrain environment interactively or headlessly.

Usage:
    # Random policy, 500 steps, no rendering (sanity check):
    python terrain_testing/scripts/run_single_terrain.py --terrain rubble_field --steps 500

    # Interactive with MuJoCo viewer:
    python terrain_testing/scripts/run_single_terrain.py --terrain pyramid_stairs --render human

    # Load a trained SB3 checkpoint:
    python terrain_testing/scripts/run_single_terrain.py \
        --terrain dreamwaq_rough \
        --checkpoint checkpoints/best_model.zip \
        --steps 2000 \
        --skill trot

    # Fixed difficulty (no randomization):
    python terrain_testing/scripts/run_single_terrain.py \
        --terrain parkour_gap \
        --difficulty 0.8 \
        --fixed-difficulty \
        --skill jump \
        --steps 1000
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TERRAIN_TESTING)
for _p in [_TERRAIN_TESTING, _REPO_ROOT, os.path.join(_REPO_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from maps.map_registry import list_terrains
from envs.base_terrain_wrapper import BaseTerrainWrapper

def load_policy(checkpoint_path, vecnorm_path=None):
    """Load SB3 PPO checkpoint with optional VecNormalize stats.

    If vecnorm_path points to a valid .pkl the policy will normalise
    observations exactly as during training — critical for correct predictions.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    _vecnorm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats: {vecnorm_path}")
        try:
            import pickle
            with open(vecnorm_path, "rb") as f:
                _vecnorm = pickle.load(f)
            _vecnorm.training = False
            _vecnorm.norm_reward = False
        except Exception as e:
            print(f"[warn] Could not load vecnorm ({e}); using raw observations")
            _vecnorm = None

    class _WrappedPolicy:
        """Thin shim that normalises obs before calling model.predict."""
        def __init__(self, m, vn):
            self._model = m
            self._vn = vn

        def predict(self, obs, deterministic=True):
            if self._vn is not None:
                obs = self._vn.normalize_obs(np.array(obs, dtype=np.float32))
            return self._model.predict(obs, deterministic=deterministic)

    return _WrappedPolicy(model, _vecnorm)

def run_episode(env, policy=None, max_steps=2000, verbose=True):
    """Run one episode. Returns dict of metrics."""
    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    forward_vels = []
    reward_components_sum = {}

    t0 = time.time()
    for step in range(max_steps):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Track forward velocity from info
        base_pos = info.get("base_pos", [0, 0, 0])
        fwd = info.get("command", [0])[0]
        forward_vels.append(fwd)

        # Accumulate reward components
        for k, v in info.get("reward_components", {}).items():
            reward_components_sum[k] = reward_components_sum.get(k, 0.0) + v

        if terminated or truncated:
            break

    elapsed = time.time() - t0
    n_steps = step + 1
    survival = not terminated   # True if episode ended by truncation, not fall

    metrics = {
        "steps": n_steps,
        "total_reward": total_reward,
        "mean_reward_per_step": total_reward / max(n_steps, 1),
        "survived": survival,
        "elapsed_s": elapsed,
        "steps_per_sec": n_steps / max(elapsed, 1e-6),
        "mean_reward_components": {
            k: v / max(n_steps, 1) for k, v in reward_components_sum.items()
        },
    }

    if verbose:
        status = "SURVIVED" if survival else "FELL"
        print(f"  [{status}] steps={n_steps:4d}  "
              f"total_reward={total_reward:+.2f}  "
              f"mean/step={metrics['mean_reward_per_step']:+.4f}  "
              f"speed={metrics['steps_per_sec']:.0f} steps/s")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run single terrain environment")
    parser.add_argument("--terrain", default="flat",
                        choices=list_terrains(),
                        help="Terrain name from registry")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--fixed-difficulty", action="store_true",
                        help="Lock difficulty (no randomization)")
    parser.add_argument("--skill", default=None,
                        choices=["walk", "run", "jump", "stand", None],
                        help="Lock skill mode (default: randomize). "
                             "Note: 'trot'/'crouch' are not valid SKILL_MODES.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--render", default="none",
                        choices=["none", "human", "rgb_array"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SB3 PPO checkpoint (.zip)")
    parser.add_argument("--vecnorm", default=None,
                        help="Path to VecNormalize .pkl (default: auto-detect "
                             "checkpoints/vec_normalize.pkl beside checkpoint)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Terrain  : {args.terrain}")
    print(f"Difficulty: {args.difficulty} (fixed={args.fixed_difficulty})")
    print(f"Skill    : {args.skill or 'random'}")
    print(f"Policy   : {'checkpoint' if args.checkpoint else 'random'}")
    print(f"Episodes : {args.episodes}  |  Max steps: {args.steps}")
    print(f"{'=' * 60}\n")

    env = BaseTerrainWrapper(
        terrain_name=args.terrain,
        difficulty=args.difficulty,
        fixed_difficulty=args.fixed_difficulty,
        fixed_skill=args.skill,
        render_mode=args.render,
        randomize_domain=True,
    )

    policy = None
    if args.checkpoint:
        # Auto-detect vecnorm: look beside checkpoint, then at project default
        vecnorm_path = args.vecnorm
        if vecnorm_path is None:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
            candidates = [
                os.path.join(ckpt_dir, "vec_normalize.pkl"),
                os.path.join(_REPO_ROOT, "checkpoints", "vec_normalize.pkl"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    vecnorm_path = c
                    break
        policy = load_policy(args.checkpoint, vecnorm_path=vecnorm_path)

    all_metrics = []
    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}:")
        m = run_episode(env, policy=policy, max_steps=args.steps)
        all_metrics.append(m)

    env.close()

    # Summary
    print(f"\n{'─' * 50}")
    print("Summary across episodes:")
    print(f"  Survival rate  : {sum(m['survived'] for m in all_metrics)}/{args.episodes}")
    print(f"  Mean total rew : {np.mean([m['total_reward'] for m in all_metrics]):.2f}")
    print(f"  Mean rew/step  : {np.mean([m['mean_reward_per_step'] for m in all_metrics]):.4f}")
    print(f"  Mean ep length : {np.mean([m['steps'] for m in all_metrics]):.0f}")
    print(f"  Mean speed     : {np.mean([m['steps_per_sec'] for m in all_metrics]):.0f} steps/s")
    print(f"{'─' * 50}")

    # Print per-term reward breakdown if available
    if all_metrics and all_metrics[0]["mean_reward_components"]:
        print("\nMean reward components (last episode):")
        for k, v in sorted(all_metrics[-1]["mean_reward_components"].items()):
            print(f"  {k:25s}: {v:+.4f}")


if __name__ == "__main__":
    main()
