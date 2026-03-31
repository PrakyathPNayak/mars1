"""
PPO training pipeline for MIT Mini Cheetah locomotion.

Architecture: Actor/Critic MLP [512, 256, 128]
Algorithm: PPO (clip=0.2, lr=3e-4, 2048 steps/rollout)

Usage: python3 src/training/train.py --total-steps 5000000
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def make_env(rank=0, **kwargs):
    """Environment factory for vectorized training."""
    # Capture absolute project root for subprocess safety
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    def _init():
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # Change to project root so relative paths work
        import os
        os.chdir(project_root)
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(
            render_mode="none",
            randomize_domain=True,
            episode_length=1000,
            **kwargs
        )
        env.reset(seed=rank)
        return env
    return _init


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    from src.training.reward_logger import RewardComponentCallback

    log_dir = Path("logs/training")
    ckpt_dir = Path("checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    class ProgressLogger(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_log = 0

        def _on_step(self):
            if self.n_calls - self._last_log >= 10000:
                self._last_log = self.n_calls
                try:
                    ep_rew = self.locals.get('rollout_buffer')
                    with open(".state/PROGRESS.md", "a") as f:
                        f.write(f"[{time.strftime('%H:%M:%S')}] Step {self.n_calls:>8,d} | "
                                f"timesteps={self.num_timesteps:,}\n")
                except Exception:
                    pass
            return True

    print(f"Creating {args.n_envs} parallel environments...")
    # Try SubprocVecEnv first, fall back to DummyVecEnv
    try:
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
        print("  Using SubprocVecEnv (parallel)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env, str(log_dir))

    # Eval env: single env, wrapped with VecMonitor to match training env type
    eval_env = VecMonitor(DummyVecEnv([make_env(999)]))

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=torch.nn.Tanh,
    )

    # MlpPolicy runs faster on CPU — GPU overhead dominates for small networks.
    # The advanced transformer training script uses GPU where it's worthwhile.
    device = "cpu"
    print(f"  Device: {device}")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=1,
            device=device,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="cheetah_ppo",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=5,
            verbose=1,
        ),
        ProgressLogger(),
        RewardComponentCallback(log_dir=str(log_dir), verbose=1),
    ]

    print(f"Training PPO for {args.total_steps:,} total steps")
    print(f"  Envs: {args.n_envs}, Device: {model.device}")
    print(f"  Logging: {log_dir}")
    print(f"  Checkpoints: {ckpt_dir}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = str(ckpt_dir / "cheetah_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved: {final_path}")

    config = {
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "algorithm": "PPO",
        "net_arch": "pi=[512,256,128], vf=[512,256,128]",
        "lr": 3e-4,
        "clip": 0.2,
        "batch_size": 256,
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    vec_env.close()
    eval_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train MIT Mini Cheetah PPO")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
