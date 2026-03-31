"""
Advanced PPO training with Hierarchical Transformer Policy.

Architecture: Hierarchical Transformer + MoE + Symmetry + World Model
Algorithm: PPO (clip=0.2, lr=3e-4, 2048 steps/rollout)

Features:
  - Sensory-group attention encoder (MSTA-inspired)
  - Temporal causal transformer (SET/TERT-inspired)
  - Morphological symmetry augmentation (MS-PPO-inspired)
  - Mixture of Experts action head (MoE-Loco-inspired)
  - Auxiliary world model loss (DWL-inspired)
  - Learning Progress adaptive curriculum (LP-ACRL-inspired)

Usage: python3 src/training/train_advanced.py --total-steps 5000000
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def make_env(rank=0, history_len=16, **kwargs):
    """Environment factory with history wrapper for vectorized training."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    def _init():
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import os
        os.chdir(project_root)
        from src.env.cheetah_env import MiniCheetahEnv
        from src.training.sb3_integration import HistoryWrapper, ActionSmoothingWrapper
        env = MiniCheetahEnv(
            render_mode="none",
            randomize_domain=True,
            episode_length=1000,
            **kwargs
        )
        env = ActionSmoothingWrapper(env, alpha=0.8)
        env = HistoryWrapper(env, history_len=history_len)
        env.reset(seed=rank)
        return env
    return _init


def train(args):
    import torch
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    from src.training.sb3_integration import (
        TransformerActorCriticPolicy,
        WorldModelCallback,
        CurriculumCallback,
    )
    from src.training.advanced_policy import AdaptiveCurriculum

    # Learning rate schedule with warmup (inspired by TERT, SET papers)
    def lr_schedule_with_warmup(progress_remaining: float) -> float:
        """Linear warmup for first 5% of training, then cosine decay."""
        progress = 1.0 - progress_remaining
        warmup_frac = 0.05
        if progress < warmup_frac:
            # Linear warmup
            return progress / warmup_frac
        else:
            # Cosine decay
            decay_progress = (progress - warmup_frac) / (1.0 - warmup_frac)
            return 0.5 * (1.0 + np.cos(np.pi * decay_progress))


    log_dir = Path("logs/training_advanced")
    ckpt_dir = Path("checkpoints/advanced")
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
                    with open(".state/PROGRESS.md", "a") as f:
                        f.write(
                            f"[{time.strftime('%H:%M:%S')}] Step {self.n_calls:>8,d} | "
                            f"timesteps={self.num_timesteps:,}\n"
                        )
                except Exception:
                    pass
            return True

    # Hyperparameters
    d_model = args.d_model
    n_layers = args.n_layers
    n_experts = args.n_experts
    history_len = args.history_len

    print(f"=" * 60)
    print(f"ADVANCED TRAINING: Hierarchical Transformer + MoE")
    print(f"=" * 60)
    print(f"  d_model={d_model}, layers={n_layers}, experts={n_experts}")
    print(f"  history_len={history_len}")
    print(f"  envs={args.n_envs}, steps={args.total_steps:,}")
    print(f"=" * 60)

    # Device selection: use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"\nCreating {args.n_envs} parallel environments with history...")
    try:
        vec_env = SubprocVecEnv(
            [make_env(i, history_len=history_len) for i in range(args.n_envs)]
        )
        print("  Using SubprocVecEnv (parallel)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv(
            [make_env(i, history_len=history_len) for i in range(args.n_envs)]
        )
    vec_env = VecMonitor(vec_env, str(log_dir))

    eval_env = DummyVecEnv([make_env(999, history_len=history_len)])

    # Policy kwargs for the transformer
    policy_kwargs = dict(
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        n_experts=n_experts,
        history_len=history_len,
        obs_dim=48,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device)
    else:
        model = PPO(
            policy=TransformerActorCriticPolicy,
            env=vec_env,
            learning_rate=lambda p: 3e-4 * lr_schedule_with_warmup(p),
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

    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    train_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    # Adaptive curriculum
    curriculum = AdaptiveCurriculum(n_envs=args.n_envs)

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="cheetah_transformer",
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
        WorldModelCallback(
            wm_lr=1e-4,
            wm_coeff=0.1,
            update_freq=2048,
            verbose=1 if args.verbose else 0,
        ),
        CurriculumCallback(curriculum, verbose=1 if args.verbose else 0),
        ProgressLogger(),
    ]

    print(f"\nStarting training for {args.total_steps:,} steps...")
    print(f"  Logging: {log_dir}")
    print(f"  Checkpoints: {ckpt_dir}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = str(ckpt_dir / "cheetah_transformer_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved: {final_path}")

    # Save config
    config = {
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "algorithm": "PPO",
        "policy": "TransformerActorCriticPolicy",
        "d_model": d_model,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "history_len": history_len,
        "lr": 3e-4,
        "clip": 0.2,
        "batch_size": 256,
        "device": device,
        "total_params": total_params,
        "trainable_params": train_params,
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    vec_env.close()
    eval_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Mini Cheetah with Hierarchical Transformer + MoE"
    )
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
