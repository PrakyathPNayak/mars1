"""
Training script for terrain-aware locomotion.

Uses AdvancedTerrainEnv with AdvancedTerrainCurriculum
and TransformerActorCriticPolicy.

Usage:
  python3 scripts/train_terrain.py --total-steps 5000000
  python3 scripts/train_terrain.py --total-steps 500000 --n-envs 4 --quick
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def make_terrain_env(rank=0, history_len=16, **env_kwargs):
    """Factory for AdvancedTerrainEnv with history wrapper."""
    def _init():
        import sys
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        os.chdir(PROJECT_ROOT)
        from src.env.terrain_env import AdvancedTerrainEnv
        from src.training.sb3_integration import HistoryWrapper, ActionSmoothingWrapper

        env = AdvancedTerrainEnv(
            render_mode="none",
            randomize_domain=True,
            randomize_terrain=True,
            randomize_skill=True,
            episode_length=1000,
            push_interval=200,
            push_magnitude=0.5,
            **env_kwargs,
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
        EvalCallback, CheckpointCallback, BaseCallback,
    )
    from src.training.sb3_integration import (
        TransformerActorCriticPolicy,
        WorldModelCallback,
    )
    from src.training.curriculum import AdvancedTerrainCurriculum

    def lr_schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        warmup = 0.05
        if progress < warmup:
            return max(progress / warmup, 1e-6)
        decay = (progress - warmup) / (1.0 - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * decay))

    log_dir = Path("logs/terrain_training")
    ckpt_dir = Path("checkpoints/terrain")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Terrain obs = 57; for the history wrapper: obs_dim is the terrain obs dim
    obs_dim = 57
    history_len = args.history_len

    print("=" * 60)
    print("TERRAIN TRAINING: AdvancedTerrainEnv + Transformer")
    print("=" * 60)
    print(f"  d_model={args.d_model}, layers={args.n_layers}, experts={args.n_experts}")
    print(f"  history_len={history_len}, obs_dim={obs_dim}")
    print(f"  envs={args.n_envs}, steps={args.total_steps:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print(f"\nCreating {args.n_envs} terrain environments...")
    try:
        vec_env = SubprocVecEnv([
            make_terrain_env(i, history_len=history_len)
            for i in range(args.n_envs)
        ])
        print("  Using SubprocVecEnv")
    except Exception as e:
        print(f"  Falling back to DummyVecEnv ({e})")
        vec_env = DummyVecEnv([
            make_terrain_env(i, history_len=history_len)
            for i in range(args.n_envs)
        ])
    vec_env = VecMonitor(vec_env, str(log_dir))

    eval_env = DummyVecEnv([make_terrain_env(999, history_len=history_len)])

    policy_kwargs = dict(
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        history_len=history_len,
        obs_dim=obs_dim,
    )

    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=vec_env,
        learning_rate=lambda p: 3e-4 * lr_schedule(p),
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

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # Curriculum
    curriculum = AdvancedTerrainCurriculum(n_envs=args.n_envs)

    class TerrainCurriculumCallback(BaseCallback):
        def __init__(self, curriculum, verbose=0):
            super().__init__(verbose)
            self.curriculum = curriculum

        def _on_step(self):
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for i, (info, done) in enumerate(zip(infos, dones)):
                if done:
                    ep_rew = info.get("episode", {}).get("r", 0.0)
                    success = ep_rew > 50.0
                    self.curriculum.record_episode(
                        i % self.curriculum.n_envs, success, ep_rew
                    )
            if self.n_calls % 10000 == 0 and self.verbose:
                print(f"  {self.curriculum.summary()}")
            return True

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="terrain_transformer",
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
            wm_lr=1e-4, wm_coeff=0.1, update_freq=2048, verbose=0,
        ),
        TerrainCurriculumCallback(curriculum, verbose=1),
    ]

    print(f"\nStarting training for {args.total_steps:,} steps...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = str(ckpt_dir / "terrain_transformer_final")
    model.save(final_path)
    print(f"\nTraining complete. Model saved: {final_path}")

    config = {
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "algorithm": "PPO",
        "policy": "TransformerActorCriticPolicy",
        "env": "AdvancedTerrainEnv",
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_experts": args.n_experts,
        "history_len": history_len,
        "obs_dim": obs_dim,
        "total_params": total_params,
    }
    with open(ckpt_dir / "terrain_config.json", "w") as f:
        json.dump(config, f, indent=2)

    vec_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train on terrain environments")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.total_steps = min(args.total_steps, 500_000)
        args.n_envs = min(args.n_envs, 4)

    train(args)


if __name__ == "__main__":
    main()
