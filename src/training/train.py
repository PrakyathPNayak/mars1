"""
PPO training pipeline for Unitree Go1 quadruped locomotion.

Architecture: Actor/Critic MLP [512, 256, 128] with ELU activation
Algorithm: PPO (clip=0.2, lr=3e-4→0 linear, 4096 steps/rollout)

Key design choices (from legged_gym / Isaac Gym conventions):
  - CPU device: SB3 MLP PPO is faster on CPU than GPU for small networks
  - VecNormalize: running observation normalization (critical for locomotion)
  - ELU activation + orthogonal init: better gradient flow for locomotion
  - log_std_init=-0.5: initial exploration std≈0.6 (matches ±0.5 action range)
  - LR linear annealing: decays from 3e-4 to 0 over training

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


def linear_schedule(initial_value: float):
    """Linear decay from initial_value to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize,
    )
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

    # ── Build vectorized environments ────────────────────────────────
    print(f"Creating {args.n_envs} parallel environments...")
    try:
        base_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
        print("  Using SubprocVecEnv (parallel)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        base_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(base_env, str(log_dir))

    # VecNormalize: running observation normalization (critical for locomotion).
    # norm_reward=False because the environment already uses only_positive_rewards
    # clipping and exp-kernel tracking — further normalization would distort.
    norm_path = str(ckpt_dir / "vec_normalize.pkl")
    if args.resume and os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = True
        print(f"  Loaded VecNormalize stats from {norm_path}")
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False,
            clip_obs=100.0, gamma=0.99,
        )

    # Eval env: single env with VecNormalize (training=False, stats synced
    # automatically by EvalCallback via sync_envs_normalization)
    eval_base = VecMonitor(DummyVecEnv([make_env(999)]))
    eval_env = VecNormalize(
        eval_base, norm_obs=True, norm_reward=False,
        training=False, clip_obs=100.0, gamma=0.99,
    )

    # ── Policy architecture ─────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=torch.nn.ELU,    # better gradient flow than Tanh for locomotion
        ortho_init=True,               # orthogonal weight init (PPO standard)
        log_std_init=-0.5,             # initial std ≈ 0.6 (matches ±0.5 action range)
    )

    # SB3 MLP PPO is faster on CPU: GPU memory-transfer overhead dominates
    # the tiny MLP forward pass. See SB3 docs warning for MlpPolicy + GPU.
    device = "cpu"
    print(f"  Device: {device}")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_schedule(3e-4),
            n_steps=4096,
            batch_size=4096,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,      # legged_gym convention; critical with 25 envs (less diversity than 4096)
            vf_coef=0.5,
            max_grad_norm=1.0,
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
    vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
    print(f"\nTraining complete. Model saved: {final_path}")
    print(f"  VecNormalize stats: {ckpt_dir / 'vec_normalize.pkl'}")

    config = {
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "algorithm": "PPO",
        "net_arch": "pi=[512,256,128], vf=[512,256,128]",
        "activation_fn": "ELU",
        "lr": "3e-4 linear decay",
        "clip": 0.2,
        "batch_size": 4096,
        "n_steps": 4096,
        "n_epochs": 5,
        "ent_coef": 0.01,
        "log_std_init": -0.5,
        "ortho_init": True,
        "device": device,
        "robot": "Unitree Go1 (mujoco_menagerie)",
        "vec_normalize": "norm_obs=True, norm_reward=False, clip_obs=100",
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
