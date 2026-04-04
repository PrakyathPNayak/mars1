"""
PPO training pipeline for Unitree Go1 quadruped locomotion.

Architecture: Actor/Critic MLP [2048, 1024, 512] with ELU activation
Algorithm: PPO (clip=0.2, lr=3e-4→1e-5 linear, 4096 steps/rollout)

Key design choices (from legged_gym / Isaac Gym conventions):
  - CPU device: SB3 MLP PPO is faster on CPU than GPU for small networks
  - VecNormalize: running observation normalization (critical for locomotion)
  - ELU activation + orthogonal init: better gradient flow for locomotion
  - log_std_init=-1.0: initial exploration std≈0.37 (tight for ±0.5 actions)
  - LR linear annealing: decays from 3e-4 to min 1e-5 over training
  - n_epochs=10, batch_size=4096: 240 gradient steps per rollout update

Hardware note: n_envs=24 targets i9-14900K (8P+16E = 24 physical cores).
SB3 SubprocVecEnv runs one MuJoCo env per process.

Training budget:
  - 24 envs × 4096 n_steps = 98,304 samples/rollout
  - batch_size=4096 → 24 minibatches × 10 epochs = 240 grad steps/update
  - At 100M steps: ~1,017 rollouts × 240 = ~244K total gradient steps
  - Papers: Walk These Ways uses ~1B steps with 4096 GPU envs;
    100M CPU steps with 24 envs provides comparable policy training.

Usage: python3 src/training/train.py --total-steps 100000000
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
            episode_length=2000,
            **kwargs
        )
        env.reset(seed=rank)
        return env
    return _init


def linear_schedule(initial_value: float, min_value: float = 1e-5):
    """Linear decay from initial_value to min_value over training.

    The floor prevents learning from stopping entirely in the final
    phase of training (previous runs showed clip_fraction → 0 and
    std freezing when LR decayed unclamped to 0).
    """
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, min_value)
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

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best").mkdir(parents=True, exist_ok=True)

    class DelayedEvalCallback(EvalCallback):
        """EvalCallback that skips evaluation until min_timesteps have elapsed.

        This prevents an untrained deterministic policy (which stands still
        and collects survival-multiplied reward over full 2000-step episodes)
        from being saved as the "best" model before the policy has had any
        meaningful training.
        """

        def __init__(self, *args, min_timesteps: int = 200_000, **kwargs):
            super().__init__(*args, **kwargs)
            self._min_timesteps = min_timesteps

        def _on_step(self) -> bool:
            if self.num_timesteps < self._min_timesteps:
                return True  # skip eval, continue training
            return super()._on_step()

    class ProgressLogger(BaseCallback):
        """Prints a one-line training summary to console every `log_every` timesteps.

        Tracks completed episodes (detected via VecMonitor's info["episode"] key)
        and reports mean/max episode reward and mean episode length so progress
        is visible even when SB3's built-in table scrolls past.
        """

        def __init__(self, total_steps: int, rollout_size: int, log_every: int = 20_000):
            super().__init__(verbose=0)
            self._total_steps = total_steps
            self._rollout_size = max(rollout_size, 1)
            self._log_every = log_every
            self._ep_rewards: list[float] = []
            self._ep_lengths: list[int] = []
            self._last_log_ts: int = 0

        def _on_step(self) -> bool:
            # VecMonitor adds info["episode"] = {"r": float, "l": int} on done
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self._ep_rewards.append(float(ep["r"]))
                    self._ep_lengths.append(int(ep["l"]))

            if self.num_timesteps - self._last_log_ts >= self._log_every:
                self._last_log_ts = self.num_timesteps
                pct = 100.0 * self.num_timesteps / self._total_steps
                rollout_n = self.num_timesteps // self._rollout_size

                if self._ep_rewards:
                    recent_rew = self._ep_rewards[-100:]
                    recent_len = self._ep_lengths[-100:]
                    mean_rew = sum(recent_rew) / len(recent_rew)
                    max_rew  = max(recent_rew)
                    mean_len = sum(recent_len) / len(recent_len)
                    ep_str = (
                        f"ep_rew={mean_rew:+7.2f} (max={max_rew:+7.2f})  "
                        f"ep_len={mean_len:.0f}  n_ep={len(self._ep_rewards)}"
                    )
                else:
                    ep_str = "ep_rew=n/a (no completed episodes yet)"

                print(
                    f"[{time.strftime('%H:%M:%S')}]  "
                    f"step {self.num_timesteps:>10,d}/{self._total_steps:,}  "
                    f"({pct:5.1f}%)  rollout #{rollout_n:,}  |  {ep_str}"
                )
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
        net_arch=dict(pi=[2048, 1024, 512], vf=[2048, 1024, 512]),
        activation_fn=torch.nn.ELU,    # better gradient flow than Tanh for locomotion
        ortho_init=True,               # orthogonal weight init (PPO standard)
        log_std_init=-1.0,             # initial std ≈ 0.37 (tighter; -0.5 led to frozen std)
    )

    # SB3 MLP PPO: CPU is typically faster for smaller networks due to
    # CPU↔GPU transfer overhead. With batch_size=512 and n_envs=8,
    # GPU can help further. Use --device cuda to try GPU.
    device = args.device
    print(f"  Device: {device}")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device)
    else:
        n_epochs = getattr(args, "n_epochs", 10)
        # n_steps=4096: long rollouts improve GAE advantage estimates for locomotion.
        # batch_size=4096: 98304 samples / 4096 = 24 minibatches per epoch.
        # n_epochs=10: 24 × 10 = 240 gradient steps per rollout update.
        # With 24 envs (i9-14900K): 3× more samples per rollout than 8 envs.
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_schedule(3e-4),
            n_steps=4096,
            batch_size=4096,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=1.0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=1,
            device=device,
        )

    rollout_size = model.n_steps * args.n_envs
    minibatches_per_epoch = rollout_size // model.batch_size
    grad_steps_per_rollout = minibatches_per_epoch * model.n_epochs
    total_rollouts = args.total_steps // rollout_size
    total_grad_steps = total_rollouts * grad_steps_per_rollout

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="cheetah_ppo",
            verbose=1,
        ),
        DelayedEvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=10,
            min_timesteps=200_000,
            verbose=1,
        ),
        ProgressLogger(
            total_steps=args.total_steps,
            rollout_size=rollout_size,
        ),
        RewardComponentCallback(log_dir=str(log_dir), verbose=1),
    ]

    print(f"Training PPO for {args.total_steps:,} total steps")
    print(f"  Envs: {args.n_envs}, Device: {model.device}")
    print(f"  Logging: {log_dir}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"\n  Gradient update budget:")
    print(f"    rollout size      = {args.n_envs} envs × {model.n_steps} steps = {rollout_size:,} samples")
    print(f"    minibatches/epoch = {rollout_size:,} / {model.batch_size} = {minibatches_per_epoch}")
    print(f"    grad steps/update = {minibatches_per_epoch} × {model.n_epochs} epochs = {grad_steps_per_rollout:,}")
    print(f"    total rollouts    = {total_rollouts:,}")
    print(f"    total grad steps  = {total_grad_steps:,}")

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
        "net_arch": "pi=[2048,1024,512], vf=[2048,1024,512]",
        "activation_fn": "ELU",
        "lr": "3e-4 linear decay (min 1e-5)",
        "clip": 0.2,
        "batch_size": 4096,
        "n_steps": 4096,
        "n_epochs": getattr(args, "n_epochs", 10),
        "ent_coef": 0.01,
        "log_std_init": -1.0,
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
    parser = argparse.ArgumentParser(description="Train Unitree Go1 PPO")
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO gradient update epochs per rollout (default: 10)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, or auto (auto uses GPU if available)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--log-dir", type=str, default="logs/training",
                        help="Directory for TensorBoard logs (default: logs/training)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
