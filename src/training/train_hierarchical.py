"""
Hierarchical training pipeline for Unitree Go1 quadruped locomotion.

Combines imitation learning (behavioral cloning) with the advanced
Transformer+MoE architecture for a three-phase training pipeline:

  Phase 1 — Expert data collection: Roll out a pre-trained (MLP PPO) expert
            and record (observation_history, action) pairs.
  Phase 2 — Behavioral cloning (BC): Supervised pre-training of the
            Transformer encoder + MLP head to mimic the expert, giving
            the temporal encoder a warm-start on real locomotion data.
  Phase 3 — Full PPO + Transformer+MoE: Continue training with the full
            hierarchical policy (sensory-group attention, symmetry
            augmentation, MoE action head, world-model auxiliary loss,
            adaptive curriculum).

References:
  - ULT (2503.08997), TERT (2212.07740): Transformer for locomotion
  - Walk These Ways (Margolis 2023): Curriculum + multi-task
  - DAgger (Ross et al. 2011): Online imitation → RL fine-tuning
  - BC + PPO warm-start is used in legged_gym / AMP pipelines

Usage:
    python3 src/training/train_hierarchical.py \\
        --expert checkpoints/best/best_model.zip \\
        --total-steps 10000000
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
            episode_length=2000,
            **kwargs
        )
        env = ActionSmoothingWrapper(env, alpha=0.8)
        env = HistoryWrapper(env, history_len=history_len)
        env.reset(seed=rank)
        return env
    return _init


def make_flat_env(rank=0, **kwargs):
    """Environment factory WITHOUT history wrapper (for expert data collection)."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    def _init():
        import sys
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
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


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Collect expert demonstrations
# ══════════════════════════════════════════════════════════════════════

def collect_expert_data(expert_path: str, n_episodes: int = 200,
                        max_steps_per_ep: int = 2000, history_len: int = 16,
                        vec_normalize_path: str | None = None, device: str = "cpu",
                        n_collect_envs: int = 4):
    """Roll out expert policy and collect (obs_history, action) pairs.

    The expert is an MLP policy (no history wrapper needed), but we
    collect overlapping observation windows of length `history_len` so
    the BC-trained transformer sees proper temporal context.

    Returns:
        history_data: np.ndarray (N, history_len * obs_dim)
        act_data: np.ndarray (N, act_dim)
    """
    from collections import deque
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    project_root = str(Path(__file__).resolve().parent.parent.parent)

    # Expert rollout is sequential single-env inference — CPU avoids GPU
    # transfer overhead and is typically faster here.
    expert = PPO.load(expert_path, device=device)

    if vec_normalize_path is not None:
        norm_path = vec_normalize_path
    else:
        norm_path = str(Path(project_root) / "checkpoints" / "vec_normalize.pkl")
    # Phase 1 runs entirely on CPU (MuJoCo simulation is CPU-bound).
    # Parallelise collection across multiple environments to cut wall time.
    n_collect_envs = min(n_collect_envs, n_episodes)
    env_fns = [make_flat_env(i) for i in range(n_collect_envs)]
    try:
        base_env = SubprocVecEnv(env_fns)
        print(f"  Using SubprocVecEnv ({n_collect_envs} parallel envs)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        base_env = DummyVecEnv(env_fns)
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, base_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = base_env

    obs_dim = vec_env.observation_space.shape[0]
    n_envs_active = vec_env.num_envs
    history_list = []
    act_list = []

    print(f"[Phase 1] Collecting expert data: {n_episodes} episodes "
          f"({n_envs_active} parallel envs), history_len={history_len}")

    # Initialise per-env observation history buffers.
    obs: np.ndarray = np.asarray(vec_env.reset())  # (n_envs, obs_dim)
    obs_bufs = [deque(maxlen=history_len) for _ in range(n_envs_active)]
    for i in range(n_envs_active):
        for _ in range(history_len):
            obs_bufs[i].append(obs[i].copy())

    eps_done = 0
    while eps_done < n_episodes:
        # Single batched forward pass covers all parallel environments.
        action, _ = expert.predict(obs, deterministic=True)  # (n_envs, act_dim)

        # Record (history, action) for every env before stepping.
        for i in range(n_envs_active):
            history_flat = np.concatenate(list(obs_bufs[i]), axis=0)
            history_list.append(history_flat)
            act_list.append(action[i].copy())

        obs_next_raw, reward, done, info = vec_env.step(action)
        obs_next: np.ndarray = np.asarray(obs_next_raw)

        for i in range(n_envs_active):
            if done[i]:
                eps_done += 1
                # obs_next[i] is the auto-reset initial obs; fill the buffer
                # so stale frames from the finished episode don't leak through.
                for _ in range(history_len):
                    obs_bufs[i].append(obs_next[i].copy())
                if eps_done % 50 == 0 or eps_done == n_episodes:
                    print(f"  Completed {eps_done}/{n_episodes} episodes, "
                          f"{len(history_list)} transitions")
            else:
                obs_bufs[i].append(obs_next[i].copy())

        obs = obs_next

    vec_env.close()

    history_data = np.array(history_list, dtype=np.float32)
    act_data = np.array(act_list, dtype=np.float32)
    print(f"  Total: {len(history_data)} transitions "
          f"(history: {history_data.shape}, act: {act_data.shape})")
    return history_data, act_data


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: Behavioral cloning on the Transformer encoder
# ══════════════════════════════════════════════════════════════════════

def behavioral_cloning_transformer(
    history_data: np.ndarray,
    act_data: np.ndarray,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    history_len: int = 16,
    obs_dim: int = 49,
    act_dim: int = 12,
    bc_epochs: int = 100,
    bc_lr: float = 5e-4,
    bc_batch: int = 256,
    device: str = "auto",
):
    """Train the Transformer feature extractor + action head via BC.

    BC is a supervised task and benefits from GPU acceleration.
    Uses the same TransformerExtractor as the full policy so weights
    can be transferred directly.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  BC device: {device}")
    from src.training.sb3_integration import TransformerExtractor
    from gymnasium import spaces

    # Create a dummy observation space matching the history wrapper output
    dummy_obs_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(history_len * obs_dim,),
        dtype=np.float32,
    )

    # Build the same extractor used by TransformerActorCriticPolicy
    extractor = TransformerExtractor(
        observation_space=dummy_obs_space,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        history_len=history_len,
        obs_dim=obs_dim,
    ).to(device)

    # Simple action head matching the extractor output
    action_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ELU(),
        nn.Linear(d_model, act_dim),
    ).to(device)

    # Orthogonal init
    for m in action_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    all_params = list(extractor.parameters()) + list(action_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=bc_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=bc_epochs)

    dataset = TensorDataset(
        torch.from_numpy(history_data),
        torch.from_numpy(act_data),
    )
    # pin_memory enables async DMA transfers to GPU; workers overlap data
    # prep with GPU compute, improving throughput during BC training.
    _pin = device.startswith("cuda")
    _workers = min(4, os.cpu_count() or 1) if _pin else 0
    loader = DataLoader(
        dataset,
        batch_size=bc_batch,
        shuffle=True,
        drop_last=True,
        pin_memory=_pin,
        num_workers=_workers,
        persistent_workers=_workers > 0,
    )

    print(f"\n[Phase 2] Behavioral cloning (Transformer): "
          f"{bc_epochs} epochs, {len(history_data)} samples")
    n_params = sum(p.numel() for p in all_params)
    print(f"  Extractor + head params: {n_params:,}")

    extractor.train()
    action_head.train()

    for epoch in range(bc_epochs):
        total_loss = 0.0
        n_batches = 0
        for hist_batch, act_batch in loader:
            hist_batch = hist_batch.to(device)
            act_batch = act_batch.to(device)

            features = extractor(hist_batch)
            pred_action = action_head(features)
            loss = nn.functional.mse_loss(pred_action, act_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{bc_epochs} — "
                  f"loss: {avg_loss:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")

    print(f"  BC training complete. Final loss: {avg_loss:.6f}")
    # Move state dict to CPU so weight injection works regardless of PPO device
    return {k: v.cpu() for k, v in extractor.state_dict().items()}


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: Full PPO training with BC warm-start
# ══════════════════════════════════════════════════════════════════════

def train_ppo_hierarchical(args, bc_extractor_state: dict):
    """PPO training with Transformer+MoE policy, warm-started from BC."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv, DummyVecEnv, VecMonitor,
    )
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback,
    )
    from src.training.sb3_integration import (
        TransformerActorCriticPolicy,
        WorldModelCallback,
        CurriculumCallback,
    )
    from src.training.advanced_policy import AdaptiveCurriculum
    from src.training.reward_logger import RewardComponentCallback

    log_dir = Path(getattr(args, 'log_dir', 'logs/training_hierarchical'))
    ckpt_dir = Path(getattr(args, 'ckpt_dir', 'checkpoints/hierarchical'))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # LR schedule with warmup + cosine decay
    def lr_schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        warmup_frac = 0.05
        if progress < warmup_frac:
            return progress / warmup_frac
        decay_progress = (progress - warmup_frac) / (1.0 - warmup_frac)
        return 0.5 * (1.0 + np.cos(np.pi * decay_progress))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Phase 3] PPO + Transformer+MoE training")
    print(f"  Device: {device}")

    print(f"  Creating {args.n_envs} parallel environments with history...")
    try:
        vec_env = SubprocVecEnv(
            [make_env(i, history_len=args.history_len) for i in range(args.n_envs)]
        )
        print("  Using SubprocVecEnv (parallel)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        vec_env = DummyVecEnv(
            [make_env(i, history_len=args.history_len) for i in range(args.n_envs)]
        )
    vec_env = VecMonitor(vec_env, str(log_dir))
    eval_env = DummyVecEnv([make_env(999, history_len=args.history_len)])

    policy_kwargs = dict(
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        history_len=args.history_len,
        obs_dim=49,
    )

    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=vec_env,
        learning_rate=lambda p: 3e-4 * lr_schedule(p),
        n_steps=2048,
        batch_size=256,
        n_epochs=getattr(args, "n_epochs", 5),
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

    # ── Inject BC-trained weights into the feature extractor ──
    print("\n  Injecting BC-trained Transformer weights...")
    extractor = model.policy.features_extractor
    extractor_params = dict(extractor.named_parameters())
    matched = 0
    skipped = 0
    for name, bc_tensor in bc_extractor_state.items():
        if name in extractor_params:
            param = extractor_params[name]
            if param.shape == bc_tensor.shape:
                with torch.no_grad():
                    param.copy_(bc_tensor.to(param.device))
                matched += 1
            else:
                skipped += 1
        else:
            # Also check buffer names
            try:
                parts = name.split(".")
                mod = extractor
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                buf = getattr(mod, parts[-1])
                if hasattr(buf, 'copy_') and buf.shape == bc_tensor.shape:
                    buf.copy_(bc_tensor)
                    matched += 1
            except (AttributeError, RuntimeError):
                skipped += 1
    print(f"  Injected {matched} parameters, skipped {skipped}")

    total_params = sum(p.numel() for p in model.policy.parameters())
    train_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    curriculum = AdaptiveCurriculum(n_envs=args.n_envs)

    class ProgressLogger(BaseCallback):
        """Prints a one-line training summary to console every `log_every` timesteps."""

        def __init__(self, total_steps: int, rollout_size: int, log_every: int = 20_000):
            super().__init__(verbose=0)
            self._total_steps = total_steps
            self._rollout_size = max(rollout_size, 1)
            self._log_every = log_every
            self._ep_rewards: list[float] = []
            self._ep_lengths: list[int] = []
            self._last_log_ts: int = 0

        def _on_step(self) -> bool:
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

    rollout_size = model.n_steps * args.n_envs
    minibatches_per_epoch = rollout_size // model.batch_size
    grad_steps_per_rollout = minibatches_per_epoch * model.n_epochs
    total_rollouts = args.total_steps // rollout_size
    total_grad_steps = total_rollouts * grad_steps_per_rollout

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="hierarchical",
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
            wm_lr=1e-4, wm_coeff=0.1, update_freq=2048,
            verbose=1 if args.verbose else 0,
        ),
        CurriculumCallback(curriculum, verbose=1 if args.verbose else 0),
        ProgressLogger(
            total_steps=args.total_steps,
            rollout_size=rollout_size,
        ),
        RewardComponentCallback(log_dir=str(log_dir), verbose=1),
    ]

    print(f"\n  Starting training for {args.total_steps:,} steps...")
    print(f"  Gradient update budget:")
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

    final_path = str(ckpt_dir / "hierarchical_final")
    model.save(final_path)
    print(f"\n  Training complete. Model saved: {final_path}")

    config = {
        "method": "hierarchical (BC + Transformer+MoE PPO)",
        "expert": args.expert,
        "n_expert_episodes": args.n_expert_episodes,
        "bc_epochs": args.bc_epochs,
        "total_steps": args.total_steps,
        "n_envs": args.n_envs,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_experts": args.n_experts,
        "history_len": args.history_len,
        "device": device,
        "total_params": total_params,
        "trainable_params": train_params,
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    vec_env.close()
    eval_env.close()
    return final_path


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

def train(args):
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(project_root)

    print("=" * 60)
    print("HIERARCHICAL TRAINING: BC → Transformer+MoE PPO")
    print("=" * 60)
    print(f"  Expert: {args.expert}")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, experts={args.n_experts}")
    print(f"  BC epochs={args.bc_epochs}, PPO steps={args.total_steps:,}")
    print("=" * 60)

    # Phase 1: Collect expert demonstrations with observation history
    # Expert rollout uses a single env sequentially — keep on CPU.
    history_data, act_data = collect_expert_data(
        args.expert,
        n_episodes=args.n_expert_episodes,
        max_steps_per_ep=2000,
        history_len=args.history_len,
        vec_normalize_path=getattr(args, 'vec_normalize', None),
        device="cpu",
        n_collect_envs=args.n_collect_envs,
    )

    # Phase 2: Behavioral cloning on the Transformer encoder
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bc_state = behavioral_cloning_transformer(
        history_data, act_data,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        history_len=args.history_len,
        obs_dim=49,
        act_dim=12,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_batch=args.bc_batch,
        device=device,
    )

    # Phase 3: PPO with full Transformer+MoE, warm-started from BC
    final_path = train_ppo_hierarchical(args, bc_state)
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical training: BC → Transformer+MoE PPO for Go1"
    )
    # Expert / data collection
    parser.add_argument("--expert", type=str, required=True,
                        help="Path to expert policy checkpoint (.zip)")
    parser.add_argument("--n-expert-episodes", type=int, default=200)
    # BC phase
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-lr", type=float, default=5e-4)
    parser.add_argument("--bc-batch", type=int, default=256)
    # PPO phase
    parser.add_argument("--total-steps", type=int, default=15_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="PPO gradient update epochs per rollout (default: 15)")
    # Architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=16)
    # System
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/hierarchical",
                        help="Directory to save checkpoints (default: checkpoints/hierarchical)")
    parser.add_argument("--log-dir", type=str, default="logs/training_hierarchical",
                        help="Directory for TensorBoard logs (default: logs/training_hierarchical)")
    parser.add_argument("--vec-normalize", type=str, default=None,
                        help="Path to VecNormalize stats (.pkl) from expert training")
    parser.add_argument("--n-collect-envs", type=int, default=4,
                        help="Parallel envs for Phase 1 expert data collection (default: 4)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
