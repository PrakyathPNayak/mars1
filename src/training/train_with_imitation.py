"""
Imitation learning + PPO fine-tuning for Unitree Go1 quadruped locomotion.

Three-phase training:
  Phase 1 — Expert data collection: Roll out a pre-trained expert policy
            and record (observation, action) pairs.
  Phase 2 — Behavioral cloning (BC): Supervised pre-training of a new policy
            to mimic the expert.  This gives a warm-start that already knows
            basic locomotion.
  Phase 3 — PPO fine-tuning: Continue training the BC-initialised policy
            with PPO to adapt and improve beyond the expert.

Usage:
    python3 src/training/train_with_imitation.py --expert checkpoints/best/best_model.zip
    python3 src/training/train_with_imitation.py --expert checkpoints/cheetah_final.zip \\
        --bc-epochs 50 --ppo-steps 10000000
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


def make_env(rank=0, **kwargs):
    """Environment factory for vectorized training."""
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


def linear_schedule(initial_value: float):
    """Linear decay from initial_value to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Collect expert demonstrations
# ══════════════════════════════════════════════════════════════════════

def collect_expert_data(expert_path: str, n_episodes: int = 200,
                        max_steps_per_ep: int = 1000, device: str = "cpu"):
    """Roll out expert policy and collect (obs, action) pairs.

    Returns:
        obs_data: np.ndarray of shape (N, obs_dim)
        act_data: np.ndarray of shape (N, act_dim)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    ckpt_dir = Path(project_root) / "checkpoints"

    # Load expert (CPU is typically faster for sequential single-env inference)
    expert = PPO.load(expert_path, device=device)

    # Load VecNormalize if available (for consistent observation normalization)
    norm_path = str(ckpt_dir / "vec_normalize.pkl")
    base_env = DummyVecEnv([make_env(999)])
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, base_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = base_env

    obs_list = []
    act_list = []

    print(f"[Phase 1] Collecting expert data: {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs = vec_env.reset()
        for step in range(max_steps_per_ep):
            action, _ = expert.predict(obs, deterministic=True)
            obs_list.append(obs.squeeze(0).copy())
            act_list.append(action.squeeze(0).copy())
            obs, reward, done, info = vec_env.step(action)
            if done[0]:
                break
        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes, "
                  f"{len(obs_list)} transitions")

    vec_env.close()

    obs_data = np.array(obs_list, dtype=np.float32)
    act_data = np.array(act_list, dtype=np.float32)
    print(f"  Total: {len(obs_data)} transitions "
          f"(obs: {obs_data.shape}, act: {act_data.shape})")
    return obs_data, act_data


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: Behavioral cloning
# ══════════════════════════════════════════════════════════════════════

def behavioral_cloning(obs_data: np.ndarray, act_data: np.ndarray,
                       obs_dim: int, act_dim: int,
                       net_arch: list, bc_epochs: int = 30,
                       bc_lr: float = 1e-3, bc_batch: int = 256,
                       device: str = "auto"):
    """Train a policy network via supervised learning on expert data.

    Returns a state_dict compatible with the SB3 PPO actor network.
    BC is a supervised task and can fully utilise GPU acceleration.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  BC device: {device}")

    # Build a simple MLP matching the PPO actor architecture
    layers = []
    in_dim = obs_dim
    for h in net_arch:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, act_dim))
    bc_net = nn.Sequential(*layers).to(device)

    # Orthogonal initialization (matching PPO convention)
    for m in bc_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    dataset = TensorDataset(
        torch.from_numpy(obs_data),
        torch.from_numpy(act_data),
    )
    loader = DataLoader(dataset, batch_size=bc_batch, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(bc_net.parameters(), lr=bc_lr)
    criterion = nn.MSELoss()

    print(f"\n[Phase 2] Behavioral cloning: {bc_epochs} epochs, "
          f"{len(obs_data)} samples, lr={bc_lr}")

    for epoch in range(bc_epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            pred = bc_net(obs_batch)
            loss = criterion(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{bc_epochs} — loss: {avg_loss:.6f}")

    print(f"  BC training complete. Final loss: {avg_loss:.6f}")
    # Move back to CPU so inject_bc_weights works regardless of PPO model device
    return bc_net.cpu().state_dict()


def inject_bc_weights(model, bc_state_dict, net_arch):
    """Copy BC weights into the SB3 PPO actor (policy) network.

    SB3 PPO MlpPolicy stores the actor as:
        policy.mlp_extractor.policy_net.{0,2,4,...}  (Linear layers)
        policy.action_net                              (final Linear)
    """
    actor_layers = []
    for name, param in model.policy.named_parameters():
        if "mlp_extractor.policy_net" in name or "action_net" in name:
            actor_layers.append((name, param))

    # Map BC sequential indices to actor parameter names
    bc_params = list(bc_state_dict.values())
    if len(bc_params) != len(actor_layers):
        print(f"  [WARN] BC has {len(bc_params)} params, "
              f"actor has {len(actor_layers)}. Skipping injection.")
        return False

    with torch.no_grad():
        for (name, param), bc_p in zip(actor_layers, bc_params):
            if param.shape == bc_p.shape:
                param.copy_(bc_p)
            else:
                print(f"  [WARN] Shape mismatch at {name}: "
                      f"{param.shape} vs {bc_p.shape}. Skipping.")
                return False

    print("  BC weights injected into PPO actor.")
    return True


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: PPO fine-tuning
# ══════════════════════════════════════════════════════════════════════

def train_ppo(model, vec_env, eval_env, args):
    """Standard PPO training loop (shared with train.py)."""
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    from src.training.reward_logger import RewardComponentCallback

    log_dir = Path("logs/training_imitation")
    ckpt_dir = Path("checkpoints/imitation")
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
                        f.write(f"[{time.strftime('%H:%M:%S')}] Step {self.n_calls:>8,d} | "
                                f"timesteps={self.num_timesteps:,}\n")
                except Exception:
                    pass
            return True

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="imitation_ppo",
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

    print(f"\n[Phase 3] PPO fine-tuning: {args.ppo_steps:,} steps")
    model.learn(
        total_timesteps=args.ppo_steps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = str(ckpt_dir / "imitation_final")
    model.save(final_path)
    vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
    print(f"\nTraining complete. Model saved: {final_path}")
    return final_path


# ══════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize,
    )

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    os.chdir(project_root)

    net_arch = [2048, 1024, 512]

    # Resolve device once; pass to all phases
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Phase 1: Collect expert demonstrations ──
    # Expert rollout uses a single env sequentially — CPU avoids GPU transfer
    # overhead and is typically faster here.
    obs_data, act_data = collect_expert_data(
        args.expert,
        n_episodes=args.n_expert_episodes,
        max_steps_per_ep=1000,
        device="cpu",
    )
    obs_dim = obs_data.shape[1]
    act_dim = act_data.shape[1]

    # ── Phase 2: Behavioral cloning (supervised — GPU accelerated) ──
    bc_state_dict = behavioral_cloning(
        obs_data, act_data,
        obs_dim=obs_dim, act_dim=act_dim,
        net_arch=net_arch,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_batch=args.bc_batch,
        device=device,
    )

    # ── Build vectorized environments ──
    log_dir = Path("logs/training_imitation")
    ckpt_dir = Path("checkpoints/imitation")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {args.n_envs} parallel environments...")
    try:
        base_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
        print("  Using SubprocVecEnv (parallel)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        base_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(base_env, str(log_dir))

    # Reuse expert's VecNormalize stats so the BC-pretrained policy sees
    # the same observation distribution it was trained on.
    expert_norm_path = str(Path(project_root) / "checkpoints" / "vec_normalize.pkl")
    if os.path.exists(expert_norm_path):
        vec_env = VecNormalize.load(expert_norm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = False
        print(f"  Loaded expert VecNormalize stats from {expert_norm_path}")
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False,
            clip_obs=100.0, gamma=0.99,
        )
        print("  No expert VecNormalize found — starting fresh")

    eval_base = VecMonitor(DummyVecEnv([make_env(999)]))
    if os.path.exists(expert_norm_path):
        eval_env = VecNormalize.load(expert_norm_path, eval_base)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_base, norm_obs=True, norm_reward=False,
            training=False, clip_obs=100.0, gamma=0.99,
        )

    # ── Create PPO model ──
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ELU,
        ortho_init=True,
        log_std_init=-0.5,
    )

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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        device=args.device,
    )

    # ── Inject BC weights into the PPO actor ──
    print("\nInjecting BC weights into PPO policy...")
    inject_bc_weights(model, bc_state_dict, net_arch)

    # ── Phase 3: PPO fine-tuning ──
    final_path = train_ppo(model, vec_env, eval_env, args)

    # Save config
    config = {
        "method": "imitation_learning + PPO",
        "expert": args.expert,
        "n_expert_episodes": args.n_expert_episodes,
        "bc_epochs": args.bc_epochs,
        "bc_lr": args.bc_lr,
        "ppo_steps": args.ppo_steps,
        "n_envs": args.n_envs,
        "net_arch": str(net_arch),
        "activation_fn": "ELU",
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    vec_env.close()
    eval_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Imitation learning + PPO for Unitree Go1"
    )
    parser.add_argument("--expert", type=str, required=True,
                        help="Path to expert policy checkpoint (.zip)")
    parser.add_argument("--n-expert-episodes", type=int, default=200,
                        help="Number of expert rollout episodes for data collection")
    parser.add_argument("--bc-epochs", type=int, default=30,
                        help="Behavioral cloning training epochs")
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="BC learning rate")
    parser.add_argument("--bc-batch", type=int, default=256,
                        help="BC mini-batch size")
    parser.add_argument("--ppo-steps", type=int, default=10_000_000,
                        help="PPO fine-tuning total timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments for PPO")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto (default: auto-detect GPU)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
