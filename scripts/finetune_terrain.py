"""
Terrain fine-tuning script for Unitree Go1 locomotion policy.

Purpose
-------
The base PPO policy (checkpoints/best/best_model.zip) was trained entirely on
flat terrain.  It falls on slope_up every time (0/5 survival) and barely moves
on stairs_up (mean_vx≈0.01 m/s).  Root causes:

  1. obs[59] (base_height) is raw MuJoCo z-coord → on stairs the robot reads
     height≈1.5 m (out-of-distribution vs. 0.27 m during flat training).
  2. Policy never saw reward signal for climbing — it just optimises flat walking.

Fix: fine-tune with a terrain curriculum (stairs_up + slope_up + rough) using
a small LR so the flat-walking skill is preserved but terrain generalisation
improves.

Usage
-----
    python3 scripts/finetune_terrain.py --steps 1000000 --n-envs 8

The script writes checkpoints to checkpoints/terrain_ft/ and copies the best
to checkpoints/best/best_model.zip on success.
"""
import os
import sys
import time
import argparse
import shutil
from pathlib import Path

# Ensure project root on path regardless of cwd
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# ─── env factory ────────────────────────────────────────────────────────────

def make_terrain_env(rank: int, terrain_type: str = "random",
                     terrain_difficulty: float = 0.4):
    """Factory for terrain-aware MiniCheetahEnv (196-dim obs, same as base)."""
    def _init():
        import sys
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        import os
        os.chdir(PROJECT_ROOT)
        from src.env.cheetah_env import MiniCheetahEnv
        env = MiniCheetahEnv(
            render_mode="none",
            terrain_type=terrain_type,
            terrain_difficulty=terrain_difficulty,
            use_terrain=True,
            episode_length=1000,
            randomize_domain=True,
        )
        env.reset(seed=rank)
        return env
    return _init


# ─── terrain probe ──────────────────────────────────────────────────────────

def probe_terrain(model, n_episodes: int = 10, max_steps: int = 200,
                  terrain_type: str = "slope_up", difficulty: float = 0.3) -> dict:
    """Quick evaluation on a specific terrain type.  Returns survival rate and
    mean forward speed, used by the agentic loop to decide if more training
    is needed."""
    from src.env.cheetah_env import MiniCheetahEnv
    import numpy as np
    env = MiniCheetahEnv(render_mode="none", terrain_type=terrain_type,
                         terrain_difficulty=difficulty, use_terrain=True,
                         randomize_domain=False)
    survived = 0
    speeds = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_vx = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(action)
            ep_vx.append(abs(float(env.data.qvel[0])))
            if done or trunc:
                break
        else:
            survived += 1
        speeds.append(float(sum(ep_vx) / max(len(ep_vx), 1)))
    env.close()
    return {
        "terrain": terrain_type,
        "difficulty": difficulty,
        "survival_rate": survived / n_episodes,
        "mean_vx": float(sum(speeds) / max(len(speeds), 1)),
    }


# ─── main fine-tune loop ─────────────────────────────────────────────────────

def finetune(args):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = args.resume
    if not os.path.exists(base_ckpt):
        print(f"[ERROR] Base checkpoint not found: {base_ckpt}")
        sys.exit(1)

    print(f"[terrain_ft] Loading base policy: {base_ckpt}")
    print(f"[terrain_ft] Fine-tune LR={args.finetune_lr}, steps={args.steps}, envs={args.n_envs}")

    # ── build envs ──────────────────────────────────────────────────────────
    # Curriculum: mix of flat (20%), rough (20%), slope_up (30%), stairs_up (30%)
    # so the robot sees challenging terrain most of the time but retains flat skill.
    terrain_schedule = (
        [("flat",       0.2)] * max(1, args.n_envs // 5) +
        [("rough",      0.4)] * max(1, args.n_envs // 5) +
        [("slope_up",   0.35)] * max(1, args.n_envs * 3 // 10) +
        [("stairs_up",  0.3)] * max(1, args.n_envs * 3 // 10)
    )
    # Trim/pad to exactly n_envs
    terrain_schedule = terrain_schedule[:args.n_envs]
    while len(terrain_schedule) < args.n_envs:
        terrain_schedule.append(("slope_up", 0.35))

    env_fns = [
        make_terrain_env(rank=i, terrain_type=t, terrain_difficulty=d)
        for i, (t, d) in enumerate(terrain_schedule)
    ]

    VecEnvCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = VecEnvCls(env_fns)
    vec_env = VecMonitor(vec_env)
    # Wrap with fresh VecNormalize — we re-use the base model weights but
    # start a new normaliser because the base ckpt's .pkl often fails to load
    # (recursion warning) and terrain obs stats differ from flat training.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── load base policy ────────────────────────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_model = PPO.load(base_ckpt, device=args.device)

    # Clone policy weights into a new PPO that knows about our envs
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=args.device,
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        learning_rate=args.finetune_lr,
        clip_range=0.1,       # tighter clip to avoid forgetting base skill
        ent_coef=0.005,       # small entropy bonus keeps exploration alive
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=str(checkpoint_dir / "tb"),
    )
    # Transfer base policy weights
    model.policy.load_state_dict(base_model.policy.state_dict(), strict=False)
    print("[terrain_ft] Base policy weights transferred ✓")

    # ── callbacks ───────────────────────────────────────────────────────────
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="terrain_ft",
    )

    class TerrainProbeCallback(BaseCallback):
        """Periodically probe slope_up and stairs_up survival and log it."""
        def __init__(self, probe_interval: int = 100_000, verbose: int = 1):
            super().__init__(verbose)
            self.probe_interval = probe_interval
            self._last_probe = 0
            self.best_combined = 0.0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_probe < self.probe_interval:
                return True
            self._last_probe = self.num_timesteps
            print(f"\n[probe @ {self.num_timesteps:,} steps]")
            # Temporarily unwrap model for inference
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probe_model = PPO.load(base_ckpt, device="cpu")
            probe_model.policy.load_state_dict(
                self.model.policy.state_dict(), strict=False
            )
            r_slope  = probe_terrain(probe_model, n_episodes=5, terrain_type="slope_up",  difficulty=0.3)
            r_stairs = probe_terrain(probe_model, n_episodes=5, terrain_type="stairs_up", difficulty=0.3)
            r_flat   = probe_terrain(probe_model, n_episodes=5, terrain_type="flat",       difficulty=0.0)
            combined = (r_slope["survival_rate"] + r_stairs["survival_rate"]) / 2.0
            print(f"  flat      survival={r_flat['survival_rate']:.2f}  vx={r_flat['mean_vx']:.3f}")
            print(f"  slope_up  survival={r_slope['survival_rate']:.2f}  vx={r_slope['mean_vx']:.3f}")
            print(f"  stairs_up survival={r_stairs['survival_rate']:.2f}  vx={r_stairs['mean_vx']:.3f}")
            print(f"  combined  {combined:.2f}  (best so far: {self.best_combined:.2f})")

            if combined > self.best_combined:
                self.best_combined = combined
                save_path = str(checkpoint_dir / "best_terrain_ft.zip")
                self.model.save(save_path)
                print(f"  ✓ New best saved → {save_path}")
            return True

    probe_cb = TerrainProbeCallback(
        probe_interval=max(100_000 // args.n_envs * args.n_envs, 50_000)
    )

    # ── train ────────────────────────────────────────────────────────────────
    print(f"[terrain_ft] Starting fine-tune: {args.steps:,} steps")
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=[ckpt_cb, probe_cb],
        reset_num_timesteps=True,
        progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"[terrain_ft] Training done in {elapsed/60:.1f} min")

    # ── save final & copy to best ─────────────────────────────────────────
    final_path = str(checkpoint_dir / "terrain_ft_final.zip")
    model.save(final_path)
    print(f"[terrain_ft] Saved final → {final_path}")

    # Find best checkpoint (probe callback may have saved best_terrain_ft.zip)
    best_ft = str(checkpoint_dir / "best_terrain_ft.zip")
    if os.path.exists(best_ft):
        # Devil's advocate check: verify it actually beats the base on slope+stairs
        print("\n[devil's advocate] Verifying fine-tuned model vs base...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ft_model  = PPO.load(best_ft, device="cpu")
            base_m    = PPO.load(base_ckpt, device="cpu")

        for terrain in ["slope_up", "stairs_up", "flat"]:
            r_ft   = probe_terrain(ft_model,  n_episodes=8, terrain_type=terrain, difficulty=0.3)
            r_base = probe_terrain(base_m,    n_episodes=8, terrain_type=terrain, difficulty=0.3)
            delta  = r_ft["survival_rate"] - r_base["survival_rate"]
            symbol = "✓" if delta >= 0 else "✗"
            print(f"  {symbol} {terrain:12s}  base={r_base['survival_rate']:.2f} → ft={r_ft['survival_rate']:.2f}  (Δ={delta:+.2f})")

        r_ft_slope  = probe_terrain(ft_model, n_episodes=8, terrain_type="slope_up",  difficulty=0.3)
        r_ft_stairs = probe_terrain(ft_model, n_episodes=8, terrain_type="stairs_up", difficulty=0.3)
        combined_ft = (r_ft_slope["survival_rate"] + r_ft_stairs["survival_rate"]) / 2.0

        if combined_ft >= 0.5:
            dest = "checkpoints/best/best_model.zip"
            os.makedirs("checkpoints/best", exist_ok=True)
            shutil.copy2(best_ft, dest)
            print(f"\n[terrain_ft] ✓ Fine-tuned model promoted to {dest}  (combined={combined_ft:.2f})")
        else:
            print(f"\n[terrain_ft] ✗ Fine-tuned model not promoted — combined survival={combined_ft:.2f} < 0.5")
            print("  Keeping base model. Suggest more steps or adjusted curriculum.")
    else:
        shutil.copy2(final_path, checkpoint_dir / "best_terrain_ft.zip")
        print("[terrain_ft] No probe-best found; final model saved as best_terrain_ft.zip")

    vec_env.close()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Go1 policy on terrain")
    parser.add_argument("--resume",   default="checkpoints/best/best_model.zip",
                        help="Base checkpoint to fine-tune from")
    parser.add_argument("--steps",    type=int, default=500_000,
                        help="Total fine-tune timesteps (default 500k)")
    parser.add_argument("--n-envs",   type=int, default=4,
                        help="Parallel envs for fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=5e-5,
                        help="Learning rate for fine-tuning (default 5e-5)")
    parser.add_argument("--device",   default="cpu",
                        help="torch device (cpu or cuda)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/terrain_ft",
                        help="Directory for fine-tune checkpoints")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
