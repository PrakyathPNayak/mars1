#!/usr/bin/env python3
"""
Visual Tour Tester — curated run of the Go1's best-performing terrains.

Cycles through terrains where the current checkpoint performs well, with full
live keyboard control, a terminal HUD, and per-episode summaries.

Usage
-----
    # Full tour (default) — cycles all best terrains, keyboard-controlled:
    python terrain_testing/scripts/visual_test.py

    # Single terrain, stay there for multiple episodes:
    python terrain_testing/scripts/visual_test.py --terrain pyramid_stairs --episodes 5

    # Lock difficulty (no randomization per episode):
    python terrain_testing/scripts/visual_test.py --terrain sloped_terrain --difficulty 0.3 --fixed-difficulty

    # List the curated terrain set and exit:
    python terrain_testing/scripts/visual_test.py --list

    # Scripted auto-walk (no pynput needed, good for headless sanity-check):
    python terrain_testing/scripts/visual_test.py --auto

    # Custom checkpoint / vecnorm:
    python terrain_testing/scripts/visual_test.py --checkpoint path/to/model.zip

Keyboard controls
-----------------
    W/↑  Forward      S/↓  Backward     A/←  Strafe L   D/→  Strafe R
    Q    Turn left    E    Turn right
    SHIFT+dir → run speed    CTRL → toggle crouch    J → jump
    SPACE → stop/stand       1/2/3 → walk/trot/run mode
    N → skip to next terrain   R → restart current episode
    ESC / Ctrl+C → quit
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_TESTING = os.path.dirname(_HERE)
_REPO_ROOT      = os.path.dirname(_TERRAIN_TESTING)

for _p in [_TERRAIN_TESTING, _REPO_ROOT, os.path.join(_REPO_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stable_baselines3 import PPO
from maps.map_registry import list_terrains
from envs.base_terrain_wrapper import BaseTerrainWrapper
from src.control.keyboard_controller import KeyboardController

# ── Curated "best-performing" terrain list ─────────────────────────────────────
# Selected based on ft_v2 probe results (combined≈0.43) and manual testing.
# Format: (terrain_name, difficulty, skill_lock, description)
BEST_TERRAINS = [
    ("flat",              0.0,  "walk", "Flat baseline — robot should walk freely"),
    ("pyramid_stairs",    0.25, "walk", "Pyramid stairs — model's strongest non-flat terrain (60% survival)"),
    ("frozen_lake",       0.4,  "walk", "Slippery ice — reduced friction, 200+ step survival"),
    ("dreamwaq_rough",    0.30, "walk", "Dual-scale rough ground (DreamWaQ paper)"),
    ("rma_rough",         0.30, "walk", "Gaussian noise rough terrain (RMA paper, A1 validated)"),
    ("wave_terrain",      0.35, "walk", "Sinusoidal undulations — gentle rolling hills"),
    ("sloped_terrain",    0.25, "walk", "Uniform slope — tests terrain-relative orientation"),
    ("anymal_rough",      0.30, "walk", "ANYmal rough ground — moderate coarse noise"),
    ("discrete_obstacles",0.20, "walk", "Small rectangular blocks 2–10 cm"),
]

TERRAIN_NAMES = [t[0] for t in BEST_TERRAINS]

# Default checkpoint locations
_DEFAULT_CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "best", "best_model.zip")
_DEFAULT_VECNORM    = os.path.join(_REPO_ROOT, "checkpoints", "vec_normalize.pkl")

# ── Extended keyboard controller with tour-navigation keys ────────────────────

class TourController(KeyboardController):
    """KeyboardController extended with N (next terrain), R (restart), ESC (quit)."""

    def __init__(self):
        super().__init__()
        self._ev_next    = threading.Event()
        self._ev_restart = threading.Event()
        self._ev_quit    = threading.Event()

    # Override the parent mode-key handler to intercept N / R / ESC.
    def _handle_mode_keys(self, key_str):
        super()._handle_mode_keys(key_str)
        if key_str == "n":
            self._ev_next.set()
        elif key_str == "r":
            self._ev_restart.set()
        elif key_str in ("escape", "esc"):
            self._ev_quit.set()

    # Convenience checks (non-blocking).
    def next_requested(self)    -> bool: return self._ev_next.is_set()
    def restart_requested(self) -> bool: return self._ev_restart.is_set()
    def quit_requested(self)    -> bool: return self._ev_quit.is_set()

    def clear_navigation(self):
        """Clear N / R events after acting on them."""
        self._ev_next.clear()
        self._ev_restart.clear()

# ── Policy loader (identical pattern to run_single_terrain.py) ─────────────────

def load_policy(checkpoint_path: str, vecnorm_path: Optional[str] = None):
    """Load SB3 PPO checkpoint, wrapping with VecNormalize if available."""
    print(f"  Loading checkpoint : {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    _vn = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        import warnings
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        print(f"  Loading VecNormalize: {vecnorm_path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _vn = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: None]))
            _vn.training = False
            _vn.norm_reward = False
        except Exception as e:
            print(f"  [warn] VecNorm load failed ({e}); using raw observations")
            _vn = None

    class _Policy:
        def predict(self, obs, deterministic=True):
            if _vn is not None:
                obs = _vn.normalize_obs(np.array(obs, dtype=np.float32))
            return model.predict(obs, deterministic=deterministic)

    return _Policy()


def _load_vecnorm(vecnorm_path: str):
    """Load VecNormalize running stats via pickle (avoids DummyVecEnv env-type check)."""
    import pickle
    import warnings
    try:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        vn.training = False
        vn.norm_reward = False
        return vn
    except Exception as e:
        print(f"  [warn] VecNorm pickle load failed ({e}); using raw observations")
        return None

# ── Terminal HUD ───────────────────────────────────────────────────────────────

_HUD_WIDTH = 72

def _hud_line(content: str) -> str:
    """Pad/truncate a line to fit inside the HUD box."""
    return f"║  {content:<{_HUD_WIDTH - 4}}║"

def print_hud(
    terrain_idx: int,
    total_terrains: int,
    terrain_name: str,
    difficulty: float,
    episode: int,
    total_episodes: int,
    step: int,
    max_steps: int,
    reward_total: float,
    reward_per_step: float,
    mode: str,
    vx: float,
    vy: float,
    wz: float,
    alive: bool,
    fps: float,
    is_auto: bool,
):
    status = "ALIVE ✓" if alive else "FELL  ✗"
    ctrl_note = "AUTO scripted walk" if is_auto else "KEYBOARD active"
    bar_len = 20
    filled = int(bar_len * step / max(max_steps, 1))
    progress = "█" * filled + "░" * (bar_len - filled)

    sep = "═" * _HUD_WIDTH
    lines = [
        f"╔{sep}╗",
        _hud_line(f"VISUAL TOUR  │  Terrain {terrain_idx}/{total_terrains}: "
                  f"{terrain_name}  (diff={difficulty:.2f})"),
        _hud_line(f"Episode {episode}/{total_episodes}  │  "
                  f"[{progress}] {step}/{max_steps}  │  FPS: {fps:.0f}"),
        _hud_line(f"R/step: {reward_per_step:+.3f}  │  Total: {reward_total:+.1f}  │  {status}"),
        _hud_line(f"Mode: {mode:<8}  │  cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}"),
        _hud_line(f"Input: {ctrl_note}"),
        f"╠{'═' * _HUD_WIDTH}╣",
        _hud_line("WASD/arrows=move  SHIFT=run  CTRL=crouch  J=jump  SPACE=stop"),
        _hud_line("1=walk  2=trot  3=run  │  N=next terrain  R=restart  ESC=quit"),
        f"╚{sep}╝",
    ]
    # Move cursor up by the number of lines and overwrite
    print(f"\033[{len(lines)}A" + "\n".join(lines), flush=True)

def _clear_hud_space(n_lines: int = 10):
    """Print blank lines to reserve HUD space on first render."""
    print("\n" * n_lines, end="", flush=True)

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    env,
    policy,
    kb: Optional[TourController],
    max_steps: int,
    terrain_idx: int,
    total_terrains: int,
    terrain_name: str,
    difficulty: float,
    episode: int,
    total_episodes: int,
    auto_vx: float = 0.4,        # scripted walk speed when --auto
) -> dict:
    """
    Run one episode with live HUD.

    Returns a dict with episode metrics. Caller checks kb events for
    early-exit (next terrain / quit).
    """
    obs, _ = env.reset()
    total_reward = 0.0
    alive = True
    t_start = time.time()
    hud_tick = 0

    _clear_hud_space(10)

    for step in range(1, max_steps + 1):
        # ── Get command ───────────────────────────────────────────────────────
        if kb is not None and kb.active:
            vx, vy, wz, mode = kb.get_command()
            # Sanitize crouch → stand (not a valid SKILL_MODE)
            if mode == "crouch":
                mode = "stand"
        else:
            # Auto scripted: walk forward
            vx, vy, wz, mode = auto_vx, 0.0, 0.0, "walk"

        env.unwrapped.set_command(vx, vy, wz, mode)

        # ── Policy predict ────────────────────────────────────────────────────
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated:
            alive = False

        # ── HUD update every ~10 steps ────────────────────────────────────────
        hud_tick += 1
        if hud_tick >= 10:
            hud_tick = 0
            elapsed = time.time() - t_start
            fps = step / max(elapsed, 1e-6)
            print_hud(
                terrain_idx=terrain_idx,
                total_terrains=total_terrains,
                terrain_name=terrain_name,
                difficulty=difficulty,
                episode=episode,
                total_episodes=total_episodes,
                step=step,
                max_steps=max_steps,
                reward_total=total_reward,
                reward_per_step=total_reward / step,
                mode=mode,
                vx=vx, vy=vy, wz=wz,
                alive=alive,
                fps=fps,
                is_auto=(kb is None or not kb.active),
            )

        # ── Navigation events ─────────────────────────────────────────────────
        if kb is not None:
            if kb.next_requested() or kb.quit_requested():
                break   # caller handles these
            if kb.restart_requested():
                kb.clear_navigation()
                break   # same as episode end → will loop to next episode

        if terminated or truncated:
            break

    elapsed = time.time() - t_start
    return {
        "steps":          step,
        "total_reward":   total_reward,
        "reward_per_step": total_reward / max(step, 1),
        "survived":       alive and not terminated,
        "elapsed_s":      elapsed,
        "fps":            step / max(elapsed, 1e-6),
    }

# ── Per-terrain summary ────────────────────────────────────────────────────────

def print_episode_summary(terrain_name: str, episode: int, m: dict):
    status = "SURVIVED" if m["survived"] else "FELL"
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [{status}]  {terrain_name}  ep {episode}  "
          f"steps={m['steps']}  "
          f"R/step={m['reward_per_step']:+.3f}  "
          f"total={m['total_reward']:+.1f}  "
          f"fps={m['fps']:.0f}")
    print(sep)

def print_terrain_summary(terrain_name: str, all_metrics: list):
    n = len(all_metrics)
    survived = sum(m["survived"] for m in all_metrics)
    avg_reward = np.mean([m["reward_per_step"] for m in all_metrics])
    avg_steps  = np.mean([m["steps"] for m in all_metrics])
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  TERRAIN COMPLETE: {terrain_name}")
    print(f"  Survival  : {survived}/{n}  ({100*survived//max(n,1)}%)")
    print(f"  Avg R/step: {avg_reward:+.4f}")
    print(f"  Avg steps : {avg_steps:.0f}")
    print(f"{sep}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual tour of the Go1's best-performing terrains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--terrain", default=None,
                        choices=list_terrains() + [None],
                        help="Run a single terrain instead of the full tour")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Override difficulty (default: per-terrain recommended)")
    parser.add_argument("--fixed-difficulty", action="store_true",
                        help="Lock difficulty — no per-episode randomization")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per terrain (default 3)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max steps per episode (default 500)")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                        help=f"Path to PPO checkpoint (default: {_DEFAULT_CHECKPOINT})")
    parser.add_argument("--vecnorm", default=None,
                        help="Path to VecNormalize .pkl (auto-detected if not given)")
    parser.add_argument("--auto", action="store_true",
                        help="Scripted auto-walk (no keyboard needed)")
    parser.add_argument("--list", action="store_true",
                        help="List curated best terrains and exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        print("\nCurated best-performing terrains:")
        print(f"  {'Name':<25} {'Diff':>5}  {'Skill':<6}  Description")
        print("  " + "─" * 70)
        for name, diff, skill, desc in BEST_TERRAINS:
            print(f"  {name:<25} {diff:>5.2f}  {skill:<6}  {desc}")
        print()
        return

    # ── Build terrain schedule ────────────────────────────────────────────────
    if args.terrain:
        # Single terrain — find it in best list or use sensible defaults
        match = next((t for t in BEST_TERRAINS if t[0] == args.terrain), None)
        if match:
            _, rec_diff, rec_skill, rec_desc = match
        else:
            rec_diff, rec_skill, rec_desc = 0.3, "walk", "(custom)"
        schedule = [(
            args.terrain,
            args.difficulty if args.difficulty is not None else rec_diff,
            rec_skill,
            rec_desc,
        )]
    else:
        schedule = [
            (name, args.difficulty if args.difficulty is not None else diff, skill, desc)
            for name, diff, skill, desc in BEST_TERRAINS
        ]

    # ── Load policy ───────────────────────────────────────────────────────────
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        print(f"[warn] Checkpoint not found: {checkpoint}")
        print("       Running with random actions. Pass --checkpoint to use a model.")
        policy = None
    else:
        # Auto-detect vecnorm beside checkpoint, then project default
        vecnorm_path = args.vecnorm
        if vecnorm_path is None:
            for candidate in [
                os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "vec_normalize.pkl"),
                _DEFAULT_VECNORM,
            ]:
                if os.path.exists(candidate):
                    vecnorm_path = candidate
                    break
        print("\nLoading policy...")
        model = PPO.load(checkpoint, device="cpu")
        _vn = _load_vecnorm(vecnorm_path) if vecnorm_path else None
        if _vn:
            print(f"  VecNormalize loaded : {vecnorm_path}")
        else:
            print("  VecNormalize        : not loaded (raw observations)")

        _model_ref, _vn_ref = model, _vn  # capture for closure below

        class _Policy:
            def predict(self, obs, deterministic=True):
                _obs = _vn_ref.normalize_obs(np.array(obs, dtype=np.float32)) if _vn_ref else obs
                return _model_ref.predict(_obs, deterministic=deterministic)

        policy = _Policy()

    # ── Keyboard controller ───────────────────────────────────────────────────
    kb = None
    if not args.auto:
        kb = TourController()
        kb.start()
        kb.print_bindings()
        print("  N = next terrain    R = restart episode\n")

    # ── Tour loop ─────────────────────────────────────────────────────────────
    n_terrains = len(schedule)
    try:
        for t_idx, (terrain_name, difficulty, skill, description) in enumerate(schedule, 1):

            print(f"\n{'═'*60}")
            print(f"  Terrain {t_idx}/{n_terrains}: {terrain_name}")
            print(f"  Difficulty: {difficulty:.2f}  │  Skill: {skill}")
            print(f"  {description}")
            print(f"{'═'*60}\n")

            env = BaseTerrainWrapper(
                terrain_name=terrain_name,
                difficulty=difficulty,
                fixed_difficulty=args.fixed_difficulty,
                fixed_skill=skill,
                render_mode="human",
                randomize_domain=True,
            )

            all_metrics = []
            for ep in range(1, args.episodes + 1):
                m = run_episode(
                    env=env,
                    policy=policy,
                    kb=kb,
                    max_steps=args.steps,
                    terrain_idx=t_idx,
                    total_terrains=n_terrains,
                    terrain_name=terrain_name,
                    difficulty=difficulty,
                    episode=ep,
                    total_episodes=args.episodes,
                )
                all_metrics.append(m)
                print_episode_summary(terrain_name, ep, m)

                # Check navigation events
                if kb is not None:
                    if kb.quit_requested():
                        print("\n[Tour] Quit requested.")
                        env.close()
                        return
                    if kb.next_requested():
                        kb.clear_navigation()
                        print(f"[Tour] Skipping to next terrain.")
                        break

            print_terrain_summary(terrain_name, all_metrics)
            env.close()

            # Check quit between terrains
            if kb is not None and kb.quit_requested():
                print("\n[Tour] Quit requested.")
                return

            # Pause briefly between terrains so the user can read the summary
            if t_idx < n_terrains:
                print(f"  Next terrain in 2s... (press N to skip immediately)\n")
                for _ in range(20):
                    time.sleep(0.1)
                    if kb is not None and (kb.next_requested() or kb.quit_requested()):
                        break
                if kb is not None:
                    kb.clear_navigation()

    except KeyboardInterrupt:
        print("\n[Tour] Interrupted.")
    finally:
        if kb is not None:
            kb.stop()

    print("\n[Tour] Complete.\n")


if __name__ == "__main__":
    main()
