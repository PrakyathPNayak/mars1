"""
MARS Robot — Focused Skill Test Runner (trained-policy edition)
================================================================
Each skill runs the ``MiniCheetahEnv`` with its built-in terrain system
and lets the *trained PPO policy* (``checkpoints/best/best_model.zip``)
drive the robot.  The test harness only issues high-level commands
(``vx, vy, wz`` and mode) — the policy is responsible for all locomotion.

  python test_skills.py walk    — flat terrain, walk → run → stop
  python test_skills.py stairs  — stairs_up terrain, walk forward up steps
  python test_skills.py slide   — slope_down terrain, walk forward down slope
  python test_skills.py jump    — flat terrain, jump mode

Keyboard (MuJoCo passive viewer):
  Space  — pause / resume
  Esc    — quit

Design notes
------------
The previous version of this file used bespoke PD-controlled gaits on
hand-written XML scenes (``env_walk.xml``, ``env_stairs.xml``, …). That
bypassed the trained policy entirely and was only a hand-tuned open-loop
controller pretending to be a skill test.  This rewrite:

  * loads the real trained policy (falls back to a stance-holding PD if
    no checkpoint is found),
  * uses ``MiniCheetahEnv`` so observations match what the policy was
    trained on (196-dim, terrain-aware),
  * issues ``set_command(...)`` per skill — the policy does the walking,
  * throttles to real-time for human-watchable playback.
"""

import os
import sys
import time
import argparse
from typing import Callable, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.env.cheetah_env import MiniCheetahEnv, HEIGHT_DEFAULT, HEIGHT_MIN  # noqa: E402
from src.utils.policy_loader import load_policy_for_inference             # noqa: E402

CTRL_HZ = 50                       # env runs at 50 Hz
DT_CTRL = 1.0 / CTRL_HZ

# ── rendering helpers ────────────────────────────────────────────────────────


def _hud(phase: str, s: int, n: int, pos: np.ndarray, extra: str = "") -> None:
    bw = 28
    f = int(bw * s / max(n, 1))
    bar = "█" * f + "░" * (bw - f)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    print(
        f"\r [{bar}] {s:4d}/{n} │{phase:28s}│ ({x:+.2f},{y:+.2f},{z:+.2f}) {extra:20s}",
        end="", flush=True,
    )


def _setup_camera(env: MiniCheetahEnv) -> None:
    """Position the passive viewer camera to track the robot's trunk."""
    if env.viewer is None:
        return
    import mujoco
    env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    trunk_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id >= 0:
        env.viewer.cam.trackbodyid = trunk_id
    env.viewer.cam.distance = 3.5
    env.viewer.cam.elevation = -20
    env.viewer.cam.azimuth = 150


# ── core runner ──────────────────────────────────────────────────────────────

# A phase is (name, n_steps, command_fn) where command_fn returns
# (vx, vy, wz, mode, target_height) given (step_in_phase, total_steps, env).
PhaseFn = Callable[[int, int, MiniCheetahEnv], Tuple[float, float, float, str, float]]


def run_skill(
    *,
    title: str,
    terrain_type: str,
    terrain_difficulty: float,
    phases: List[Tuple[str, int, PhaseFn]],
    episode_length: int = 100_000,
    realtime: bool = True,
) -> None:
    """Load the env+policy and drive the phases."""
    print(f"\n{'═' * 70}")
    print(f"  MARS SKILL TEST │ {title}")
    print(f"  Terrain={terrain_type!r}  difficulty={terrain_difficulty:.2f}")
    print(f"{'═' * 70}")

    print("  Loading trained policy …")
    policy, normalize_fn = load_policy_for_inference()
    if policy is None:
        print("  [WARN] No trained policy — robot will hold a stance (no locomotion).")

    env = MiniCheetahEnv(
        render_mode="human",
        use_terrain=(terrain_type != "flat"),
        terrain_type=terrain_type,
        terrain_difficulty=terrain_difficulty,
        episode_length=episode_length,
        randomize_domain=False,
    )

    obs, _ = env.reset(seed=0)
    env.render()
    _setup_camera(env)

    # History-aware policies need their history buffer seeded.
    if policy is not None and hasattr(policy, "reset_history"):
        policy.reset_history(obs)

    total_steps = sum(n for _, n, _ in phases)
    global_s = 0

    try:
        for phase_name, n_phase, cmd_fn in phases:
            print(f"\n  ┌─ {phase_name:28s}  ({n_phase} steps, ~{n_phase / CTRL_HZ:.1f}s)")
            t_start = time.time()
            for ps in range(n_phase):
                if env.viewer is not None and not env.viewer.is_running():
                    print("\n  Viewer closed — aborting.")
                    env.close()
                    return

                vx, vy, wz, mode, target_h = cmd_fn(ps, n_phase, env)
                env.set_command(vx, vy, wz, mode=mode, height=target_h)

                if policy is not None:
                    action, _ = policy.predict(normalize_fn(obs), deterministic=True)
                else:
                    # Fallback: zero delta (policy adds offsets to DEFAULT_STANCE).
                    action = np.zeros(env.action_space.shape, dtype=np.float32)

                obs, reward, done, truncated, info = env.step(action)
                env.render()

                _hud(
                    phase_name, ps + 1, n_phase, env.data.qpos[0:3],
                    extra=f"mode={mode} vx={vx:+.2f}",
                )

                if realtime:
                    lag = (ps + 1) * DT_CTRL - (time.time() - t_start)
                    if lag > 0:
                        time.sleep(lag)

                # Soft-reset if policy somehow terminates mid-phase (e.g. fall).
                # We keep going anyway — this is a visual skill test, not a grade.
                if done or truncated:
                    obs, _ = env.reset(seed=0)
                    if policy is not None and hasattr(policy, "reset_history"):
                        policy.reset_history(obs)

                global_s += 1

            print(f"\n  └─ done ({time.time() - t_start:.1f}s, final x={env.data.qpos[0]:+.2f}m)")

        print(f"\n{'═' * 70}")
        print(f"  COMPLETE  │ final pos: {np.round(env.data.qpos[0:3], 3).tolist()}")
        print(f"{'═' * 70}\n")
        print("  Viewer open — close the window or press Ctrl+C to exit.\n")

        # Idle hold: keep simulating with stand command so the viewer stays alive.
        env.set_command(0.0, 0.0, 0.0, mode="stand", height=HEIGHT_DEFAULT)
        try:
            while env.viewer is not None and env.viewer.is_running():
                if policy is not None:
                    action, _ = policy.predict(normalize_fn(obs), deterministic=True)
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, _, done, truncated, _ = env.step(action)
                env.render()
                if done or truncated:
                    obs, _ = env.reset(seed=0)
                    if policy is not None and hasattr(policy, "reset_history"):
                        policy.reset_history(obs)
                time.sleep(DT_CTRL)
        except KeyboardInterrupt:
            pass
    finally:
        env.close()


# ── skill definitions ────────────────────────────────────────────────────────


def _walk_phase(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Gentle ramp 0.3 → 1.0 m/s over the whole phase.
    vx = 0.3 + (ps / max(n - 1, 1)) * 0.7
    return vx, 0.0, 0.0, "walk", HEIGHT_DEFAULT


def _run_phase(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Run faster: ramp 1.2 → 2.0 m/s.
    vx = 1.2 + (ps / max(n - 1, 1)) * 0.8
    return vx, 0.0, 0.0, "run", HEIGHT_DEFAULT


def _stop_phase(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Decelerate then stand.
    frac = 1.0 - ps / max(n - 1, 1)
    vx = 1.0 * frac
    mode = "walk" if vx > 0.1 else "stand"
    return vx, 0.0, 0.0, mode, HEIGHT_DEFAULT


def test_walk() -> None:
    phases = [
        ("WALK  — warm-up 0.3→1.0 m/s", 250, _walk_phase),
        ("RUN   — sprint 1.2→2.0 m/s", 300, _run_phase),
        ("STOP  — decelerate to stand", 150, _stop_phase),
    ]
    run_skill(
        title="WALK / RUN on flat ground",
        terrain_type="flat",
        terrain_difficulty=0.0,
        phases=phases,
    )


def _stairs_approach(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    return 0.5, 0.0, 0.0, "walk", HEIGHT_DEFAULT


def _stairs_climb(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Slower, steadier — climbing benefits from reduced forward speed.
    return 0.35, 0.0, 0.0, "walk", HEIGHT_DEFAULT


def _stairs_hold(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    return 0.0, 0.0, 0.0, "stand", HEIGHT_DEFAULT


def test_stairs() -> None:
    # The built-in 'stairs_up' terrain in MiniCheetahEnv places the
    # staircase starting ~0m in x with steps ascending in +x.  We spawn
    # on flat ground at the origin and walk forward into the stairs.
    phases = [
        ("APPROACH — flat to step-base", 100, _stairs_approach),
        ("CLIMB    — steady forward",    500, _stairs_climb),
        ("HOLD     — stand on top",      100, _stairs_hold),
    ]
    run_skill(
        title="STAIR CLIMB",
        terrain_type="stairs_up",
        terrain_difficulty=0.3,           # ≈4cm steps (robot-proportional)
        phases=phases,
    )


def _slide_approach(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    return 0.4, 0.0, 0.0, "walk", HEIGHT_DEFAULT


def _slide_descend(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # On a slope_down, moderate forward speed; policy handles balance.
    return 0.5, 0.0, 0.0, "walk", HEIGHT_DEFAULT


def _slide_brake(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    frac = 1.0 - ps / max(n - 1, 1)
    vx = 0.5 * frac
    mode = "walk" if vx > 0.1 else "stand"
    return vx, 0.0, 0.0, mode, HEIGHT_DEFAULT


def test_slide() -> None:
    phases = [
        ("APPROACH — to slope edge",   100, _slide_approach),
        ("DESCEND  — down slope",      450, _slide_descend),
        ("BRAKE    — stop on flat",    150, _slide_brake),
    ]
    run_skill(
        title="DESCEND SLOPE",
        terrain_type="slope_down",
        terrain_difficulty=0.4,           # ≈10° slope, ~0.4m rise
        phases=phases,
    )


def _jump_approach(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Build up forward speed before triggering jump mode.
    vx = 0.5 + (ps / max(n - 1, 1)) * 1.0
    return vx, 0.0, 0.0, "run", HEIGHT_DEFAULT


def _jump_leap(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    # Ask the policy's jump mode to execute its jump trajectory.
    return 1.2, 0.0, 0.0, "jump", HEIGHT_DEFAULT


def _jump_land(ps: int, n: int, env) -> Tuple[float, float, float, str, float]:
    return 0.0, 0.0, 0.0, "stand", HEIGHT_DEFAULT


def test_jump() -> None:
    phases = [
        ("RUN-UP   — build speed",  120, _jump_approach),
        ("LEAP     — jump mode",     80, _jump_leap),
        ("LAND     — settle",       120, _jump_land),
    ]
    run_skill(
        title="JUMP",
        terrain_type="flat",
        terrain_difficulty=0.0,
        phases=phases,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

SKILLS = {
    "walk":   test_walk,
    "stairs": test_stairs,
    "slide":  test_slide,
    "jump":   test_jump,
}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MARS Robot trained-policy skill tests")
    p.add_argument("skill", choices=list(SKILLS.keys()), help="Which skill to test")
    args = p.parse_args()
    SKILLS[args.skill]()
