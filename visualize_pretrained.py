#!/usr/bin/env python3
"""
visualize_pretrained.py
=======================
Load a pretrained legged locomotion policy from MuJoCo Playground and
run it in the MuJoCo passive viewer.  Optionally record an MP4.

Usage
-----
    # Live viewer (loops until you close the window):
    python visualize_pretrained.py

    # Record a 10-second MP4 and exit:
    python visualize_pretrained.py --record --duration 10 --output rollout.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def load_playground_policy():
    """Return (mj_model, mj_data, policy_fn) from MuJoCo Playground.

    MuJoCo Playground ships pretrained locomotion policies.  We attempt to
    load one of the available quadruped / biped environments in order of
    preference.
    """
    try:
        from playground import load  # MuJoCo Playground public API
    except ImportError:
        print(
            "ERROR: `playground` package not found.  "
            "Install it with:  pip install playground",
            file=sys.stderr,
        )
        sys.exit(1)

    # Preferred environments — tried in order.
    preferred_envs = [
        "Go1SanityCheck-JoystickPolicy",
        "Go1-JoystickPolicy",
        "BarkourVelocity-JoystickPolicy",
        "PantherJoystick",
        "Go1JoystickFlatTerrain",
        "Go1Joystick",
    ]

    env = None
    env_name_used = None
    for name in preferred_envs:
        try:
            env = load(name)
            env_name_used = name
            break
        except Exception:
            continue

    # Fallback: try listing available environments and pick the first one.
    if env is None:
        try:
            from playground import list_environments

            available = list_environments()
            if available:
                env = load(available[0])
                env_name_used = available[0]
        except Exception:
            pass

    if env is None:
        print(
            "ERROR: Could not load any pretrained environment from "
            "MuJoCo Playground.  Please check your installation.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded environment: {env_name_used}")
    return env


def run_viewer(env, record: bool, output: str, duration: float, fps: int):
    """Run the policy in the MuJoCo passive viewer or record to MP4."""
    model = env.model
    data = env.data

    if record:
        _record_mp4(env, model, data, output, duration, fps)
    else:
        _launch_live_viewer(env, model, data)


def _launch_live_viewer(env, model, data):
    """Open the passive viewer and loop the policy until the window closes."""
    print("Launching MuJoCo viewer — close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Step the environment (applies the pretrained policy internally)
            try:
                env.step()
            except Exception:
                # Some Playground wrappers expose step differently
                mujoco.mj_step(model, data)

            viewer.sync()

            # Realtime throttle
            elapsed = time.time() - step_start
            dt = model.opt.timestep
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("Viewer closed.")


def _record_mp4(env, model, data, output: str, duration: float, fps: int):
    """Render off-screen and write frames to an MP4 file."""
    try:
        import imageio
    except ImportError:
        print(
            "ERROR: imageio is required for recording.  "
            "Install it with:  pip install imageio imageio-ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)

    width, height = 1280, 720
    renderer = mujoco.Renderer(model, height=height, width=width)

    total_steps = int(duration / model.opt.timestep)
    frame_interval = max(1, int(1.0 / (fps * model.opt.timestep)))

    out_path = Path(output)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264")

    print(f"Recording {duration}s @ {fps} FPS → {out_path}")

    for step_i in range(total_steps):
        try:
            env.step()
        except Exception:
            mujoco.mj_step(model, data)

        if step_i % frame_interval == 0:
            renderer.update_scene(data)
            frame = renderer.render()
            writer.append_data(frame)

    writer.close()
    renderer.close()
    print(f"Saved recording to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise a pretrained locomotion policy from MuJoCo Playground."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record an MP4 instead of launching the live viewer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rollout.mp4",
        help="Output MP4 path (default: rollout.mp4).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording (default: 30).",
    )
    args = parser.parse_args()

    env = load_playground_policy()
    run_viewer(env, record=args.record, output=args.output,
               duration=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()
