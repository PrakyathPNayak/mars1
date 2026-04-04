#!/usr/bin/env python3
"""
Interactive visualization and keyboard control for the trained Unitree Go1 policy.

Keyboard input is read from the TERMINAL (not the MuJoCo viewer window).
This means you can type commands in the terminal that launched this script
while watching the robot in the MuJoCo viewer.

Usage:
    python3 scripts/interactive_control.py                          # default best model
    python3 scripts/interactive_control.py --checkpoint checkpoints/cheetah_final.zip
    python3 scripts/interactive_control.py --no-policy              # PD standing only

Controls (toggle on/off, Space to stop all):
    W / ↑        : Forward          S / ↓       : Backward
    A / ←        : Strafe left      D / →       : Strafe right
    Q            : Turn left        E           : Turn right
    J            : Jump             C           : Toggle crouch
    1 / 2 / 3    : Walk / Trot / Run speed
    Space        : Stop all motion
    ESC          : Quit
"""
import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.cheetah_env import MiniCheetahEnv
from src.control.terminal_input import TerminalKeyController, print_terminal_bindings
from src.utils.policy_loader import load_policy_for_inference


def run(checkpoint_path=None, use_policy=True):
    """Main interactive control loop with terminal-based input."""
    if use_policy:
        policy, normalize_fn = load_policy_for_inference(checkpoint_path)
    else:
        policy, normalize_fn = None, lambda obs: obs

    # Create env with render_mode="none" — we manage the viewer ourselves
    env = MiniCheetahEnv(
        render_mode="none",
        randomize_domain=False,
        episode_length=100_000,
    )
    env.randomize_commands = False

    # Terminal-based controller — reads from THIS terminal, not the viewer
    ctrl = TerminalKeyController()
    print_terminal_bindings()

    obs, _ = env.reset()
    env.set_command(0.0, 0.0, 0.0, "stand")
    # Prime the history buffer for history-based (hierarchical) policies.
    if policy is not None and hasattr(policy, 'reset_history'):
        policy.reset_history(obs)

    # Launch MuJoCo viewer WITHOUT key_callback — input comes from terminal
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    # Start terminal key reading AFTER viewer launches
    ctrl.start()

    total_reward = 0.0
    ep_reward = 0.0
    step = 0
    ep_num = 1
    t0 = time.time()

    print("[RUN] Simulation started. Type keys in THIS terminal to control the robot. Ctrl+C to quit.\n")

    try:
        while viewer.is_running() and not ctrl.quit:
            step_start = time.time()

            vx, vy, wz, mode = ctrl.get_command()
            env.set_command(vx, vy, wz, mode)

            # Choose action
            if policy is not None:
                action, _ = policy.predict(normalize_fn(obs), deterministic=True)
            else:
                q_offset = obs[:12]
                action = -q_offset * 0.15
                action = np.clip(action, -0.5, 0.5).astype(np.float32)

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            total_reward += reward
            step += 1

            # Sync viewer with updated physics state
            viewer.sync()

            # Status display every 50 steps (~1 sec at 50Hz)
            if step % 50 == 0:
                elapsed = time.time() - t0
                fps = step / max(elapsed, 1e-6)
                height = info.get("base_height", 0.0)
                sys.stdout.write(
                    f"\r  ep={ep_num} step={step:6d} | "
                    f"cmd=({vx:+.1f},{vy:+.1f},{wz:+.1f}) mode={mode:8s} | "
                    f"rew={ep_reward:+8.1f} h={height:.3f} fps={fps:.0f}  "
                )
                sys.stdout.flush()

            if done or truncated:
                sys.stdout.write(
                    f"\n[RESET] Episode {ep_num} ended (step {step}, reward {ep_reward:.1f}). Resetting...\n"
                )
                sys.stdout.flush()
                obs, _ = env.reset()
                env.set_command(0.0, 0.0, 0.0, "stand")
                ctrl.reset_motion()
                if policy is not None and hasattr(policy, 'reset_history'):
                    policy.reset_history(obs)
                ep_reward = 0.0
                ep_num += 1

            # ── Real-time rate limiting ──
            elapsed_step = time.time() - step_start
            sleep_time = env.dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[QUIT] Stopped by user.")
    finally:
        ctrl.stop()
        viewer.close()
        env.close()
        elapsed = time.time() - t0
        print(f"[DONE] {step} steps in {elapsed:.1f}s, avg reward {total_reward / max(step, 1):.2f}/step")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)

    parser = argparse.ArgumentParser(description="Interactive Unitree Go1 control")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--no-policy", action="store_true", help="Use PD controller only (no learned policy)")
    args = parser.parse_args()

    run(checkpoint_path=args.checkpoint, use_policy=not args.no_policy)
